import sys
import torch
import torch.nn as nn
import logging
import numpy as np
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL
from crowd_sim.envs.utils.state import JointState

from sgan.models import TrajectoryGenerator

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)

def stateToTrajTensors(trunc_states): # batch * obs_len *

    self_states = [trunc_state[-1].self_state for trunc_state in trunc_states]
    self_info = torch.Tensor([[self_state.px,self_state.py,self_state.vx,self_state.vy, self_state.radius,self_state.gx,self_state.gy,\
                                self_state.v_pref] for self_state in self_states])

    robot_trajs = np.asarray([[np.around([state.self_state.px,state.self_state.py],decimals=4) for state in trunc_state]for trunc_state in trunc_states])
    trajs = np.asarray([[np.concatenate(([robot_trajs[i,j,:]],[np.around([human.px,human.py],decimals=4) for human in state.human_states]),axis=0) for j,state in enumerate(trunc_state)] for i,trunc_state in enumerate(trunc_states)])

    rel_trajs = np.zeros(trajs.shape)
    rel_trajs[:,1:,:,:] = trajs[:,1:,:,:] - trajs[:,:-1,:,:]

    return torch.Tensor(trajs), torch.Tensor(rel_trajs), self_info

class ValueNetwork(nn.Module):
    def __init__(self, generator, obs_len, device, policy_learning, h_dim = 64, info_dim = 8):
        super().__init__()
        
        self.policy_learning = policy_learning
        self.obs_len = obs_len
        self.info_dim = info_dim
        self.encoder = generator.encoder
        self.h_dim = h_dim
        self.device = device
        self.generator = generator.to(self.device)
        self.mlp = mlp(self.encoder.h_dim + self.generator.pool_net.mlp_dim + self.info_dim,[self.h_dim,32,16,1]).to(self.device) # Hardcoded for now
        self.policy_mlp = mlp(self.encoder.h_dim + self.generator.pool_net.mlp_dim + self.info_dim,[self.h_dim,32,16,2]).to(self.device)
        self.test_net = mlp(4*6*2 + info_dim,[self.h_dim,32,2]).to(self.device)

        self.last_params = {}
        self.last_input = None
        self.last_traj_output = None
        for name, p in self.named_parameters():
            self.last_params[name] = p.data

    # State should be a list of Joint-States
    def forward(self, states, to_convert = True):

        if to_convert:
            assert np.asarray(states).ndim == 2
            trunc_states = np.asarray(states)[:,-self.obs_len:]
            num_humans = len(trunc_states[0,0].human_states) # Not flexible for differnet amounts of human

            trajs, rel_trajs, self_info = stateToTrajTensors(trunc_states)

        else:
            trajs = states[0]
            rel_trajs = states[1]
            self_info = states[2]
            assert np.asarray(trajs).ndim == 4
            assert np.asarray(rel_trajs).ndim == 4
            assert np.asarray(self_info).ndim == 2
            assert np.asarray(self_info).shape[1] == self.info_dim
            num_humans = rel_trajs.shape[-2]-1 # Not flexible for differnet amounts of human

        num_batch = trajs.shape[0]

        if self.last_input is None:
            self.last_input = (trajs[0,:,:,:].unsqueeze(0),rel_trajs[0,:,:,:].unsqueeze(0), self_info[0,:].unsqueeze(0))

        #value = self.test_net(torch.cat([rel_trajs.view(-1,6*4*2), self_info.to(self.device)], dim=1))
        #return value

        rel_trajs = rel_trajs.permute(1,0,2,3).contiguous().view(self.obs_len,-1,2).to(self.device)
        trajs = trajs.permute(1,0,2,3).contiguous().view(self.obs_len,-1,2).to(self.device)

        hidden_value = self.encoder(rel_trajs) # 1*(batch*num_humans+1)*64

        seq_start_end = torch.IntTensor([(int((num_humans+1)*i),int((num_humans+1)*(i+1))) for i in np.arange(num_batch)]).to(self.device)

        # print(trajs[0,:,:])
        # print(self.generator(trajs, rel_trajs, seq_start_end, user_noise=None)[0,:,:])

        end_pos = trajs[-1, :, :]

        pool_h = self.generator.pool_net(hidden_value, seq_start_end, end_pos)

        pool_h = pool_h[::num_humans+1,:] # Take all robot pools

        hidden_value = hidden_value[:,::num_humans+1,:] # Take all robot hidden_values

        mlp_decoder_context_input = torch.cat(
            [hidden_value.view(-1, self.h_dim), pool_h.view(-1, self.generator.pool_net.mlp_dim)], dim=1)

        mlp_input = torch.cat((mlp_decoder_context_input, self_info.to(self.device)), dim=1)

        if self.policy_learning:
            value = self.policy_mlp(mlp_input)
        else:
            value = self.mlp(mlp_input)

        return value


class TLSGAN(MultiHumanRL):
    def __init__(self,device):
        super().__init__()
        self.name = 'SGAN'
        self.device = device
        self.policy_learning = False

    def set_generator_parameters(self,config):
        self.obs_len = config.getint('frozen','obs_len')
        self.pred_len = config.getint('frozen','pred_len')
        self.mlp_dim = config.getint('frozen','mlp_dim')
        self.num_layers = config.getint('frozen','num_layers')
        self.pooling_type = config.get('frozen','pooling_type')
        self.pool_every_timestep = config.getboolean('frozen','pool_every_timestep')
        self.dropout = config.getfloat('frozen','dropout')
        self.bottleneck_dim = config.getint('frozen','bottleneck_dim')
        self.neighborhood_size = config.getfloat('frozen','neighborhood_size')
        self.grid_size = config.getint('frozen','grid_size')
        self.batch_norm = config.getboolean('frozen','batch_norm')
        self.multiagent_training = config.getboolean('frozen', 'multiagent_training')

    def configure(self, config):
        self.set_common_parameters(config)

        self.set_generator_parameters(config)

        checkpoint = torch.load(config.get('frozen','model_file'), map_location = 'cpu')

        self.with_pretrained = config.getboolean('frozen','with_pretrained')
        self.frozen_training = config.getboolean('frozen','frozen_training')
        self.policy_learning = config.getboolean('frozen','policy_learning')

        self.generator = TrajectoryGenerator(
            obs_len=self.obs_len,
            pred_len=self.pred_len,
            embedding_dim=checkpoint['hyperparams']['embedding_dim'],
            encoder_h_dim=checkpoint['hyperparams']['encoder_h_dim_g'],
            decoder_h_dim=checkpoint['hyperparams']['decoder_h_dim_g'],
            mlp_dim=self.mlp_dim,
            num_layers=self.num_layers,
            noise_dim=(0,),
            noise_type='gaussian',
            noise_mix_type='ped',
            pooling_type=self.pooling_type,
            pool_every_timestep=self.pool_every_timestep,
            dropout=self.dropout,
            bottleneck_dim=self.bottleneck_dim,
            neighborhood_size=self.neighborhood_size,
            grid_size=self.grid_size,
            batch_norm=self.batch_norm)

        #self.generator.train(not self.frozen_training)
        if self.with_pretrained:
            self.generator.load_state_dict(checkpoint['g_best_state'])
        else:
            self.generator.apply(init_weights)

        self.model = ValueNetwork(self.generator, self.obs_len, self.device, self.policy_learning).to(self.device)
        for p in self.model.parameters():
            p.requires_grad = True
        for p in self.model.generator.parameters():
            p.requires_grad = not self.frozen_training

        logging.info('Policy: {} '.format(self.name))

    def predict(self, states):
        """
        A base class for all methods that takes pairwise joint state as input to value network.
        The input to the value network is always of shape (batch_size, # humans, rotated joint state length)

        states is a list of joint states

        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        if self.reach_destination(states[-1]):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        if self.action_space is None:
            self.build_action_space(states[-1].self_state.v_pref)

        if self.policy_learning:
            next_position = self.model([states]).data.squeeze(0)
            action = ActionXY(next_position[0],next_position[1])
            return action

        else:
            occupancy_maps = None
            probability = np.random.random()
            if self.phase == 'train' and probability < self.epsilon:
                max_action = self.action_space[np.random.choice(len(self.action_space))]
            else:
                self.action_values = list()
                max_value = float('-inf')
                max_action = None
                for action in self.action_space:
                    next_self_state = self.propagate(states[-1].self_state, action) # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta'
                    if self.query_env:
                        next_human_states, reward, done, info = self.env.onestep_lookahead(action)
                    else:
                        next_human_states = [self.propagate(human_state, ActionXY(human_state.vx, human_state.vy))
                                           for human_state in states.human_states] # 'px1', 'py1', 'vx1', 'vy1', 'radius1'
                        reward = self.compute_reward(next_self_state, next_human_states)

                    #batch_next_states = torch.cat([torch.Tensor([next_self_state + next_human_state]).to(self.device)
                    #                              for next_human_state in next_human_states], dim=0)

                    if self.with_om: # Didn't check this 
                        if occupancy_maps is None:
                            occupancy_maps = self.build_occupancy_maps(next_human_states).unsqueeze(0)
                        rotated_batch_input = torch.cat([rotated_batch_input, occupancy_maps], dim=2)


                    # VALUE UPDATE
                    next_state_value = self.model([states + [JointState(next_self_state, next_human_states)]]).data.item()
                    value = reward + pow(self.gamma, self.time_step * states[-1].self_state.v_pref) * next_state_value
                    self.action_values.append(value)
                    if value > max_value:
                        max_value = value
                        max_action = action
                if max_action is None:
                    raise ValueError('Value network is not well trained. ')

                #indexes = sorted(range(len(self.action_values)), key=lambda k: self.action_values[k]) 
                #sorted_actions = sorted(self.action_space, key=lambda k: indexes.index(self.action_space.index(k)))
                #print(set(zip(sorted_actions,sorted(self.action_values))))


        if self.phase == 'train':
            self.last_state = states

        return max_action

