import logging
import copy
import torch
import numpy as np
import time
from crowd_sim.envs.utils.info import *
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.state import JointState, FullState, ObservableState
from crowd_nav.policy.frozen import stateToTrajTensors
from crowd_sim.envs.utils.action import ActionXY
from crowd_nav.policy.frozen import Frozen
from crowd_nav.policy.sarl import SARL


class Explorer(object):
    def __init__(self, env, robot, device, memory=None, gamma=None, target_policy=None):
        self.env = env
        self.robot = robot
        self.device = device
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.target_model = None

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    # @profile
    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False, episode=None,
                       print_failure=False, with_render=False):
        self.robot.policy.set_phase(phase)
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        too_close = 0
        min_dist = []
        cumulative_rewards = []
        collision_cases = []
        timeout_cases = []
        for i in range(k):
            ob = self.env.reset(phase)
            done = False
            states = []
            actions = []
            rewards = []
            while not done:
                action = self.robot.act(ob)
                ob, reward, done, info = self.env.step(action)
                states.append(self.robot.policy.last_state)
                actions.append(action)
                rewards.append(reward)

                if isinstance(info, Danger):
                    too_close += 1
                    min_dist.append(info.min_dist)

            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self.env.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')

            if update_memory:
                if isinstance(info, ReachGoal) or isinstance(info, Collision):
                    # only add positive(success) or negative(collision) experience in experience set
                    self.update_memory(states, actions, rewards, imitation_learning)

            cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                           * reward for t, reward in enumerate(rewards)]))

        success_rate = success / k
        collision_rate = collision / k
        assert success + collision + timeout == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}'.
                     format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time,
                            average(cumulative_rewards)))
        if phase in ['val', 'test']:
            total_time = sum(success_times + collision_times + timeout_times) * self.robot.time_step
            logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                         too_close / total_time, average(min_dist))

        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

    def update_memory(self, states, actions, rewards, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i, state in enumerate(states):
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)
                # value = pow(self.gamma, (len(states) - 1 - i) * self.robot.time_step * self.robot.v_pref)
                value = sum([pow(self.gamma, max(t - i, 0) * self.robot.time_step * self.robot.v_pref) * reward
                             * (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            else:
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    next_state = states[i + 1]
                    gamma_bar = pow(self.gamma, self.robot.time_step * self.robot.v_pref)
                    value = reward + gamma_bar * self.target_model(next_state.unsqueeze(0)).data.item()
            value = torch.Tensor([value]).to(self.device)

            # # transform state of different human_num into fixed-size tensor
            # if len(state.size()) == 1:
            #     human_num = 1
            #     feature_size = state.size()[0]
            # else:
            #     human_num, feature_size = state.size()
            # if human_num != 5:
            #     padding = torch.zeros((5 - human_num, feature_size))
            #     state = torch.cat([state, padding])
            self.memory.push((state, value))

class Traj_Explorer(Explorer):

    def __init__(self, env, robot, policy_learning, device, train_memory, val_memory, gamma, logger, target_policy=None, obs_len=8):
        super().__init__(env, robot, device, None, gamma, target_policy)
        self.train_memory = train_memory
        self.val_memory = val_memory
        self.obs_len = obs_len
        self.policy_learning = policy_learning
        self.logger = logger

    def run_k_episodes(self, k_train, k_val, phase, update_memory=False, episode=None, imitation_learning = False,
                       print_failure=False, with_render = False):

        self.robot.policy.set_phase(phase)
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        too_close = 0
        min_dist = []
        cumulative_rewards = []
        collision_cases = []
        timeout_cases = []
        mem_additions = 0
        k = k_val + k_train

        for i in range(k):
            ob = self.env.reset(phase='gen')

            robot_state = self.robot.get_full_state()
            done = False
            states = []
            actions = []
            rewards = []
            prev_obs = [JointState(robot_state,ob)]

            for j in range(self.obs_len-1):
                if isinstance(self.robot.policy,Frozen):
                    #action = ActionXY(0,0)
                    action = self.robot.act(prev_obs,test_policy = True)
                else:
                    action = self.robot.act(prev_obs)
                ob, reward, done, info = self.env.step(action)
                robot_state = self.robot.get_full_state()
                prev_obs.append(JointState(robot_state,ob))


            while not done:
                action = self.robot.act(prev_obs,test_policy=True)
                #action2 = self.robot.act(prev_obs,test_policy=True)
                #print('1',action)
                #print('2',action2)
                #print('______________\n')

                ob, reward, done, info = self.env.step(action)

                robot_state = self.robot.get_full_state()

                # trajs, rel_trajs, self_info = stateToTrajTensors([prev_obs])
                # rel_trajs = rel_trajs.permute(1,0,2,3).contiguous().view(self.obs_len,-1,2)
                # trajs = trajs.permute(1,0,2,3).contiguous().view(self.obs_len,-1,2)
                # num_humans = 5
                # num_batch = 1
                # seq_start_end = torch.IntTensor([(int((num_humans+1)*i),int((num_humans+1)*(i+1))) for i in np.arange(num_batch)]).to(self.device)

                # print(rel_trajs[:,:,:])
                # print(self.robot.policy.model.generator(trajs, rel_trajs, seq_start_end, user_noise=None)[0,:,:])

                states.append(prev_obs)
                actions.append(action)
                rewards.append(reward)
                prev_obs = prev_obs[1:]
                prev_obs.append(JointState(robot_state,ob))

                if isinstance(info, Danger):
                    too_close += 1
                    min_dist.append(info.min_dist)

            if with_render and (i ==0):
                self.env.render(mode='video')

            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self.env.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')

            if update_memory:
                if isinstance(info, ReachGoal) or isinstance(info, Collision):
                    # only add positive(success) or negative(collision) experience in experience set
                    mem_additions +=1
                    if i < k_train:
                        memory = self.train_memory
                    else:
                        memory = self.val_memory
                    self.update_memory(memory, states, actions, rewards, imitation_learning)

            cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                           * reward for t, reward in enumerate(rewards)]))

        success_rate = success / k
        collision_rate = collision / k
        assert success + collision + timeout == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        self.logger.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}'.
                     format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time,
                            average(cumulative_rewards)))
        if phase in ['val', 'test']:
            total_time = sum(success_times + collision_times + timeout_times) * self.robot.time_step
            self.logger.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                         too_close / total_time, average(min_dist))

        if print_failure:
            self.logger.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            self.logger.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

    def update_memory(self, memory, states, actions, rewards, imitation_learning = False):
        if memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i, state in enumerate(states):
            reward = rewards[i]
            trajs, rel_trajs, self_info = stateToTrajTensors([state])
            pushed_state = (trajs.to(self.device), rel_trajs.to(self.device), self_info.to(self.device))

            if imitation_learning:
                if self.policy_learning:
                    value = [actions[i].vx, actions[i].vy]
                    value = torch.Tensor(value).to(self.device)
                else:
                    value = sum([pow(self.gamma, max(t - i, 0) * self.robot.time_step * self.robot.v_pref) * reward
                             * (1 if t >= i else 0) for t, reward in enumerate(rewards)])
                    value = torch.Tensor([value]).to(self.device)


            else:
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    next_state = states[i + 1]
                    gamma_bar = pow(self.gamma, self.robot.time_step * self.robot.v_pref)
                    value = reward + gamma_bar * self.target_model([next_state], to_convert = True).item() # Removed an unsqueeze here
                value = torch.Tensor([value]).to(self.device)

            # # transform state of different human_num into fixed-size tensor
            # if len(state.size()) == 1:
            #     human_num = 1
            #     feature_size = state.size()[0]
            # else:
            #     human_num, feature_size = state.size()
            # if human_num != 5:
            #     padding = torch.zeros((5 - human_num, feature_size))
            #     state = torch.cat([state, padding])

            #if isinstance(self.target_policy,SARL):
            #    state = self.target_policy.transform(state[-1])
            #    memory.push((state, value))
            #else:
            memory.push((pushed_state, value))


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
