import logging
import copy
import torch
import numpy as np
import time
from crowd_sim.envs.utils.info import *
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.state import JointState, FullState, ObservableState
from crowd_nav.policy.tlsgan import stateToTrajTensors, TLSGAN
from crowd_sim.envs.utils.action import ActionXY
from crowd_nav.policy.sarl import SARL


class Explorer(object):

    def __init__(self, env, robot, policy_learning, device, train_memory, val_memory, gamma, logger, target_policy, obs_len):
        self.train_memory = train_memory
        self.val_memory = val_memory
        self.obs_len = obs_len
        self.policy_learning = policy_learning
        self.logger = logger

        self.env = env
        self.robot = robot
        self.device = device
        self.gamma = gamma
        self.target_policy = target_policy
        self.target_model = None

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

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
        k = k_val + k_train

        for i in range(k):

            ob = self.env.reset(phase = 'gen')
            robot_state = self.robot.get_full_state()
            done = False
            states = []
            actions = []
            rewards = []
            prev_obs = [JointState(robot_state,ob)]

            # If the policy is TLSGAN, either stay and observe or move according to ORCA
            for j in range(self.obs_len-1):
                if isinstance(self.robot.policy,TLSGAN):
                    #action = ActionXY(0,0)
                    action = self.robot.act(prev_obs,test_policy = True)
                else:
                    action = self.robot.act(prev_obs)
                ob, reward, done, info = self.env.step(action)
                robot_state = self.robot.get_full_state()
                prev_obs.append(JointState(robot_state,ob))

            while not done:
                action = self.robot.act(prev_obs,test_policy=False)

                ob, reward, done, info = self.env.step(action)
                robot_state = self.robot.get_full_state()

                states.append(prev_obs)
                actions.append(action)
                rewards.append(reward)
                prev_obs = prev_obs[1:]
                prev_obs.append(JointState(robot_state,ob))

                if isinstance(info, Danger):
                    too_close += 1
                    min_dist.append(info.min_dist)

            # Render only the first episode
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

        if imitation_learning:

            for i, state in enumerate(states):
                reward = rewards[i]
                trajs, rel_trajs, self_info = stateToTrajTensors([state])
                pushed_state = (trajs.to(self.device), rel_trajs.to(self.device), self_info.to(self.device))
            
                if self.policy_learning:
                    value = [actions[i].vx, actions[i].vy]
                    value = torch.Tensor(value).to(self.device)
                else:
                    value = sum([pow(self.gamma, max(t - i, 0) * self.robot.time_step * self.robot.v_pref) * reward
                             * (1 if t >= i else 0) for t, reward in enumerate(rewards)])
                    value = torch.Tensor([value]).to(self.device)

                memory.push((pushed_state, value))

        else:
            n = len(rewards)
            discounted_return = 0
            gamma_bar = pow(self.gamma, self.robot.time_step * self.robot.v_pref)

            for i, state in enumerate(reversed(states)):
                action = [actions[n-i-1].vx,actions[n-i-1].vy]
                if i==0:
                    discounted_return = rewards[n-1]
                else:
                    discounted_return = rewards[n-i-1] + gamma_bar*discounted_return

                trajs, rel_trajs, self_info = stateToTrajTensors([state])
                pushed_state = (trajs.to(self.device), rel_trajs.to(self.device), self_info.to(self.device))

                pushed_action = torch.Tensor(action).to(self.device)
                pushed_return = torch.Tensor([discounted_return]).to(self.device)

                memory.push((pushed_state, pushed_return, pushed_action))


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
