import gym
import configparser
import argparse

from crowd_nav.policy.do_nothing import DO_NOTHING
from crowd_sim.envs.utils.robot import Robot


parser = argparse.ArgumentParser('Parse configuration file')
parser.add_argument('--config_file', default='env_gen.config',type=str)
parser.add_argument('--num_sims',default=1000,type=int)
parser.add_argument('--save_path', default="dataset_generation", type=str)
parser.add_argument('--max_humans',default=7,type=int)
parser.add_argument('--min_humans',default=4,type=int)

args = parser.parse_args()


env_config = configparser.RawConfigParser()
env_config.read(args.config_file)
env = gym.make('CrowdSim-v0')
env.configure(env_config)

robot = Robot(env_config, 'robot')
policy = DO_NOTHING()
robot.set_policy(policy)
robot.set_test_policy(policy)
env.set_robot(robot)
env.generate_trajectories(args)
