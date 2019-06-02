import sys
import logging
import argparse
import configparser
import os
import shutil
import torch
import gym

#sys.path.append('/home/romain/sgan')
#sys.path.append('/home/romain/CrowdNav')
sys.path.append('/Users/romain/Desktop/MasterCourseProjects/VITA_Project/sgan/')


from crowd_sim.envs.utils.robot import TrajRobot
from crowd_nav.utils.trainer import TrajTrainer, Trainer
from crowd_nav.utils.memory import ReplayMemory
from crowd_nav.utils.explorer import Traj_Explorer, Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.policy.tlsgan import TLSGAN
from crowd_sim.envs.policy.orca import ORCA
from crowd_nav.policy.sarl import SARL




def main():

    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--train_config', type=str, default='configs/train.config')
    parser.add_argument('--output_dir', type=str, default='data')
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--gpu_num', default="0", type=str)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    ########## CONFIGURE PATHS ##########

    make_new_dir = True
    outputs = os.listdir(args.output_dir)

    # Find the 'output' files in output directory
    output_nums = [name[name.find("output")+6:] for name in outputs if 'output' in name and os.path.isdir(os.path.join(args.output_dir,name))]
    num = 0
    if output_nums != [] :
        num = max([int(num) if num != '' else 0 for num in output_nums])
        if num == 0:
            num = ''
        key = input('Continue from last output ? (y/n)')
        if key == 'y' and not args.resume:
            args.output_dir = os.path.join(args.output_dir,"output"+str(num))
            args.env_config = os.path.join(args.output_dir, os.path.basename(args.env_config))
            args.policy_config = os.path.join(args.output_dir, os.path.basename(args.policy_config))
            args.train_config = os.path.join(args.output_dir, os.path.basename(args.train_config))
        else:
            key = input('Overwrite last output directory instead of new ? (y/n)')
            if key == 'y' and not args.resume:
                args.output_dir = os.path.join(args.output_dir,'output'+str(num))
                shutil.rmtree(args.output_dir)
                os.makedirs(args.output_dir)
            else:
                num = num+1 if num else 1
                args.output_dir = os.path.join(args.output_dir,'output'+str(num))
                os.makedirs(args.output_dir)
            shutil.copy(args.env_config, args.output_dir)
            shutil.copy(args.policy_config, args.output_dir)
            shutil.copy(args.train_config, args.output_dir)
    else :
        args.output_dir = os.path.join(args.output_dir,'output')
        os.makedirs(args.output_dir)
        shutil.copy(args.env_config, args.output_dir)
        shutil.copy(args.policy_config, args.output_dir)
        shutil.copy(args.train_config, args.output_dir)

    log_file = os.path.join(args.output_dir, 'output.log')
    il_weight_file = os.path.join(args.output_dir, 'il_model.pth')
    rl_weight_file = os.path.join(args.output_dir, 'rl_model.pth')


    ########## CONFIGURE LOGGING ##########

    mode = 'a' if args.resume else 'w'
    logger = logging.getLogger('train_sgan')
    level = logging.INFO if not args.debug else logging.DEBUG
    logger.setLevel(level)
    file_handler = logging.FileHandler(log_file, mode=mode)
    file_handler.setLevel(level)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s, %(levelname)s: %(message)s')
    stdout_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logger.info('Using device: %s', device)


    ########## CONFIGURE POLICY ##########

    policy = SGAN(device)
    if args.policy_config is None:
        parser.error('Policy config has to be specified for a trainable network')
    policy_config = configparser.RawConfigParser()
    policy_config.read(args.policy_config)

    policy.configure(policy_config)
    policy.set_device(device)


    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    robot = TrajRobot(env_config, 'robot')
    env.set_robot(robot)

    # read training parameters
    if args.train_config is None:
        parser.error('Train config has to be specified for a trainable network')
    train_config = configparser.RawConfigParser()
    train_config.read(args.train_config)
    rl_learning_rate = train_config.getfloat('train', 'rl_learning_rate')
    train_batches = train_config.getint('train', 'train_batches')
    train_episodes = train_config.getint('train', 'train_episodes')
    sample_episodes = train_config.getint('train', 'sample_episodes')
    target_update_interval = train_config.getint('train', 'target_update_interval')
    evaluation_interval = train_config.getint('train', 'evaluation_interval')
    capacity = train_config.getint('train', 'capacity')
    epsilon_start = train_config.getfloat('train', 'epsilon_start')
    epsilon_end = train_config.getfloat('train', 'epsilon_end')
    epsilon_decay = train_config.getfloat('train', 'epsilon_decay')
    checkpoint_interval = train_config.getint('train', 'checkpoint_interval')

    # configure trainer and explorer
    train_memory = ReplayMemory(capacity)
    val_memory = ReplayMemory(capacity)
    model = policy.get_model()
    batch_size = train_config.getint('trainer', 'batch_size')
    trainer = TrajTrainer(model, train_memory, val_memory, device, batch_size, args.output_dir, logger)
    #trainer = Trainer(model, train_memory, device, batch_size)
    explorer = Traj_Explorer(env, robot, policy.policy_learning, device, train_memory, val_memory, policy.gamma,logger, target_policy=policy, obs_len = policy.obs_len)
    #explorer = Explorer(env, robot, device, train_memory, policy.gamma, target_policy=policy)

    # Resume
    if args.resume:
        if not os.path.exists(rl_weight_file):
            logger.error('RL weights does not exist')
        model.load_state_dict(torch.load(rl_weight_file, map_location=device))
        rl_weight_file = os.path.join(args.output_dir, 'resumed_rl_model.pth')
        logger.info('Load reinforcement learning trained weights. Resume training')
    elif os.path.exists(il_weight_file):
        model.load_state_dict(torch.load(il_weight_file, map_location=device))
        logger.info('Load imitation learning trained weights.')
        if robot.visible:
            safety_space = 0
        else:
            safety_space = train_config.getfloat('imitation_learning', 'safety_space')
        il_policy = train_config.get('imitation_learning', 'il_policy')
        il_policy = policy_factory[il_policy]()
        il_policy.multiagent_training = policy.multiagent_training
        il_policy.safety_space = safety_space
        #robot.set_policy(il_policy)
        robot.set_test_policy(il_policy)
        robot.set_policy(policy)
        policy.set_env(env)
        robot.policy.set_epsilon(epsilon_end)

        #explorer.run_k_episodes(1000, 'train', update_memory=True, imitation_learning=True, with_render = False)
        explorer.run_k_episodes(200, 0, 'val', update_memory=True, imitation_learning=True, with_render = True)
        logger.info('Experience training set size: %d/%d', len(train_memory), train_memory.capacity)
    else:
    # Imitation learrning
        train_episodes = train_config.getint('imitation_learning', 'train_episodes')
        validation_episodes = train_config.getint('imitation_learning', 'validation_episodes')
        logger.info('Starting imitation learning on %d training episodes and %d validation episodes',train_episodes,validation_episodes)

        il_policy = train_config.get('imitation_learning', 'il_policy')
        il_epochs = train_config.getint('imitation_learning', 'il_epochs')
        il_learning_rate = train_config.getfloat('imitation_learning', 'il_learning_rate')
        trainer.set_learning_rate(il_learning_rate)
        if robot.visible:
            safety_space = 0
        else:
            safety_space = train_config.getfloat('imitation_learning', 'safety_space')
        il_policy = policy_factory[il_policy]()
        il_policy.multiagent_training = policy.multiagent_training
        il_policy.safety_space = safety_space
        robot.set_test_policy(il_policy)
        robot.set_policy(il_policy)
        robot.set_policy(policy)
        robot.policy.set_epsilon(epsilon_end)
        explorer.run_k_episodes(train_episodes, validation_episodes, 'train', update_memory=True, imitation_learning=True, with_render=False)
        trainer.optimize_epoch(il_epochs, robot, with_validation = True)
        #trainer.optimize_batch(30)
        torch.save(model.state_dict(), il_weight_file)
        logger.info('Finish imitation learning. Weights saved.')
        logger.info('Experience train_set size: %d/%d', len(train_memory), train_memory.capacity)
        logger.info('Experience validation_set size: %d/%d', len(val_memory), val_memory.capacity)
    # Reinforcement learning
    policy.set_env(env)
    robot.set_policy(policy)

    robot.print_info()
    trainer.set_learning_rate(rl_learning_rate)

    explorer.update_target_model(model)
    robot.policy.set_epsilon(epsilon_end)

    episode = 0
    while episode < train_episodes:
        if args.resume:
            epsilon = epsilon_end
        else:
            if episode < epsilon_decay:
                epsilon = epsilon_start + (epsilon_end - epsilon_start) / epsilon_decay * episode
            else:
                epsilon = epsilon_end
        robot.policy.set_epsilon(epsilon)

        # evaluate the model
        if episode % evaluation_interval == 0:
            explorer.run_k_episodes(env.case_size['val'],0, 'val', episode=episode, with_render = False)

        # sample k episodes into memory and optimize over the generated memory
        explorer.run_k_episodes(sample_episodes, 0, 'train', update_memory=True, episode=episode)
        trainer.optimize_batch(train_batches)
        episode += 1

        if episode % target_update_interval == 0:
            explorer.update_target_model(model)

        if episode != 0 and episode % checkpoint_interval == 0:
            torch.save(model.state_dict(), rl_weight_file)

    # final test
    explorer.run_k_episodes(env.case_size['test'], 0,'test', episode=episode)


if __name__ == '__main__':
    main()
