import gym
from gym_seasonals.envs.seasonals_env import SeasonalsEnv 
from gym_seasonals.envs.seasonals_env import portfolio_var
from gym_seasonals.envs.remote_env import RemoteEnv 
from wrappers import EnvWrap
import os, sys, json
import numpy as np
import csv
from parse import parse
import argparse
import functools
from functools import partial

from tensorforce.agents import DQNAgent, PPOAgent, Agent
from tensorforce.execution.runner import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

import tensorflow as tf
from anyrl.algos import DQN, TFScheduleValue, LinearTFSchedule
from anyrl.envs import BatchedGymEnv
from anyrl.envs.wrappers import BatchedFrameStack
from anyrl.models import MLPDistQNetwork, noisy_net_dense, MLPQNetwork, EpsGreedyQNetwork
from anyrl.rollouts import BatchedPlayer, PrioritizedReplayBuffer, NStepPlayer, BasicPlayer, UniformReplayBuffer
from anyrl.spaces import gym_space_vectorizer


def episode_finished(episode, reward):
    print("Reward: {reward} for episode {ep}".format(
        ep=episode, 
        reward=reward))
    return True


def test(agent, env, start_index=None, 
        end_index=None):

    history = {'States':[], 'Actions':[], 'Score':[]}
    state = env.reset()
    agent.reset()
    if (start_index or end_index) and (start_index > 0 or end_index < 1969):
        env.set_simulation_window(start_index=start_index, end_index=end_index)
        state, terminal, reward_diff = env.execute(env.null_action)
    reward = 0

    while True:
        action = agent.act(state, deterministic=True, independent=True)
        state, terminal, reward_diff = env.execute(action)
        reward += reward_diff
        history['States'].append(state)
        history['Actions'].append(action)
        history['Score'].append(reward)
        if terminal:
            break
    return reward, history


def train(agent, env, num_episodes=1, start_index=None, 
        end_index=None, test_env=None):

    rewards = []
    test_rewards = []
    test_episodes = []

    for episode in range(num_episodes):
        state = env.reset()
        agent.reset()
        if (start_index or end_index) and (start_index > 0 or end_index < 1969):
            env.set_simulation_window(start_index=start_index, end_index=end_index)
            state, terminal, reward_diff = env.execute(env.null_action)
        

        reward = test_reward = 0
        while True:
            action = agent.act(state)
            state, terminal, reward_diff = env.execute(action)
            agent.observe(terminal=terminal, reward=reward_diff)
            reward += reward_diff
            if terminal:
                break
        rewards.append(reward)
        episode_finished(episode, reward)
        if episode % 10 == 0 and test_env:
            test_reward, _ = test(agent, test_env, start_index=(
                test_env.first_trading_day + 252 * 5 ) \
                        if hasattr(test_env, 'first_trading_day') else None)
            test_rewards.append(test_reward)
            test_episodes.append(episode)
    if test_env:
        return rewards, test_episodes, test_rewards
    else:
        return rewards


def loss(score): 
    return (30 - score)


def graph_episode(history, save_path=None):
    import matplotlib.pyplot as plt
    plt.clf()
    plt.subplot(3, 1, 1)
    plt.plot(history['States'])
    plt.subplot(3, 1, 2)
    plt.plot(history['Score'])
    plt.subplot(3, 1, 3)
    plt.plot(history['Actions'])
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()


def iter_group_indices(start, end, half_window):
    for k in range(end - start):
        yield slice(max(start + k, 0), min(start + k + 2 * half_window, end + half_window))

def mean_std_dev(ys):
    mean = np.mean(ys)
    std_dev = np.sqrt(sum([(x - mean)**2 for x in ys])/len(ys))
    return mean, std_dev

def moving_average(ys, half_window=10):
    aves = []
    ave_tops = []
    ave_bottoms = []
    for sl in iter_group_indices(-half_window, len(ys) - half_window, half_window):
        mean, std_dev = mean_std_dev(ys[sl])
        aves.append(mean)
        ave_tops.append(mean + 1.5*std_dev)
        ave_bottoms.append(mean - 1.5*std_dev)
#        aves.append(np.mean(ys[sl]))
#        ave_tops.append(max(ys[sl]))
#        ave_bottoms.append(min(ys[sl]))
    return aves, ave_tops, ave_bottoms



def plot_rewards(rewards, save_dir=None, test_rewards=None, test_episodes=None, ylims=None, half_window=10):
    import matplotlib.pyplot as plt
    plt.clf()

    aves, ave_tops, ave_bottoms = moving_average(rewards, half_window)
    print(aves)
    print(ave_tops)
    print(ave_bottoms)
    plt.plot(rewards, 'bo', markersize=2)
    plt.plot(aves, 'r')
    plt.fill_between(range(len(ave_tops)), ave_tops, ave_bottoms,
                     color='r', alpha=.5)
    if ylims:
        plt.ylim(ylims)
    if test_rewards:
        plt.plot(test_episodes, test_rewards)
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'training.png'), bbox_inches='tight')
        plt.clf()
        for k in range(5):
            plt.plot(rewards[k::5])
            plt.savefig(
                    os.path.join(save_dir, 'training_{}.png'.format(k)), 
                    bbox_inches='tight')
            plt.clf()
    else:
        plt.show()


def geom_mean(*args):
    return int(np.round(np.prod(args) ** (1./len(args))))


def default(o):
    if isinstance(o, np.integer): return int(o)
    elif isinstance(o, np.float): return float(o)
    return o


def setup_agent(states, actions, args, save_dir=None,
        load_dir=None, base_agent_file="agent.json"):

    if save_dir:
        saver = {'directory':save_dir,
            'steps':1000}
    else: saver = None


    with open(base_agent_file, 'r') as fp:
        base_agent = json.load(fp=fp)
    
    preprocessing = [{'type':'running_standardize'}]

    network = [
            {'type':"dense", 'size':args['layer_1_size'], 
                'activation':args['layer_1_activation']},
            {'type':"dense", 'size':args['layer_2_size'],
                'activation':args['layer_2_activation']},
    ]
    if args['has_third_layer']:
        network.append(
            {'type':"dense", 'size':geom_mean(actions['num_actions'], 
                args['layer_2_size']), 'activation':'sigmoid'},
    )
    kwargs=dict(
        states=states,
        actions=actions,
        network=network,
        saver=saver,
        states_preprocessing=preprocessing,
    )
    if base_agent['type'] == 'ppo_agent':
        args['baseline_learning_rate'] = (
            args['learning_rate'] * args['baseline_lr_mult'])
        baseline_optimizer = {
                'type':"multi_step",
                'optimizer':{
                    'type':"adam",
                    'learning_rate':args['baseline_learning_rate']
                },
                'num_steps':5
        }
        step_optimizer = {
            'type':"adam",
            'learning_rate':args['learning_rate']
        }
        baseline = {
            'type':'mlp',
            'sizes':[args['layer_1_size'], geom_mean(args['layer_1_size'], 5), 5]
        }
        kwargs.update(
            dict(
                step_optimizer=step_optimizer,
                baseline=baseline,
                baseline_optimizer=baseline_optimizer
            )
        )
        agent_spec = base_agent
        agent_spec.update(kwargs)
        agent_spec = {k:default(v) for k, v in agent_spec.items()}
        agent = PPOAgent(**{key:value 
            for key, value in agent_spec.items() if key != 'type'})
    
    elif base_agent['type'] == 'dqn_agent':
        actions_exploration = {
            'type':"epsilon_decay",
            'initial_epsilon':float(args['initial_epsilon']),
            'final_epsilon':0.1,
            'timesteps':40000
        }
        optimizer = {
            'type':"adam",
            'learning_rate':float(args['learning_rate'])
        }

        kwargs.update(
            dict(
                target_sync_frequency=int(args['target_sync_frequency']),
                actions_exploration=actions_exploration,
                optimizer=optimizer
            )

        )
        agent_spec = base_agent
        agent_spec.update(kwargs)
        agent_spec = {k:default(v) for k, v in agent_spec.items()}
        agent = DQNAgent(**{key:value 
            for key, value in agent_spec.items() if key != 'type'})
    
    with open(os.path.join(save_dir, "agent.json"), 'w') as f:
        f.write(json.dumps(agent_spec))
    
    if load_dir:
        agent.restore_model(directory=load_dir)
    return agent


def load_agent(agent_folder):
    with open(os.path.join(agent_folder, "agent.json"), 'r') as fp:
        agent_spec = json.load(fp=fp)
    return Agent.from_spec(agent_spec)


class Record:
    def __init__(self):
        self.rewards = []
        self.ep = 0


def handle_ep_with_context(num_episodes, context, ts, rew):
    print("Reward: {} for episode {} at timestep {}".format(rew, context.ep, ts))
    context.rewards.append(rew)
    context.ep += 1
    if context.ep >= num_episodes:
        raise gym.error.Error


class DQNBuilder:

    def __init__(self, env, args):
        self.args = args
        self.env = env

    def build_network(self, sess, name):
        return MLPDistQNetwork(
            sess, 
            self.env.action_space.n, 
            gym_space_vectorizer(self.env.observation_space), 
            name, 51, -10, 10, 
            layer_sizes=[self.args['layer_1_size'], self.args['layer_2_size']],
            dueling=True, 
            dense=partial(noisy_net_dense, sigma0=self.args['sigma0']))

    def finish(self, sess, dqn):
        eps_decay_sched = TFScheduleValue(sess,
                LinearTFSchedule(
                    self.args['exploration_timesteps'],
                    self.args['initial_epsilon'], 
                    self.args['final_epsilon']
                    )
                ) if self.args['epsilon_decay'] else self.args['epsilon']
        return {
            "player": BasicPlayer(
                self.env, EpsGreedyQNetwork(dqn.online_net, eps_decay_sched)),
            "optimize_op": dqn.optimize(learning_rate=self.args['learning_rate']),
            "replay_buffer": UniformReplayBuffer(100000),
        }


class RainbowDQNBuilder(DQNBuilder):
    """
    Rainbow DQN: Noisy double dueling distributional deep q-network 
    with prioritized experience replay, as specified in this paper:
    https://arxiv.org/pdf/1710.02298.pdf

    Should only be used in atari environments, and those with similarly 
    complex input, as the PER slows down training considerably without
    sufficient parralelization
    """

    def build_network(self, sess, name):
        return MLPQNetwork(
            sess,
            self.env.action_space.n,
            gym_space_vectorizer(self.env.observation_space),
            name,
            layer_sizes=[
                    self.args['layer_1_size'],
                    self.args['layer_2_size']]
        )

    def finish(self, sess, dqn):
        env = BatchedGymEnv([[self.env]])
        return {
            "player": NStepPlayer(BatchedPlayer(self.env, dqn.online_net), 3),
            "optimize_op": dqn.optimize(learning_rate=0.002),
            "replay_buffer": PrioritizedReplayBuffer(20000, 0.5, 0.4, epsilon=0.2),
        }


def dqn_experiment(args, env_name, base_agent="agent.json", 
        agent_folder=None, visualize=True, num_episodes=500, rainbow=False):

    tf.reset_default_graph()

    ep_record = Record()
    handle_ep = functools.partial(handle_ep_with_context, num_episodes, ep_record)

    env = gym.wrappers.Monitor(gym.make(env_name), 
            os.path.join(agent_folder, "monitor"))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    builder_class = RainbowDQNBuilder if rainbow else DQNBuilder
    builder = builder_class(env, args)
    with tf.Session(config=config) as sess:
        online = builder.build_network(sess, 'online')
        target = builder.build_network(sess, 'target')
        dqn = DQN(online, target)
        train_kwargs = builder.finish(sess, dqn)
        sess.run(tf.global_variables_initializer())
        try:
            dqn.train(num_steps=3000000,
                      train_interval=4,
                      target_interval=args['target_sync_frequency'],
                      batch_size=32,
                      min_buffer_size=400,
                      handle_ep=handle_ep,
                      **train_kwargs)
        except (gym.error.Error, KeyboardInterrupt):
            plot_rewards(ep_record.rewards, save_dir=agent_folder, ylims=(-500, 250), half_window=round(num_episodes / 20))
            pass
        
        experiment_data = {"final_test_reward":np.mean(ep_record.rewards[-1]),
                "test_average_last_50":np.mean(ep_record.rewards[-50::10]),
                "train_average_last_50":np.mean(ep_record.rewards[-50:]),
                "test_average_last_10":np.mean(ep_record.rewards[-10::5]),
                "train_average_last_10":np.mean(ep_record.rewards[-10:]),
                }
        experiment_data.update(args)
    del builder
    del dqn
    env.close()
    return experiment_data


def experiment(args, env_name, base_agent="agent.json", 
        agent_folder=None, visualize=True, num_episodes=1000):
    
    seasonals = (env_name=="seasonals-v1")

    train_env = OpenAIGym(env_name) \
            if not seasonals else EnvWrap(
                    gym.make('seasonals-v1'), batched=True,
                    subep_len=252, num_subeps=5)
    test_env = OpenAIGym(env_name, monitor_video=1, 
            monitor=os.path.join(agent_folder, "monitor")) \
                if not seasonals else EnvWrap(gym.make('seasonals-v1'))

    agent = setup_agent(train_env.states, train_env.actions, args, 
            save_dir=agent_folder, base_agent_file=base_agent)

    rewards, test_episodes, test_rewards = train(
            agent, train_env, num_episodes=num_episodes, 
            test_env=train_env)
    train_env.close()
    if visualize:
        plot_rewards(rewards, 
                test_episodes=test_episodes,
                test_rewards=test_rewards,
                save_dir=agent_folder)
    reward, history = test(agent, test_env, start_index=(
        test_env.first_trading_day + 252 * 5 if seasonals else None))
    graph_episode(history, 
            save_path=os.path.join(agent_folder, "test.png"))
    test_env.close()
    agent.close()
    experiment_data = {"final_test_reward":reward,
            "test_average_last_50":np.mean(test_rewards[-10:]),
            "train_average_last_50":np.mean(rewards[-50:]),
            "test_average_last_10":np.mean(test_rewards[-2:]),
            "train_average_last_10":np.mean(rewards[-10:]),
            }
    experiment_data.update(args)
    return experiment_data


def write_csv(filename, data):
    exists = os.path.exists(filename)
    with open(filename, 'a' if exists else 'w') as f:
        dw = csv.DictWriter(f, fieldnames=data[0].keys())
        if exists:
            dw.writeheader()
        for row in data:
            dw.writerow(row)


def is_int(s):
    try: 
        int(s)
    except ValueError:
        return False
    return True


def is_float(s):
    try: 
        float(s)
    except ValueError:
        return False
    return True


def hyperparam_search(param_space, env_name="seasonals-v1", 
        num_tests=20, num_episodes=1000, save_folder="./models/", 
        base_agent="agent.json", dqn=False, rainbow=False):
    data = []
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
        first = 0
    else:
        first = 1 + max([int(fname) for fname in os.listdir(
            save_folder) if is_int(fname)]) \
                if len(os.listdir(save_folder)) > 0 else 0
    try:
        for agent_num in range(first, first + num_tests):
            args = {key: np.random.choice(s).item()
                    for key, s in param_space.items()}
            exp = dqn_experiment if dqn else experiment
            experiment_data = exp(args, env_name, 
                agent_folder=os.path.join(
                    save_folder, str(agent_num),
                    ), 
                num_episodes=num_episodes,
                base_agent=base_agent,
                rainbow=rainbow)
            data.append(experiment_data)
    except (gym.error.Error, KeyboardInterrupt):
        write_csv(
                os.path.join(save_folder, "trials.csv"), 
                data)
        raise
    write_csv(
            os.path.join(save_folder, "trials.csv"), 
            data)

if __name__=="__main__":

    parser = argparse.ArgumentParser(
            description="Suite of tools for training reinforcement learning agents")
    subparsers = parser.add_subparsers()
    subparsers.required = True
    subparsers.dest = 'mode'

    parser.add_argument('-f', '--folder', type=str, 
            help="Folder the agent files are stored in (default: ./models)", default="./models")
    parser.add_argument('-d', '--dqn', action="store_true", help="Use DQN agent")
    parser.add_argument('-r', '--rainbow', action="store_true", 
            help="Use rainbow DQN agent")

    parser_hyperopt = subparsers.add_parser('hyperopt')
    parser_train = subparsers.add_parser('train')
    parser_test_remote = subparsers.add_parser('test-remote')

    parser_hyperopt.add_argument("environment", type=str, 
            help="OpenAI Gym environment to train the agent on")
    parser_hyperopt.add_argument("num_agents", type=int, 
            help="Number of agents to be generated and tested")
    parser_hyperopt.add_argument("num_episodes", type=int, 
            help="Number of episodes each agent should be trained on")
    parser_hyperopt.add_argument('-b', '--base-agent', type=str, 
            help="JSON file to load the base agent spec from", default="agent.json")
    parser_hyperopt.add_argument('-hp', '--hyperparam-space', type=str,
            help="JSON file to load the hyperparameter search space from", 
            default="hyperparam_space.json")

    parser_train.add_argument("agent_name", type=str,
            help="Name of the folder the agent is stored in \
                    within the folder specified by --folder (-f)")
    parser_train.add_argument("num_episodes", type=int,
            help="Number of episodes to train on")

    parser_test_remote.add_argument("agent_name", type=str,
            help="Name of the folder the agent is stored in \
                    within the folder specified by --folder (-f)")

    args = parser.parse_args()

    if args.mode == "hyperopt":
        with open(args.hyperparam_space, 'r') as fp:
            spacedict = json.load(fp=fp)
        hyperparam_search(spacedict, env_name=args.environment, 
                num_tests=args.num_agents, num_episodes=args.num_episodes, 
                save_folder=args.folder, base_agent=args.base_agent, 
                dqn=args.dqn,
                rainbow=args.rainbow)

    elif args.mode == 'train':

        seasonals = (args.environment == 'seasonals-v1')
        save_dir = "./models/{}/".format(agent_name)
        train_env = EnvWrap(gym.make('seasonals-v1'), batched=False, 
                subep_len=252, num_subeps=5
                ) if seasonals else OpenAIGym(env_name)
        test_env = EnvWrap(gym.make('seasonals-v1')
                ) if seasonals else OpenAIGym(env_name, 
                        monitor_video=1, 
                        monitor=os.path.join(save_dir, 'monitoring')) 

        agent = setup_agent(train_env.states, train_env.actions, int(layer_1_size), 
                int(layer_2_size), layer_1_activation, layer_2_activation, 
                True if has_third_layer=='True' else False,
                float(learning_rate), float(baseline_learning_rate),
                save_dir=save_dir)
        
        rewards, test_rewards, test_episodes = train(
                agent, train_env, num_episodes=num_episodes)
        agent.close()
        train_env.close()
        plot_rewards(rewards, test_rewards=test_rewards, 
                test_episodes=test_episodes)
        loss, history = test(agent, test_env)
        graph_episode(history)

    elif args.mode == 'test-remote':
        agent_folder = args.folder
        env = EnvWrap(gym.make('seasonals-v1'), 
            batched=True, subep_len=252, num_subeps=5)
        # env = EnvWrap(RemoteEnv(os.environ['REMOTE_HOST']), remote_testing=True)
        agent = load_agent(agent_folder, save_dir="./{}/".format(agent_folder))
        test(agent, env)
        agent.close()
        env.close()
