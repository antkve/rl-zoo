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
from anyrl.algos import DQN

from anyrl_builders import DQNBuilder, PPOBuilder


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
        if start_index:
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
        if episode % 10 == 0 and test_env is not None:
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
    std_dev = np.sqrt(sum([(x - mean)**2 for x in ys])/len(ys)) if len(ys) else 0
    return mean, std_dev

def moving_average(ys, half_window=10):
    print(ys)
    aves = []
    ave_tops = []
    ave_bottoms = []
    if len(ys) < half_window:
        return [0], [0], [0], 0
    try:
        for sl in iter_group_indices(-half_window, len(ys) - half_window, half_window):
            mean, std_dev = mean_std_dev(ys[sl])
            aves.append(mean)
            ave_tops.append(mean + 1.5*std_dev)
            ave_bottoms.append(mean - 1.5*std_dev)
#        aves.append(np.mean(ys[sl]))
#        ave_tops.append(max(ys[sl]))
#        ave_bottoms.append(min(ys[sl]))
        return aves, ave_tops, ave_bottoms, std_dev
    except UnboundLocalError:
        return [0], [0], [0], 0



def plot_rewards(rewards, save_dir=None, test_rewards=None, test_episodes=None, ylims=None, half_window=10):
    import matplotlib.pyplot as plt
    plt.clf()

    aves, ave_tops, ave_bottoms, std_dev = moving_average(rewards, half_window)
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
    return aves, std_dev




def default(o):
    if isinstance(o, np.integer): return int(o)
    elif isinstance(o, np.float): return float(o)
    return o

def setup_agent(states, actions, args, save_dir=None,
        load_dir=None, base_agent_file="agent.json"):

    with open(base_agent_file, 'r') as fp:
        base_agent = json.load(fp=fp)

    config_builders = {'ppo_agent':PPOConfig,
            'dqn_agent':DQNConfig}
    agent_type = config_builders[base_agent['type']]
    
    agent_config = AgentConfig(base_agent)

    config_builder = config_builders[base_agent['type']]

    if base_agent['type'] == 'ppo_agent':
        kwargs.update(get_ppo_agent_spec(args))
    elif base_agent['type'] == 'dqn_agent':

        kwargs.update(

        )
    agent_spec = base_agent
    agent_spec.update(kwargs)
    agent_spec = {k:default(v) for k, v in agent_spec.items()}
    agent = agent_type(**{key:value 
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
    def __init__(self, filename, save_rate=20):
        self.rewards = []
        self.ep = 0
        self.solved = False
        self.filename = filename
        self.save_rate = save_rate


def handle_ep_with_context(num_episodes, context, ts, rew):
    print("Reward: {} for episode {} at timestep {} \t\t\
            Last 100 average: {}".format(
                rew, context.ep, ts, np.mean(context.rewards[-100:])))
    context.rewards.append(rew)
    context.ep += 1
    if np.mean(context.rewards[-100:]) > 200:
        context.solved = context.ep
        raise gym.error.Error

    if context.rewards[-1] > 30:
        raise gym.error.Error

#    if np.mean(context.rewards[-5:]) > 0:
#        if not os.path.exists(context.filename):
#            os.mkdir(context.filename)
#        tf.train.Saver().save(context.sess, os.path.join(context.filename, 'model.ckpt_{}'.format(context.ep)))
    if context.ep >= num_episodes:
        raise gym.error.Error




def load_anyrl_agent(agent_number):
    with open("trials.csv", 'r') as f:
        dr = csv.DictReader(f)
        data = [row for row in dr]


def anyrl_experiment(args, env_name, agent_folder=None, visualize=True, 
        num_episodes=500, rainbow=False, seasonals=False):

    tf.reset_default_graph()

    seasonals = env_name == 'seasonals-v1'
    if seasonals:
        os.mkdir(agent_folder)
    
    ep_record = Record(filename=os.path.join(agent_folder, "model"))
    handle_ep = functools.partial(handle_ep_with_context, num_episodes, ep_record)

    def video_callable_with_context(context, episode_id):
        if episode_id == 50 or episode_id % 100 == 0:
            return True
        if np.mean(context.rewards[-98:]) > 200:
            return True
        return False

    env = gym.wrappers.Monitor(
            gym.make(env_name), 
            os.path.join(agent_folder, "monitor"), 
            video_callable=functools.partial(
                video_callable_with_context, ep_record
                )
            ) if not seasonals  else EnvWrap(
                gym.make('seasonals-v1'))

    env.set_simulation_window(start_index=1500, end_index=1968)
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
                      min_buffer_size=200,
                      handle_ep=handle_ep,
                      **train_kwargs)
        except gym.error.Error:
            aves, std_dev = plot_rewards(ep_record.rewards, 
                    save_dir=agent_folder, ylims=((-500, 250) 
                        if not seasonals else None), 
                    half_window=round(num_episodes / 20))
            pass
        except KeyboardInterrupt:
            aves, std_dev = plot_rewards(ep_record.rewards, 
                    save_dir=agent_folder, ylims=((-500, 250) 
                        if not seasonals else None), 
                    half_window=round(num_episodes / 20))
            raise
        filename = os.path.join(agent_folder, "model")
        if not os.path.exists(filename):
            os.mkdir(filename)
        tf.train.Saver().save(sess, os.path.join(filename, "model-best.ckpt"))
        experiment_data = {"final_test_reward":np.mean(ep_record.rewards[-1]),
                "test_average_last_50":np.mean(ep_record.rewards[-50::10]),
                "train_average_last_50":np.mean(ep_record.rewards[-50:]),
                "test_average_last_10":np.mean(ep_record.rewards[-10::5]),
                "train_average_last_10":np.mean(ep_record.rewards[-10:]),
                "max_score":max(ep_record.rewards)
                }
        experiment_data.update(args)
        experiment_data['solved'] = True \
                if np.mean(ep_record.rewards[-100:]) > 200 \
                else False
        experiment_data['final_std_dev'] = std_dev
    del builder
    del dqn
    env.close()
    experiment_data['agent'] = agent_folder.split('/')[-1]
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
    if exists:
        with open(filename, 'r') as rf:
            dr = csv.DictReader(rf)
            data = [row for row in dr] + data

    with open(filename, 'w') as f:
        dw = csv.DictWriter(f, fieldnames=data[0].keys())
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
        base_agent="agent.json", anyrl=False, rainbow=False):
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
            if anyrl:
                experiment_data = anyrl_experiment(
                        args, env_name, 
                        agent_folder=os.path.join(
                            save_folder, str(agent_num),
                        ),
                        num_episodes=num_episodes,
                        rainbow=rainbow)
            else:
                experiment_data = experiment(
                        args, env_name, 
                        agent_folder=os.path.join(
                            save_folder, str(agent_num),
                        ),
                        num_episodes=num_episodes,
                        base_agent=base_agent)
            data.append(experiment_data)
    except (gym.error.Error, KeyboardInterrupt):
        write_csv(
                os.path.join(save_folder, "trials.csv"), 
                data)
        print("Search terminated and results written.")
        return True
    write_csv(
            os.path.join(save_folder, "trials.csv"), 
            data)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description="Suite of tools for training reinforcement learning agents")
    subparsers = parser.add_subparsers()
    subparsers.required = True
    subparsers.dest = 'mode'

    parser.add_argument('-f', '--folder', type=str, 
            help="Folder the agent files are stored in (default: ./models)", default="./models")
    parser.add_argument('-d', '--anyrl', action="store_true", help="Use DQN agent")
    parser.add_argument('-r', '--rainbow', action="store_true", 
            help="Use rainbow DQN agent")
    parser.add_argument('-s', '--seasonals', action="store_true", 
            help="Use seasonals environment")

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
                anyrl=args.anyrl,
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

        if not args.anyrl:
            agent_folder = args.folder
            env = EnvWrap(gym.make('seasonals-v1'))
            # env = EnvWrap(RemoteEnv(os.environ['REMOTE_HOST']), remote_testing=True)
            agent = load_agent(agent_folder, save_dir="./{}/".format(agent_folder))
            test(agent, env)
            agent.close()
            env.close()
        else:
            online = builder.build_network(sess, 'online')
            target = builder.build_network(sess, 'target')
            dqn = DQN(online, target)
            sess.run(tf.global_variables_initializer())
            with tf.Session(graph=tf.Graph()) as sess:
            sess.run(tf.global_variables_initializer())
            with tf.Session(graph=tf.Graph()) as sess:
