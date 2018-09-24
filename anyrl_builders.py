from anyrl.algos import DQN, TFScheduleValue, LinearTFSchedule
from anyrl.envs import BatchedGymEnv
from anyrl.envs.wrappers import BatchedFrameStack
from anyrl.models import MLPDistQNetwork, noisy_net_dense, \
        MLPQNetwork, EpsGreedyQNetwork, RNNCellAC
from anyrl.rollouts import BatchedPlayer, PrioritizedReplayBuffer, \
        NStepPlayer, BasicPlayer, UniformReplayBuffer
from anyrl.spaces import gym_space_vectorizer

class ModelBuilder:
    def __init__(self, env, args):
        self.args = args
        self.env = env

    def build_network(self, sess, name):
        raise NotImplementedError

    def finish(self, sess, dqn):
        raise NotImplementedError

    def build_model(self):
        return 


class DQNBuilder(ModelBuilder):

    def build_network(self, sess, name):
        layer_sizes = [self.args['layer_1_size'], self.args['layer_2_size']]
        if self.args['has_third_layer']:
            layer_sizes.append(
                    geom_mean(self.args['layer_2_size'], self.env.action_space.n)
                    )
        return MLPQNetwork(
            sess,
            self.env.action_space.n,
            gym_space_vectorizer(self.env.observation_space),
            name,
            layer_sizes=layer_sizes)

    def finish(self, sess, dqn, optimize=True):
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
            "optimize_op": dqn.optimize(learning_rate=self.args['learning_rate']) if optimize else None,
            "replay_buffer": UniformReplayBuffer(1000),
        }


class RainbowDQNBuilder(ModelBuilder):
    """
    Rainbow DQN: Noisy double dueling distributional deep q-network 
    with prioritized experience replay, as specified in this paper:
    https://arxiv.org/pdf/1710.02298.pdf

    Should only be used in atari environments, and those with similarly 
    complex input, as the PER slows down training considerably without
    sufficient parallelization
    """

    def build_network(self, sess, name):
        return MLPDistQNetwork(
            sess, 
            self.env.action_space.n, 
            gym_space_vectorizer(self.env.observation_space), 
            name, 51, -10, 10, 
            layer_sizes=layer_sizes,
            dueling=True, 
            dense=partial(noisy_net_dense, sigma0=self.args['sigma0']))


    def finish(self, sess, dqn):
        env = BatchedGymEnv([[self.env]])
        return {
            "player": NStepPlayer(BatchedPlayer(self.env, dqn.online_net), 3),
            "optimize_op": dqn.optimize(learning_rate=0.002),
            "replay_buffer": PrioritizedReplayBuffer(20000, 0.5, 0.4, epsilon=0.2),
        }


class LSTMACBuilder(ModelBuilder):
    """
    Experimental LSTM actor-critic model. 
    Not explicitly supported by anyrl.
    """
    def build_network(self, sess, name):
        return RNNCellAC(sess,
                  self.env.action_space.n,
                  gym_space_vectorizer(self.env.observation_space),
                  make_cell=lambda: tf.contrib.rnn.LSTMCell(self.args['layer_1_size']))

    def finish(self, sess, model):
        eps_decay_sched = TFScheduleValue(sess,
                LinearTFSchedule(
                    self.args['exploration_timesteps'],
                    self.args['initial_epsilon'], 
                    self.args['final_epsilon']
                    )
                ) if self.args['epsilon_decay'] else self.args['epsilon']
        return {
            "player": BasicPlayer(
                self.env, EpsGreedyQNetwork(model.online_net, eps_decay_sched)),
            "optimize_op": dqn.optimize(
                learning_rate=self.args['learning_rate']) if optimize else None,
            "replay_buffer": UniformReplayBuffer(1000),
        }


