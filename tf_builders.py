
def geom_mean(*args):
    return int(np.round(np.prod(args) ** (1./len(args))))

class AgentConfig:

    def __init__(self, base_agent, states, actions, save_dir=None):

        self.type = base_agent['type']
        self.full_spec = {k:v for k, v in base_agent.items() if k != 'type'}
        if save_dir:
            saver = {'directory':save_dir,
                'steps':1000}
        else: 
            saver = None
        self.full_spec.update({'states':states, 'actions':actions, 'saver':saver})

    def get_type(self):
        return self.type

    def update_with_args(self, args):
        
        preprocessing = [{'type':'running_standardize'}]

        network = [
            {'type':"dense", 'size':args['layer_1_size'], 
                'activation':args['layer_1_activation']},
            {'type':"dense", 'size':args['layer_2_size'],
                'activation':args['layer_2_activation']}
        ]
        if args['has_third_layer']:
            network.append(
                {'type':"dense", 'size':geom_mean(actions['num_actions'], 
                    args['layer_2_size']), 'activation':'sigmoid'}
            )


        self.full_spec.update({
                "target_sync_frequency" : int(args['target_sync_frequency']),
                "actions_exploration" : actions_exploration,
                "optimizer" : optimizer,
                "saver" : saver,
                "states_preprocessing" : preprocessing,
                "network" : network,
            })

        self.full_spec.update(self.setup_special_functions(args))

    def 

    def setup_special_functions(self, args):
        raise NotImplementedError


class PPOConfig(AgentConfig):

    def setup_special_functions(self, args):
        baseline_learning_rate = (
            args['learning_rate'] * args['baseline_lr_mult'])
        baseline_optimizer = {
                'type':"multi_step",
                'optimizer':{
                    'type':"adam",
                    'learning_rate':baseline_learning_rate
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

        return {
            "step_optimizer" : step_optimizer,
            "baseline" : baseline,
            "baseline_optimizer" : baseline_optimizer
            }


class DQNConfig(AgentConfig):

    def setup_special_functions(self, args):
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
        return {
                "actions_exploration" : actions_exploration,
                "optimizer" : optimizer
                }
