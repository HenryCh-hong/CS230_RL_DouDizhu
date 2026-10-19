import argparse

from douzero.dmc.belief import HAND_MODELS

parser = argparse.ArgumentParser(description='DouZero: PyTorch DouDizhu AI')

# General Settings
parser.add_argument('--xpid', default='douzero',
                    help='Experiment id (default: douzero)')
parser.add_argument('--save_interval', default=30, type=int,
                    help='Time interval (in minutes) at which to save the model')    
parser.add_argument('--objective', default='adp', type=str, choices=['adp', 'wp', 'logadp'],
                    help='Use ADP or WP as reward (default: ADP)')    

# Training settings
parser.add_argument('--actor_device_cpu', action='store_true',
                    help='Use CPU as actor device')
parser.add_argument('--gpu_devices', default='0', type=str,
                    help='Which GPUs to be used for training')
parser.add_argument('--num_actor_devices', default=1, type=int,
                    help='The number of devices used for simulation')
parser.add_argument('--num_actors', default=5, type=int,
                    help='The number of actors for each simulation device')
parser.add_argument('--training_device', default='0', type=str,
                    help='The index of the GPU used for training models. `cpu` means using cpu')
parser.add_argument('--load_model', action='store_true',
                    help='Load an existing model')
parser.add_argument('--disable_checkpoint', action='store_true',
                    help='Disable saving checkpoint')
parser.add_argument('--savedir', default='artifacts/runs',
                    help='Root dir where experiment data will be saved (one run directory per --xpid)')

# Hyperparameters
parser.add_argument('--total_frames', default=100000000000, type=int,
                    help='Total environment frames to train for')
parser.add_argument('--exp_epsilon', default=0.01, type=float,
                    help='The probability for exploration')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Learner batch size')
parser.add_argument('--unroll_length', default=100, type=int,
                    help='The unroll length (time dimension)')
parser.add_argument('--num_buffers', default=50, type=int,
                    help='Number of shared-memory buffers')
parser.add_argument('--num_threads', default=4, type=int,
                    help='Number learner threads')
parser.add_argument('--max_grad_norm', default=40., type=float,
                    help='Max norm of gradients')

# Optimizer settings
parser.add_argument('--learning_rate', default=0.0001, type=float,
                    help='Learning rate')
parser.add_argument('--alpha', default=0.99, type=float,
                    help='RMSProp smoothing constant')
parser.add_argument('--momentum', default=0, type=float,
                    help='RMSProp momentum')
parser.add_argument('--epsilon', default=1e-5, type=float,
                    help='RMSProp epsilon')

# CS230 hand-belief study
parser.add_argument('--hand_model', default='baseline', choices=('baseline',) + HAND_MODELS,
                    help='baseline: the original DouZero network, without a belief input. none (M0c), '
                         'next (M1), derived_both (M2c), learned_both (M2) and oracle feed a 150-d '
                         'belief to the Q-network')
parser.add_argument('--predictor_dir', default=None,
                    help='Directory with the frozen <seat>.ckpt hand predictors (next, derived_both, learned_both)')
parser.add_argument('--allow_untrained_predictors', action='store_true',
                    help='Accept predictor files that are not marked as trained (smoke runs only)')
parser.add_argument('--warm_start_dir', default=None,
                    help='DouZero checkpoint directory with <seat>.ckpt files to start the Q-networks from')
parser.add_argument('--freeze_predictor', default=True, action=argparse.BooleanOptionalAction,
                    help='Keep the predictors frozen, as in the main experiment; joint fine-tuning is not implemented')
parser.add_argument('--seed', default=None, type=int,
                    help='Seeds initialization, deals and exploration; drawn at random and recorded when omitted')
parser.add_argument('--allow_dirty', action='store_true',
                    help='Allow training from a git tree with uncommitted changes (recorded in config.json)')
