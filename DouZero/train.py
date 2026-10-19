import os
import sys

from douzero.dmc import parser, train

if __name__ == '__main__':
    flags = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_devices
    status = train(flags)
    sys.exit(0 if status == 'completed' else 1)  # an interrupted or failed run must not look successful
