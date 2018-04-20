import argparse

from network import Network

def get_args():# {{{
    parser = argparse.ArgumentParser(description='Training of CNN')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-5)
    parser.add_argument('-s', '--step', type=int, default=2000)
    parser.add_argument('-b', '--batch-size', type=int, default=8)
    parser.add_argument('--display-step', type=int, default=20)
    parser.add_argument('--logdir', default='train_log', type=str)
    return parser.parse_args()
# }}}


if __name__ == '__main__':
    args = get_args()   
    print(args)

    network = Network(args)
    network.train()
