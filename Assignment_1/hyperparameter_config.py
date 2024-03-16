import argparse

def configParse():
    parser = argparse.ArgumentParser(description='Train neural network with specified parameters.')
    parser.add_argument('--wandb_project', type = str, default = 'CS6910_Assignment_1', help = 'project name')
    parser.add_argument('--wandb_entity', type = str, default='CS23M024', help = 'wandb entity')
    parser.add_argument('--dataset', type = str, default = 'fashion_mnist', help = 'dataset')
    parser.add_argument('--epochs', type = int, default = 10, help='epochs')
    parser.add_argument('--batch_size', type = int, default = 64, help='batch size')
    parser.add_argument('--loss', type=str, default = 'cross_entropy', help='loss function')
    parser.add_argument('--optimizer', type=str, default = 'nadam', help='optimizer')
    parser.add_argument('--learning_rate', type=float, default = 1e-3, help='learning rate')
    parser.add_argument('--momentum', type=float, default = 0.9, help='Momentum')
    parser.add_argument('--beta', type=float, default = 0.9, help='beta')
    parser.add_argument('--beta1', type=float, default = 0.9, help='beta1')
    parser.add_argument('--beta2', type=float, default = 0.999, help='beta2')
    parser.add_argument('--epsilon', type=float, default = 1e-8, help='epsilon')
    parser.add_argument('--weight_decay', type=float, default = 0.0, help='weight decay')
    parser.add_argument('--weight_init', type=str, default = "Xavier", help='weight initialization')
    parser.add_argument('--num_layers', type=int, default = 3, help='number of hidden layers')
    parser.add_argument('--hidden_size', type=int, default = 128, help='size of a hidden layer')
    parser.add_argument('--activation', type=str, default = "tanh", help='activation function')
    args = parser.parse_args()

    return args
