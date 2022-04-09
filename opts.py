import argparse


def parse_opt():

    parser = argparse.ArgumentParser()


    #learning settings
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001)
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=10)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=250)
    
    # algo settings
    parser.add_argument(
        '--n_clusters',
        type=int,
        default=10)
    parser.add_argument(
        '--embedding_size',
        type=int,
        default=10)
    parser.add_argument(
        '--lambda_pk',
        type=float,
        default=10.0)
    parser.add_argument(
        '--lambda_reg',
        type=float,
        default=0.01)
    parser.add_argument(
        '--p_rate',
        type=float,
        default=1.5) # rate at which the power_k value decreases in each iteration
    parser.add_argument(
        '--power_k',
        type=int,
        default=-3) # initial value of the power in power mean objective
    parser.add_argument(
        '--maxSvalues',
        type=int,
        default=20) #the number of smoother power-mean objectives to use; basically number of annealing steps

    # path settings
    parser.add_argument('--input_path', type=str, default="./inputs/data.npz")
    parser.add_argument('--output_path', type=str, default="./outputs/")
    parser.add_argument('--plot_path', type=str, default="./plots/")


    args = parser.parse_args()

    return args
