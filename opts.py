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
        default=1.5)
    parser.add_argument(
        '--power_k',
        type=int,
        default=-3)
    parser.add_argument(
        '--maxSvalues',
        type=int,
        default=20)

    # path settings
    parser.add_argument('--input_path', type=str, default="./inputs/data.npz")
    parser.add_argument('--output_path', type=str, default="./outputs/")
    parser.add_argument('--plot_path', type=str, default="./plots/")


    args = parser.parse_args()

    return args