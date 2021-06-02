import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run RANEDDI.")
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='../Data/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')

    parser.add_argument('--dataset', nargs='?', default='deepddi_data',
                        help='Choose a dataset from {deepddi_data,collected_data}')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=300,
                        help='Number of epoch.')

    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[200]',
                        help='Output sizes of every layer')

    parser.add_argument('--dnn_size', nargs='?', default='[400,200,100,86]',
                        help='The size of DNN predictor.')

    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size.')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 

    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]',
                        help='Regularization')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')

    parser.add_argument('--model_type', nargs='?', default='raneddi',
                        help='default raneddi')
    parser.add_argument('--adj_type', nargs='?', default='si',
                        help='Specify the type of the adjacency (laplacian) matrix from {bi, si}.')
    parser.add_argument('--alg_type', nargs='?', default='bi',
                        help='Specify the type of the graph convolutional layer from {bi, gcn, graphsage}.')
    parser.add_argument('--adj_uni_type', nargs='?', default='sum',
                        help='Specify a loss type (uni, sum).')

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')

    parser.add_argument('--mess_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--save_flag', type=int, default=1,
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')

    parser.add_argument('--margin', type=int, default=1,
                        help='the score margin between pos and neg samples')
    parser.add_argument('--B', type=float, default=35,
                        help='the number of shared relation matrix') 

    return parser.parse_known_args()[0]