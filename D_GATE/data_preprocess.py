import numpy as np
import networkx as nx
import scipy.sparse as sp
import argparse
np.random.seed(2024)



def generate_graph_adj_and_feature(c_network, d_network, c_feature, d_feature):
    np.random.seed(2024)

    c_features = sp.csr_matrix(c_feature).tolil().todense()
    d_features = sp.csr_matrix(d_feature).tolil().todense()

    c_graph = nx.from_numpy_matrix(c_network)
    c_adj = nx.adjacency_matrix(c_graph)
    c_adj = sp.coo_matrix(c_adj)

    d_graph = nx.from_numpy_matrix(d_network)
    d_adj = nx.adjacency_matrix(d_graph)
    d_adj = sp.coo_matrix(d_adj)

    return c_adj, d_adj, c_features, d_features

def single_generate_graph_adj_and_feature(network, feature):
    np.random.seed(2024)

    features = sp.csr_matrix(feature).tolil().todense()

    graph = nx.from_numpy_array(network)
    adj = nx.adjacency_matrix(graph)
    adj = sp.coo_matrix(adj)


    return adj, features

def prepare_graph_data(adj):
    np.random.seed(2024)

    # adapted from preprocess_adj_bias
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    data = adj.tocoo().data
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()
    return (indices, adj.data, adj.shape), adj.row, adj.col


def conver_sparse_tf2np(input):
    np.random.seed(2024)

    # Convert Tensorflow sparse matrix to Numpy sparse matrix
    return [sp.coo_matrix((input[layer][1], (input[layer][0][:, 0], input[layer][0][:, 1])),
                          shape=(input[layer][2][0], input[layer][2][1])) for layer in input]


def parse_args(epochs,l):
    np.random.seed(2024)

    '''
    Parses the arguments.
    '''
    parser = argparse.ArgumentParser(description="Run gate.")

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate. Default is 0.001.')

    parser.add_argument('--n-epochs', default=epochs, type=int,
                        help='Number of epochs')

    parser.add_argument('--hidden-dims', type=list, nargs='+', default=[128,64],
                        help='Number of dimensions.')

    parser.add_argument('--lambda-', default=l, type=float,
                        help='Parameter controlling the contribution of graph structure reconstruction in the loss function.')

    parser.add_argument('--dropout', default=0.5, type=float,
                        help='Dropout.')

    parser.add_argument('--gradient_clipping', default=5.0, type=float,
                        help='gradient clipping')

    return parser.parse_args()