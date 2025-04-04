import tensorflow._api.v2.compat.v1 as tf
from data_preprocess import prepare_graph_data, parse_args
from gate_trainer import GATETrainer
tf.random.set_random_seed(2024)


def get_gate_feature(adj, features, epochs, l):
    tf.random.set_random_seed(2024)

    args = parse_args(epochs=epochs,l=l)
    feature_dim = features.shape[1]
    args.hidden_dims = [feature_dim] + args.hidden_dims

    G, S, R = prepare_graph_data(adj)
    gate_trainer = GATETrainer(args)
    gate_trainer(G, features, S, R)
    embeddings, attention= gate_trainer.infer(G, features, S, R)
    tf.reset_default_graph()
    return embeddings
