import itertools
import pandas as pd
import gate_feature
from data_preprocess import single_generate_graph_adj_and_feature
from sim_processing import sim_thresholding


association = pd.read_csv("MD_A.csv", header=0, index_col=0).values

#non-linear feature
m_threshold = [0.5]  # generate subgraph
d_threshold = [0.5]
epochs=[400]
m_sim = pd.read_csv("m_kernel_normalized.csv", header=None, sep=',').values
d_sim = pd.read_csv("d_kernel_normalized.csv", header=None, sep=',').values

m_embeddings =[]
d_embeddings =[]
for s in itertools.product(m_threshold,d_threshold,epochs):
    # GATE
    print(s[0],s[1])
    m_network = sim_thresholding(m_sim, s[1])
    d_network = sim_thresholding(d_sim, s[0])
    m_adj, m_features = single_generate_graph_adj_and_feature(m_network, association)
    d_adj, d_features = single_generate_graph_adj_and_feature(d_network, association.T)
    m_embeddings = gate_feature.get_gate_feature(m_adj, m_features, s[2], 1)
    d_embeddings = gate_feature.get_gate_feature(d_adj, d_features, s[2], 1)

m_emb = pd.DataFrame(m_embeddings)
# m_emb.to_csv('8-GATE_microbe_feature.csv',header=False,index=False)
d_emb = pd.DataFrame(d_embeddings)
d_emb.to_csv('64_HSIC_GATE_disease_feature.csv',header=False,index=False)


print("Finished")