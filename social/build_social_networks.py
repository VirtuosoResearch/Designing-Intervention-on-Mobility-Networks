from itertools import count
from scipy import sparse
import numpy as np
import pandas as pd
import networkx as nx

num_nodes = {
    'facebook': 4039,
    'enron': 36692,
    'stanford': 281904,
    'google': 916428,
    'ptop': 10879,
    'advogato': 6542,
    'darkweb': 7178,
    'bitcoin': 3783
}

data_dirs = {
    'facebook': '../data/social_networks/facebook_combined.txt',
    'enron': '../data/social_networks/email-Enron.txt',
    'stanford': '../data/social_networks/web-Stanford.txt',
    'google': '../data/social_networks/web-Google.txt',
    'ptop': '../data/social_networks/p2p-Gnutella05.txt',
    'stanford': '../data/social_networks/web-Stanford.txt',
    'notredame': '../data/social_networks/web-NotreDame.txt',
    'berkstan': '../data/social_networks/web-BerkStan.txt',
    'orkut': '../data/social_networks/com-orkut.ungraph.txt',
    'livejournal': '../data/social_networks/com-lj.ungraph.txt',
    'topcats': '../data/social_networks/wiki-topcats.txt',
    'friendster': '../data/social_networks/friendster/com-friendster.ungraph.txt'
}

def generate_random_network():
    sizes = [1000, 1000]
    probs = [[0.05, 0.01], [0.01, 0.05]]
    g = nx.stochastic_block_model(sizes, probs, seed=0)
    return nx.to_scipy_sparse_matrix(g, dtype=float, weight=None)

def generate_social_network(net_name = 'facebook'):
    edge_row = []
    edge_col = []
    with open(data_dirs[net_name], 'r') as f:
        for line in f.readlines():
            if line[0] == '#':
                continue
            src, dst = line.split()
            edge_row.append(int(src)) 
            edge_col.append(int(dst))

    num_node = max(edge_row+edge_col)+1
    network = sparse.csr_matrix(([1.0]*len(edge_row), (edge_row, edge_col)), shape=(num_node, num_node))
    return network

def generate_bipartite_social_graph(net_name = 'facebook'): 
    edge_row = [] # record nodes in original graph
    edge_col = [] # record edges in original graph with equal weights

    num_edge = 0
    with open(data_dirs[net_name], 'r') as f:
        for line in f.readlines():
            src, dst = line.split()
            edge_row.append(int(src)); edge_row.append(int(dst))
            edge_col.append(num_edge); edge_col.append(num_edge)
            num_edge += 1
    num_node = num_nodes[net_name]

    network = sparse.csr_matrix(([1]*len(edge_row), (edge_row, edge_col)), shape=(num_node, num_edge))

    cbg_population = np.ones(shape=(1, num_node))
    beta_poi = np.ones(shape=(1, num_edge))
    return network, beta_poi, cbg_population

def generate_advogato_network():
    path = '../data/social_networks/advogato/out.advogato'

    edge_row = []
    edge_col = []
    edge_val = []
    with open(path) as f:
        for line in f.readlines():
            line = line.split()
            if len(line) > 3:
                continue
            src, dst, weight = int(line[0]), int(line[1]), float(line[2])
            edge_row.append(src)
            edge_col.append(dst)
            edge_val.append(weight)

    num_node = num_nodes['advogato']
    network = sparse.csr_matrix((edge_val, (edge_row, edge_col)), shape=(num_node, num_node))
    # print(network.dtype)
    return network

def generate_darkweb_network():
    node_path = '../data/social_networks/darkweb/darkweb-nodes.ss'
    edge_path = '../data/social_networks/darkweb/darkweb-edges.ss'
    
    node_dict = dict()
    with open(node_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            node_id = line.split(";")[0]
            node_dict[node_id] = i
        
    edge_row = []
    edge_col = []
    edge_val = []
    with open(edge_path, 'r') as f:
        for line in f.readlines():
            src, dst, _, weight = line.split(";")
            edge_row.append(node_dict[src])
            edge_col.append(node_dict[dst])
            edge_val.append(float(weight))

            
    num_node = num_nodes['darkweb']
    network = sparse.csr_matrix((edge_val, (edge_row, edge_col)), shape=(num_node, num_node), dtype=float)
    network.data = np.clip(network.data, a_min = 0, a_max=10)
    return network

def generate_bitcoin_network():
    path = '../data/social_networks/bitcoin/soc-sign-bitcoinalpha.csv'
    df = pd.read_csv(path)

    src_list = df['7188'].to_list()
    dst_list = df['1'].to_list()

    node_dict = dict()
    counter = 0
    for src in src_list:
        if src not in node_dict:
            node_dict[src] = counter
            counter += 1
    
    for dst in dst_list:
        if dst not in node_dict:
            node_dict[dst] = counter
            counter += 1

    weight_list = df['10'].to_list()
    edge_row = [node_dict[node] for node in src_list]
    edge_col = [node_dict[node] for node in dst_list]

    num_node = num_nodes['bitcoin']
    network = sparse.csr_matrix((weight_list, (edge_row, edge_col)), shape=(num_node, num_node), dtype=float)
    # print(weight_list)
    network.data = np.exp(network.data/5)
    return network

def generate_airport_network():
    path = '../data/social_networks/airport/openflights.txt'
    weight_list = []
    src_list = []
    dst_list = []
    with open(path, "r") as f:
        for line in f.readlines():
            src, dst, weight = line.split()
            src_list.append(int(src))
            dst_list.append(int(dst))
            weight_list.append(float(weight))

    network = sparse.csr_matrix((weight_list, (src_list, dst_list)), dtype=float)
    return network