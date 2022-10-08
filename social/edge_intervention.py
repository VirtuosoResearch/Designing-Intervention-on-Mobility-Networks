from math import exp
from scipy.sparse import coo_matrix
from scipy.sparse.csr import csr_matrix
from scipy.sparse.linalg import svds, eigs
import numpy as np
from scipy import sparse
import cvxpy as cp

def scale_uniformly(network, budget):
    ratio = max(1.0 - budget / network.sum(), 0.0)
    # print('uniform', ratio)
    return network * ratio

def scale_by_weights(network, budget):
    # reduce weights proportional to edge weights
    reduced_weights = np.square(network.data) * budget / np.square(network.data).sum()
    network.data = np.maximum(network.data - reduced_weights, 0)
    return network 
    
def delete_by_edge_centrality(network, budget, components=1):
    if budget == 0:
        return network
    assert budget > 0
    original_weights = network.data.sum()

    U, D, Vt = svds(network, k=components)
    indices_weights = network.nonzero()

    if components == 1:
        num_nonzero = len(indices_weights[0])
        Cr = np.zeros(shape=(num_nonzero, ))
        for idx in range(components):
            for k in range(num_nonzero):
                i, j = indices_weights[0][k], indices_weights[1][k]
                Cr[k] += U[i, idx]*Vt[idx, j]
        centrality_matrix = coo_matrix((Cr, indices_weights), shape=network.shape)
    else:
        indicator_matrix = coo_matrix(([1] * len(indices_weights[0]), indices_weights), shape=network.shape)
        centrality_matrix = indicator_matrix.multiply(U @ np.diag(D) @ Vt)
    sort_indices = np.argsort(centrality_matrix.data)
    
    # find the top_k edges whose sum is exactly budget
    tmp_k = 100
    tmp_indices = sort_indices[-tmp_k:]
    while network.data[tmp_indices].sum() < budget:
        tmp_k += 200
        tmp_indices = sort_indices[-tmp_k:]
    
    for top_k in range(max(tmp_k-200, 1), tmp_k+1):
        tmp_indices = sort_indices[-top_k:]
        if network.data[tmp_indices].sum() > budget:
            break
    
    network.data[tmp_indices] = 0
    # print("Reduced weights: {}\t Budget: {}".format(original_weights - network.data.sum(), budget))
    return network

def greedy_selection_by_edge_centrality(network, budget, centrality_matrix):
    if budget == 0:
        return network
    assert budget > 0
    original_weights = network.data.sum()

    sort_indices = np.argsort(centrality_matrix.data)    
    # find the top_k edges whose sum is exactly budget
    tmp_k = 100
    tmp_indices = sort_indices[-tmp_k:]
    while network.data[tmp_indices].sum() < budget:
        tmp_k += 200
        tmp_indices = sort_indices[-tmp_k:]
    
    for top_k in range(max(tmp_k-200, 1), tmp_k+1):
        tmp_indices = sort_indices[-top_k:]
        if network.data[tmp_indices].sum() > budget:
            break
    
    network.data[tmp_indices[1:]] = 0

    budget_left = budget - (original_weights - network.data.sum())
    network.data[tmp_indices[0]] = max(0, network.data[tmp_indices[0]] - budget_left)

    return network    

def assign_weights_globally_sin(network, budget, components = 10, epochs = 20):
    update_network = network.copy()
    for step in range(epochs):
        U, D, Vt = svds(update_network, k=components)
        indices_weights = update_network.nonzero()
        ''' Unweighted edge centrality '''
        if components == 1:
            num_nonzero = len(indices_weights[0])
            Cr = np.zeros(shape=(num_nonzero, ))
            for idx in range(components):
                for k in range(num_nonzero):
                    i, j = indices_weights[0][k], indices_weights[1][k]
                    Cr[k] += U[i, idx]*Vt[idx, j]
            centrality_matrix = coo_matrix((Cr, indices_weights), shape=network.shape)
        else:
            indicator_matrix = coo_matrix(([1] * len(indices_weights[0]), indices_weights), shape=network.shape)
            centrality_matrix = indicator_matrix.multiply(U @ np.diag(D) @ Vt)

        M_i = greedy_selection_by_edge_centrality(network.copy(), budget, centrality_matrix)

        # search for step size
        def f(net):
            _, Ds, _ = svds(net, k=components)
            return Ds.sum()

        # search alpha
        alphas = np.concatenate([np.arange(1, 10)*1e-1, np.arange(1, 10)*1e-2, np.arange(1, 10)*1e-3])
        # alphas = np.concatenate([np.arange(1, 10)*1e-3, np.arange(1, 10)*1e-4, np.arange(1, 10)*1e-5])
        values = []
        for alpha in alphas:
            values.append((
                alpha, f((1-alpha)*update_network + alpha*M_i )
            ))

        # find alpha with smallest
        values.sort(key=lambda item: item[1])
        eta = values[0][0]

        update_network[indices_weights] = (1-eta)*update_network[indices_weights] + eta*M_i[indices_weights]

        print("Step {:d} step size is {}".format(step, eta))
        print("Step {:d} sum of top {:d} sigular values {}".format(step, components, D.sum()))
        print("Step {:d} weights {}".format(step, update_network.sum()))
    return update_network

def modify_weights(network, budget, strategy='none', strategy_params={}):
    '''
    Contains both static and temporal logic
    '''
    if strategy == 'none':
        return network

    if strategy == 'uniform':
        if strategy_params['type_of_network'] == 'static':
            return scale_uniformly(network, budget)
        
    if strategy == 'edge_weight':
        if strategy_params['type_of_network'] == 'static':
            return scale_by_weights(network, budget)

    if strategy == "edge_centrality_delete":
        if strategy_params['type_of_network'] == 'static':
            return delete_by_edge_centrality(network, budget, strategy_params['components'])     

    if strategy == 'global':
        if strategy_params['type_of_network'] == 'static':
            original_weights = network.sum()
            network =  assign_weights_globally_sin(network.copy(), 
                original_weights*strategy_params['lp_budget'], 
                components=strategy_params['lp_components'], 
                epochs=strategy_params['lp_epochs'])
            print(budget, original_weights - network.sum())
            if budget > (original_weights - network.sum()):
                # network = scale_by_edge_centrality(network, budget - (original_weights - network.sum()), strategy_params['components'], strategy_params['top_k'])
                network = delete_by_edge_centrality(network, budget - (original_weights - network.sum()), strategy_params['components'])
            return network