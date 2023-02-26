from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import svds
import numpy as np
from scipy import sparse

def scale_uniformly(network, budget):
    ratio = max(1.0 - budget / network.sum(), 0.0)
    # print('uniform', ratio)
    return network * ratio

def scale_by_weights(network, budget):
    # reduce weights proportional to edge weights
    reduced_weights = np.square(network.data) * budget / np.square(network.data).sum()
    network.data = np.maximum(network.data - reduced_weights, 0)
    return network 

def cap_max_occupancy(network, budget):
    # calculate each POI's maximum occupancy
    max_deg = np.sum(network, axis=0) # (1, m)
    # distribute budget to each POI
    budget_for_pois = budget * max_deg / max_deg.sum() # (1, m)
    # Apply uniform scaling in each POI (each column)
    for col in range(network.shape[1]):
        if network[:, col].sum() < 1e-3:
            continue
        network[:, col] = network[:, col] * max(1.0 - budget_for_pois[0, col] / network[:, col].sum() , 0.0)
    return network

def lockdown_poi_categories(network, budget, poi_indexes):
    # calculate each POI's maximum occupancy
    max_deg = np.sum(network, axis=0) # (1, m)
    # calculate budget only on target POI category
    max_deg = max_deg[:, poi_indexes]
    # distribute budget to each POI
    budget_for_pois = budget * max_deg / max_deg.sum() # (1, m)
    for i, col in enumerate(poi_indexes):
        if network[:, col].sum() < 1e-3:
            continue
        network[:, col] = network[:, col] * max(1.0 - budget_for_pois[0, i] / network[:, col].sum() , 0.0)
    return network

def delete_by_edge_centrality(network, budget, components=1):
    if budget == 0:
        return network
    assert budget > 0
    original_weights = network.data.sum()

    U, D, Vt = svds(network, k=components)
    indices_weights = network.nonzero()

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
    
    network.data[tmp_indices[1:]] = 0

    budget_left = budget - (original_weights - network.data.sum())
    network.data[tmp_indices[0]] = max(0, network.data[tmp_indices[0]] - budget_left)
    # print("Reduced weights: {}\t Budget: {}".format(original_weights - network.data.sum(), budget))
    return network    

def scale_by_edge_centrality(network, budget, components):
    ''' 
    One step edge centrality reduction
    '''
    if budget == 0:
        return network
    assert budget > 0
    # compute the edge centrality scores
    U, D, Vt = svds(network, k=components)
    indices_weights = network.nonzero()

    indicator_matrix = coo_matrix(([1] * len(indices_weights[0]), indices_weights), shape=network.shape)
    centrality_matrix = indicator_matrix.multiply(U @ np.diag(D) @ Vt)

    # apply greedy selection 
    network = greedy_selection_by_edge_centrality(network, budget, centrality_matrix)

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
    '''
    Frank-Wolfe EC algorithm that iterates greedy selection 
    '''
    update_network = network.copy()
    for step in range(epochs):
        U, D, Vt = svds(update_network, k=components)
        indices_weights = update_network.nonzero()
        indicator_matrix = coo_matrix(([1] * len(indices_weights[0]), indices_weights), shape=network.shape)
        centrality_matrix = indicator_matrix.multiply(U @ np.diag(D) @ Vt)

        M_i = greedy_selection_by_edge_centrality(network.copy(), budget, centrality_matrix)

        # search for step size
        def f(net):
            _, Ds, _ = svds(net, k=components)
            return Ds.sum()

        # search alpha
        alphas = np.concatenate([np.arange(1, 10)*1e-1, np.arange(1, 10)*1e-2, np.arange(1, 10)*1e-3])
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
        else:
            print('ERROR.')

    if strategy == 'edge_weight':
        if strategy_params['type_of_network'] == 'static':
            return scale_by_weights(network, budget)
    
    if strategy == 'capped':
        return cap_max_occupancy(network, budget)

    if strategy == 'category':
        return lockdown_poi_categories(network, budget, strategy_params['poi_indexes'])

    if strategy == "edge_centrality_delete":
        return delete_by_edge_centrality(network, budget, strategy_params['components'])        

    if strategy == 'global':
        original_weights = network.sum()
        network =  assign_weights_globally_sin(network.copy(), 
            original_weights*strategy_params['lp_budget'], 
            components=strategy_params['lp_components'], 
            epochs=strategy_params['lp_epochs'])
        print(budget, original_weights - network.sum())
        if budget > (original_weights - network.sum()):
            network = scale_by_edge_centrality(network, budget - (original_weights - network.sum()), strategy_params['components'])
        return network

def compute_lambda_one(network):
    U, D, Vt = svds(network, k=1)
    return D[0]

def multiply_weights(networks):
    '''
    Return the multiplication of network weights
    '''
    if not networks:
        return 1
    matmul = networks[0] 
    for network in networks[1:]:
        matmul = matmul @ network 
    return matmul

def multiply_weights_transpose(networks):
    '''
    Return the multiplication of network @ network.transpose
    '''
    if not networks:
        return 1
    networks = [network.todense() for network in networks]
    matmul = networks[0] @ networks[0].transpose()
    for network in networks[1:]:
        matmul = matmul @ (network @ network.transpose())
    return matmul

def compute_static_edge_centrality(network, components):
    U, D, Vt = svds(network, k=components)
    indices_weights = network.nonzero()

    indicator_matrix = coo_matrix(([1] * len(indices_weights[0]), indices_weights), shape=network.shape)
    centrality_matrix = indicator_matrix.multiply(U @ np.diag(D) @ Vt)
    return centrality_matrix, D

def compute_temporal_edge_centrality(networks, components):
    '''
    Input: 
        networks: a list a sparse matrices corresponding to network weight at time i
    Output:
        edge centrality matrices: a list of sparse matrices corresponding to edge centrality score at time i
    '''
    # conduct SVD on the multiplication of matrices
    networks_transpose = [network.todense() for network in networks]
    networks_transpose = [network @ network.transpose() for network in networks_transpose]
    network_matmul = multiply_weights(networks_transpose)
    centrality_matrix, D = compute_static_edge_centrality(network_matmul, components)
    centrality_matrix = centrality_matrix.todense()

    # multiplication with other matrices
    centrality_matrices = []
    for i, network in enumerate(networks):
        if i == 0:
            tmp_centrality_matrix = \
                centrality_matrix @ \
                multiply_weights([net for j, net in enumerate(networks_transpose) if j>i]) 
        elif i == len(networks)-1:
            tmp_centrality_matrix = \
                multiply_weights([net for j, net in enumerate(networks_transpose) if j<i]) @ \
                centrality_matrix
        else:
            tmp_centrality_matrix = \
                multiply_weights([net for j, net in enumerate(networks_transpose) if j<i]) @ \
                centrality_matrix @ \
                multiply_weights([net for j, net in enumerate(networks_transpose) if j>i]) 
        tmp_centrality_matrix = (tmp_centrality_matrix + tmp_centrality_matrix.transpose()) @ network
        indices_weights = network.nonzero()
        indicator_matrix = coo_matrix(([1] * len(indices_weights[0]), indices_weights), shape=network.shape)
        tmp_centrality_matrix = indicator_matrix.multiply(tmp_centrality_matrix)
        centrality_matrices.append(tmp_centrality_matrix)
    
    return centrality_matrices, D

def greedy_selection_by_temporal_edge_centrality(networks, budget, centrality_matrices):
    ''' Greedily select edges that have top-k edge centrality score '''
    if budget == 0:
        return networks
    assert budget > 0
    original_weights = sum([network.data.sum() for network in networks])

    sort_indices = []
    for i, centrality_matrix in enumerate(centrality_matrices):
        sort_indices += [(i, j, ec) for j, ec in enumerate(centrality_matrix.data)]
    sort_indices.sort(key=lambda x:x[2])
    
    def compute_weight_sum(indices):
        sum = 0
        for (network_idx, weight_idx, _) in indices:
            sum += networks[network_idx].data[weight_idx]
        return sum

    # find the top_k edges whose sum is exactly budget
    tmp_k = 100
    tmp_indices = sort_indices[-tmp_k:]
    while compute_weight_sum(tmp_indices) < budget:
        tmp_k += 200
        tmp_indices = sort_indices[-tmp_k:]
    
    for top_k in range(max(tmp_k-200, 1), tmp_k+1):
        tmp_indices = sort_indices[-top_k:]
        if compute_weight_sum(tmp_indices) > budget:
            break

    for (network_idx, weight_idx, _) in tmp_indices[1:]:
        networks[network_idx].data[weight_idx] = 0
    
    budget_left = budget - (original_weights - sum([network.data.sum() for network in networks]))
    networks[tmp_indices[0][0]].data[tmp_indices[0][1]] = max(
        0, networks[tmp_indices[0][0]].data[tmp_indices[0][1]] - budget_left
        )

    print("Complete one greedy selection!")
    return networks

def assign_temporal_weights_globally(networks, budget, components, epochs):
    '''
    Frank-Wolfe EC algorithm that iterates greedy selection 
    '''
    update_networks = [network.copy() for network in networks]
    for step in range(epochs):
        centrality_matrices, D = compute_temporal_edge_centrality(update_networks, components)

        M_is = greedy_selection_by_temporal_edge_centrality([network.copy() for network in networks], budget, centrality_matrices)

        # search for step size
        def f(nets):
            _, Ds, _ = svds(multiply_weights_transpose(nets), k=components)
            return Ds.sum()

        # search alpha
        alphas = np.concatenate([np.arange(1, 10)*1e-1, np.arange(1, 10)*1e-2])
        values = []
        for alpha in alphas:
            values.append((
                alpha, 
                f(
                    [(1-alpha)*update_networks[i] + alpha*M_is[i] for i in range(len(update_networks))]
                )
            ))

        # find alpha with smallest
        values.sort(key=lambda item: item[1])
        eta = values[0][0]

        update_networks = [
            (1-eta)*update_networks[i] + eta*M_is[i] for i in range(len(update_networks))
            ]

        print("Step {:d} step size is {}".format(step, eta))
        print("Step {:d} sum of top {:d} sigular values {}".format(step, components, D.sum()))
        print("Step {:d} weights {}".format(step, sum([network.sum() for network in update_networks])))
    return update_networks

def scale_temporal_network_by_edge_centrality(networks, budget, components):
    # compute the edge centrality scores 
    centrality_matrices, _ = compute_temporal_edge_centrality(networks, components)

    # greedy selection by edge centrality (equal to one iteration of Frank-Wolfe EC)
    networks = greedy_selection_by_temporal_edge_centrality(networks, budget, centrality_matrices)

    return networks

def scale_temporal_network_uniformly(networks, budget):
    original_weight_sum = sum([network.data.sum() for network in networks])
    ratio = max(1.0 - budget/original_weight_sum, 0.0)
    networks = [network*ratio for network in networks]
    return networks

def scale_temporal_network_by_weights(networks, budget):
    original_weight_sum_square = sum([np.square(network.data).sum() for network in networks])
    reduced_weights = [np.square(network.data) * budget / original_weight_sum_square for network in networks]
    for i, network in enumerate(networks):
        network.data = np.maximum(network.data - reduced_weights[i], 0)
    return networks

def distribute_budget_among_weeks(networks, budget, start_week, end_week, temporal_strategy = 'uniform'):
    '''
    Input: the overall budget
    Output: a list of budget for each week [start_week, ..., end_week]
    '''
    if temporal_strategy == "uniform":
        budget = np.array([budget]*(end_week - start_week + 1))
        budget = budget/(end_week - start_week + 1)
        return budget

    if temporal_strategy == "exponential":
        ratios = np.arange(end_week-start_week+1)+1
        ratios = np.exp(-ratios)
        ratios = ratios/ratios.sum()
        budget = budget * ratios
        return budget

    if temporal_strategy == "weights":
        ratios = [network.sum() for network in networks]
        ratios = np.array(ratios)
        ratios = ratios/ratios.sum()
        budget = budget * ratios
        return budget        

    if temporal_strategy == "one_shot":
        tmp_budget = np.zeros(shape=((end_week - start_week + 1), ))
        tmp_budget[0] = budget
        return tmp_budget

    if temporal_strategy == "lambda":
        network_list = [networks[i] for i in range(start_week-1, end_week)]
        lambda_list = np.array([compute_lambda_one(net) for net in network_list])
        # normalize by lambda_list
        lambda_list = lambda_list / lambda_list.sum()
        budget = budget*lambda_list
        return budget 

def scale_temporal_network(networks, budget, start_week, end_week, 
    temporal_strategy = 'uniform', scale_strategy = "uniform", strategy_params = {}):
    # get temporal bughet list
    budget_list = distribute_budget_among_weeks(networks, budget, start_week, end_week, temporal_strategy = temporal_strategy)
    print("Budget list for weeks {}".format(budget_list))
    assert len(budget_list) == end_week - start_week + 1

    for i in range(start_week-1, end_week):
        if scale_strategy == "uniform":
            networks[i] = scale_uniformly(
                networks[i], 
                budget_list[i - start_week+1]
            )
        
        elif scale_strategy == "edge_centrality":
            networks[i] = scale_by_edge_centrality(
                networks[i], 
                budget_list[i - start_week+1], 
                strategy_params['components'], 
                strategy_params['top_k']
            )

    return networks

def modify_temporal_weights(networks, budget, temporal_strategy, scale_strategy, strategy_params={}):
    '''
    Contains only temporal logic
    '''
    if temporal_strategy == 'none':
        return networks
    elif temporal_strategy == 'global':
        original_weights = sum([network.data.sum() for network in networks])
        networks = assign_temporal_weights_globally(networks,
            budget=original_weights*strategy_params['lp_budget'], 
            components=strategy_params['lp_components'], 
            epochs=strategy_params['lp_epochs'])
        print(budget, original_weights - sum([network.data.sum() for network in networks]))
        if budget > (original_weights - sum([network.data.sum() for network in networks])):
            networks = scale_temporal_network_by_edge_centrality(networks, 
                budget - (original_weights - sum([network.data.sum() for network in networks])), 
                strategy_params['components'])
        return networks
    elif temporal_strategy == 'global_uniform':
        return scale_temporal_network_uniformly(networks, budget)
    elif temporal_strategy == 'global_weighted':
        return scale_temporal_network_by_weights(networks, budget)
    else:
        return scale_temporal_network(networks, budget, 
            start_week=strategy_params["start_week"], 
            end_week=strategy_params["end_week"],
            temporal_strategy = temporal_strategy, 
            scale_strategy = scale_strategy,
            strategy_params = strategy_params)