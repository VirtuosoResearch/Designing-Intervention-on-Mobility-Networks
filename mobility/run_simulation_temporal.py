import os
import argparse
import numpy as np
from scipy import sparse
from pickle import load
from numpy.core.fromnumeric import shape
from edge_intervention import modify_weights, modify_temporal_weights
from seir_model import SEIR
import matplotlib.pyplot as plt

from helper_constants import *
from helper_functions import load_networks, load_cbg_population, generate_networks_from_safegraph, load_static_networks, load_temporal_networks

def main(args=None):
    msa_name = args.MSA

    # weights shape should be (num_cbg, num_poi)
    weights, beta_poi, cbg_population = load_temporal_networks(msa_name)
    beta_poi = beta_poi*args.poi_psi
    original_weight_sum = sum([weight.data.sum() for weight in weights])
    # Calculate the budget in number of people (weights)
    budget = args.budget*original_weight_sum

    num_cbg = weights[0].shape[0]
    num_poi = weights[0].shape[1]
    total_population = cbg_population.sum()

    print(f"======== Successfully loaded mobility network for {msa_name} ========")
    print(f"======== Number of CBGs: {num_cbg} ========")
    print(f"======== Number of POIs: {num_poi} ========")
    print(f"======== Reduced amount of visitors: {budget} ========")
    
    strategy_param = {
                        'type_of_network': 'temporal',
                        # scale_by_edge_centrality params
                        'components': args.components,
                        'top_k': args.top_k,
                        # global_assignment params
                        'lp_budget': args.lp_budget,
                        'lp_components': args.lp_components, 
                        'lp_epochs': args.lp_epochs,

                        'start_week': 1,
                        'end_week': 10,
                    }
    modified_weights = modify_temporal_weights(
        weights, budget, args.temporal_strategy, args.scale_strategy, strategy_param
        )
    modified_weight_sum = sum([weight.data.sum() for weight in modified_weights])
    print(budget, original_weight_sum - modified_weight_sum)
    # expands to daily networks
    modified_weights = [modified_weights[0]]*7 + \
        [modified_weights[1]]*7 + [modified_weights[2]]*7 + \
        [modified_weights[3]]*7 + [modified_weights[4]]*7 + \
        [modified_weights[5]]*7 + [modified_weights[6]]*7 + \
        [modified_weights[7]]*7 + [modified_weights[8]]*7 + [modified_weights[9]]*7

    S_list = []
    E_list = []
    I_list = []
    R_list = []
    for run in range(args.runs):
        model = SEIR(
            num_cbg=num_cbg, # real data
            num_poi=num_poi, # real data
            graph_weights=modified_weights, # real data
            beta_base=args.beta_base, # real data (vary among cities)
            beta_poi=beta_poi, # load real data (vary among cities)
            incubation_period=96/6, # real data 
            recover_period=84/6, # real data 
            initial_prevalence=args.p_zero, # real data (vary among cities)
            cbg_population=cbg_population, # load real data (vary among cities)
            simulation_time=args.epochs,
            type_of_network=strategy_param['type_of_network']
        )
        model.run_simulation()

        # plotting logic
        x_axis = np.array(range(args.epochs+1))
        S, E, I, R = model.get_statistics()

        S_list.append(S)
        E_list.append(E)
        I_list.append(I)
        R_list.append(R)

    S = np.mean(np.array(S_list), axis=0); S_std = np.std(np.array(S_list), axis=0)
    E = np.mean(np.array(E_list), axis=0); E_std = np.std(np.array(E_list), axis=0)
    I = np.mean(np.array(I_list), axis=0); I_std = np.std(np.array(I_list), axis=0)
    R = np.mean(np.array(R_list), axis=0); R_std = np.std(np.array(R_list), axis=0)
    print("Total Number of infected people: {:.3f} +/- {:.3f}".format(R[-1]/1e3, R_std[-1]/1e3))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=50)
    parser.add_argument('--MSA', type=str, default='NY')
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument(
        '--temporal_strategy', type=str, default='none')
    parser.add_argument(
        '--scale_strategy', type=str, default='edge_centrality', 
        choices=['none', 'uniform', 'edge_centrality'])
    parser.add_argument('--budget', type=float, default=0.0)

    # for edge centrality
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--components', type=int, default=1)

    # for global assignment
    parser.add_argument('--lp_budget', type=float, default=0.1)
    parser.add_argument('--lp_components', type=int, default=5)
    parser.add_argument('--lp_epochs', type=int, default=10)

    ''' SEIR Parameters '''
    parser.add_argument('--beta_base', type=float, default=0.001)
    parser.add_argument('--poi_psi', type=float, default=2700)
    parser.add_argument('--p_zero', type=float, default=1e-4)
    
    args = parser.parse_args()

    main(args)
