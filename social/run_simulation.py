from seir_model import SEIR
from build_social_networks import generate_airport_network, generate_bitcoin_network, generate_darkweb_network, generate_random_network, generate_social_network, generate_advogato_network
from edge_intervention import modify_weights
import numpy as np
from scipy import sparse

import argparse


def main(args):
    if args.net_name == 'advogato':
        network = generate_advogato_network()
    elif args.net_name == 'darkweb':
        network = generate_darkweb_network()
    elif args.net_name == 'bitcoin':
        network = generate_bitcoin_network()
    elif args.net_name == 'random':
        network = generate_random_network()
    elif args.net_name == 'airport':
        network = generate_airport_network()
    else:
        network = generate_social_network(args.net_name)
    budget = args.budget*network.sum()
    print("Network size: {}".format(network.shape))
    
    strategy_param = {
                        'type_of_network': 'static',
                        # scale_by_edge_centrality params
                        'components': args.components,
                        'top_k': args.top_k,
                        # global_assignment params
                        'lp_budget': args.lp_budget,
                        'lp_components': args.lp_components, 
                        'lp_epochs': args.lp_epochs,
                    }
    modified_weights = modify_weights(network.copy(), budget, args.strategy, strategy_param)
    try:
        U, D, Vt = sparse.linalg.svds(modified_weights, k=10)
        print('top eigs for modified networks', D)
    except:
        pass
    print("Modfied weights: {}".format(network.sum() - modified_weights.sum()))
    
    R_list = []
    for i in range(args.runs):
            
        model = SEIR(network=modified_weights, simulation_time=args.epochs, 
                    beta=args.beta_base, incubation_period=96/6, 
                    recover_period = 84/6, initial_prevalence=args.p_zero)
        model.run_simulation()

        S_num, E_num, I_num, R_num = model.get_statistics()
        R_list.append(R_num)
    R_list = np.array(R_list)
    R_mean = np.mean(R_list, axis=0)
    R_std = np.std(R_list, axis=0)
    print("Total Number of infected people: {:.3f} +/- {:.3f}".format(R_mean[-1], R_std[-1]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=50)
    parser.add_argument('--net_name', type=str, default='facebook')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--strategy', type=str, default='none')
    parser.add_argument('--budget', type=float, default=0.0)
    # for edge centrality
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--components', type=int, default=1)
    # for global assignment
    parser.add_argument('--lp_budget', type=float, default=0.15)
    parser.add_argument('--lp_components', type=int, default=5)
    parser.add_argument('--lp_epochs', type=int, default=10)

    ''' SEIR Parameters '''
    parser.add_argument('--beta_base', type=float, default=0.05)
    parser.add_argument('--p_zero', type=float, default=1e-2)
    
    args = parser.parse_args()

    main(args)