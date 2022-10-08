import numpy as np

class SEIR:
    def __init__(self, num_cbg, num_poi, graph_weights, 
                beta_base, 
                beta_poi, 
                incubation_period, 
                recover_period, 
                initial_prevalence,
                simulation_time,
                cbg_population,
                type_of_network):
        ''' Network Parameters '''
        self.num_cbg = num_cbg # n
        self.num_poi = num_poi # m
        self.mobility_weights = graph_weights # (n, m) or (T, n, m)

        ''' Diffusion Parameters: Obtained from Mobility Network '''
        self.beta_base = beta_base # transmission rate inside each CBG 
        self.beta_poi = beta_poi # transmission rate inside each POI (1, m)
        self.delta_E = incubation_period # latency/incubation period
        self.delta_I = recover_period # recover/infectious perpiod
        self.p_0 = initial_prevalence

        ''' Simulation Parameters'''
        self.T = simulation_time # number of iterations

        ''' Population Numbers '''
        self.num_population = cbg_population.astype(int) # (1, n)
        self.num_susceptible = np.zeros(shape=(self.T+1, self.num_cbg), dtype=int) # (T+1, n) or (1, n)
        self.num_exposed = np.zeros(shape=(self.T+1, self.num_cbg), dtype=int) # (T+1, n) or (1, n)
        self.num_infected = np.zeros(shape=(self.T+1, self.num_cbg), dtype=int) # (T+1, n) or (1, n)
        self.num_recovered = np.zeros(shape=(self.T+1, self.num_cbg), dtype=int) # (T+1, n) or (1, n) 
        ''' Population Initialization '''
        init_exposed = self._sample_from_binomial(self.num_population, self.p_0)
        self.num_susceptible[0, :] = self.num_population - init_exposed
        self.num_exposed[0, :] = init_exposed

        ''' Network Type '''
        self.type_of_network = type_of_network

    def init_population(self, S = None, E = None, I = None, R = None):
        ''' Customize Initilization population '''
        if S is not None:
            self.num_susceptible[0, :] = S
        if E is not None:
            self.num_exposed[0, :] = E
        if I is not None:
            self.num_infected[0, :] = I
        if R is not None:
            self.num_recovered[0, :] = R

    def get_statistics(self):
        S = np.sum(self.num_susceptible, axis=1)
        E = np.sum(self.num_exposed, axis=1)
        I = np.sum(self.num_infected, axis=1)
        R = np.sum(self.num_recovered, axis=1)
        return S, E, I, R

    def save_statistics(self):
        pass

    def run_one_iteration(self, t, mobility_weights):
        ''' Update one SEIR iteration'''
        N = self.num_population # (1, n)
        S = self.num_susceptible[t:t+1, :] # (1, n)
        E = self.num_exposed[t:t+1, :] # (1, n)
        I = self.num_infected[t:t+1, :] # (1, n)
        R = self.num_recovered[t:t+1, :] # (1, n)
        
        # V_poi = np.sum(mobility_weights, axis=0, keepdims=True) # (1, m)
        I_poi = (I / N) @ mobility_weights # (1, m)
        lam_poi = self.beta_poi * I_poi  # (1, m)
        lam_cbg = self.beta_base * I / N # (1, n)
        lambda_poi_sum = lam_poi @ mobility_weights.transpose() # (1, n)

        num_S_to_E = self._sample_from_poisson(lambda_poi_sum*S/N) \
                        + self._sample_from_binomial(S, lam_cbg)
        num_E_to_I = self._sample_from_binomial(E, 1/self.delta_E)
        num_I_to_R = self._sample_from_binomial(I, 1/self.delta_I)

        # In case they become < 0
        num_S_to_E = np.minimum(num_S_to_E, S)
        num_E_to_I = np.minimum(num_E_to_I, E + num_S_to_E)
        num_I_to_R = np.minimum(num_I_to_R, I + num_E_to_I)

        self.num_susceptible[t+1, :] = S - num_S_to_E
        self.num_exposed[t+1, :] = E + num_S_to_E - num_E_to_I
        self.num_infected[t+1, :] = I + num_E_to_I - num_I_to_R
        self.num_recovered[t+1, :] = R + num_I_to_R

    def run_simulation(self):
        for t in range(self.T):
            ''' Could add more logic into this function'''
            if self.type_of_network == 'static':
                self.run_one_iteration(t, self.mobility_weights)
            elif self.type_of_network == 'temporal':
                self.run_one_iteration(t, self.mobility_weights[t])
            else:
                print('Unknown type of network.')

    def _sample_from_poisson(self, lam):
        lam = np.maximum(lam, [0] * len(lam))
        return np.random.poisson(lam).astype(int)

    def _sample_from_binomial(self, num, prob):
        return np.random.binomial(num, prob).astype(int)