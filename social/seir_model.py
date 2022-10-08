import numpy as np

# states
S = 1
E = 2
I = 3
R = 4

class SEIR:

    def __init__(self, network, simulation_time, 
        beta = 0.01, incubation_period=96, recover_period=86, initial_prevalence = 1e-4):
        self.network = network
        self.num_nodes = network.shape[0]

        self.T = simulation_time
        self.history_states = np.ones(shape=(self.T+1, self.num_nodes))
        self.states = np.ones(shape=(self.num_nodes, ))

        self.beta = beta
        self.incubation_period = incubation_period
        self.recover_period = recover_period
        ''' Population Initialization '''
        initial_E = np.random.binomial(1, initial_prevalence, size=(self.num_nodes, ))
        self.states = self.states + initial_E
        self.history_states[0, :] = self.states

    def run_one_iteration(self, t):
        S_index = np.where(self.states == S)[0]
        E_index = np.where(self.states == E)[0]
        I_index = np.where(self.states == I)[0]
        # R_index = np.where(self.states == R)[0]

        ''' 
        For current S nodes
            1. get neighbor weights
            2. calculate each node infected probability
            3. sample acccording to the prob
        '''
        if len(S_index) != 0 and len(I_index)!=0:
            neighbor_weights = self.network[S_index, :][:, I_index]
            infected_probs = neighbor_weights.todense()
            infected_probs = 1 - np.prod(
                np.minimum(np.maximum(1 - infected_probs*self.beta, 1e-3), 1), 
                axis=1)
            infected = np.random.binomial(np.ones_like(infected_probs, dtype=int), infected_probs)
            self.states[S_index] = self.states[S_index] + infected[:, 0]

        '''
        For current E nodes
        '''
        if len(E_index) != 0:
            incubated = np.random.binomial(1, 1/self.incubation_period, size=E_index.shape)
            self.states[E_index] = self.states[E_index] + incubated

        '''
        For current I nodes
        '''
        if len(I_index) != 0:
            recovered = np.random.binomial(1, 1/self.recover_period, size=I_index.shape)
            self.states[I_index] = self.states[I_index] + recovered

        self.history_states[t, :] = self.states

    def run_simulation(self):
        for t in range(1, self.T+1):
            ''' Could add more logic into this function'''
            self.run_one_iteration(t)

    def get_statistics(self):
        '''
        Get the S, E, I, R along T time steps: (T+1, 1)
        '''
        S_nums = np.sum(self.history_states == S, axis=1)
        E_nums = np.sum(self.history_states == E, axis=1)
        I_nums = np.sum(self.history_states == I, axis=1)
        R_nums = np.sum(self.history_states == R, axis=1)
        return S_nums, E_nums, I_nums, R_nums

            