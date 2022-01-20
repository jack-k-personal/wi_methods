import numpy as np 
import pandas as pd 
import mathprog_methods
import pickle
import time
import params

par = params.Params()

# RMAB params
N = 2
S = 2
A = 2
gamma = par.gamma
C = np.array([[0,1] for _ in range(N)])
R = np.array([[0, 1] for _ in range(N)])
T = np.zeros((N,S,A,S))


# how precise to make probabilities when saving
round_precision = 4


# setting up the probability space grid
N_GRID = par.N_GRID
prob_epsilon = par.prob_epsilon
lower_prob = 0 + prob_epsilon
upper_prob = 1 - prob_epsilon


# use to save results
wi_grid_blp_compute_list = np.zeros((N_GRID**4,6))


no_differences = True

current_state = np.array([0,1])

i = 0
start = time.time()
for p01p in np.linspace(lower_prob, upper_prob, N_GRID):
    for p11p in np.linspace(lower_prob, upper_prob, N_GRID):
        for p01a in np.linspace(lower_prob, upper_prob, N_GRID):
            for p11a in np.linspace(lower_prob, upper_prob, N_GRID):

                key_tup = (p01p, p11p, p01a, p11a)
                # round these to make them more weildy for keying
                key_tup = tuple(list(map(lambda x: np.around(x,round_precision), key_tup)))

                wi_grid_blp_compute_list[i, :4] = key_tup

                T[:,0,0,0] = 1 - p01p
                T[:,0,0,1] = p01p
                T[:,1,0,0] = 1 - p11p
                T[:,1,0,1] = p11p

                T[:,0,1,0] = 1 - p01a
                T[:,0,1,1] = p01a
                T[:,1,1,0] = 1 - p11a
                T[:,1,1,1] = p11a

                L_vals, blp_indexes, z_vals, bina_vals = mathprog_methods.blp_to_compute_index(T, R, C, current_state,  gamma=gamma)
                # print('blp indexes\t',blp_indexes)
                # wi_grid_blp_compute[key_tup] = blp_indexes
                wi_grid_blp_compute_list[i, 4:] = blp_indexes
                i+=1

end = time.time()
blp_time = end - start

colnames = ['p01p','p11p','p01a','p11a','index0','index1']
blp_df = pd.DataFrame(wi_grid_blp_compute_list, columns=colnames)

with open('indexes/blp_indexes_df_NGRID%s.pickle'%N_GRID, 'wb') as handle:
    pickle.dump(blp_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("blp time",blp_time)
# print(blp_df)





