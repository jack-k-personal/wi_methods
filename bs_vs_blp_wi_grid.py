import numpy as np 
import pandas as pd 
import mathprog_methods
import mdptoolbox
import pickle
from wi_binary_search import binary_search_all_arms
import time
import params

par = params.Params()

# binary search params
index_lb = -1
index_ub = 1
bs_tolerance = 1e-4

# RMAB params
N = 2
S = 2
A = 2
gamma = par.gamma
C = np.array([[0,1] for _ in range(N)])
R = np.array([[0, 1] for _ in range(N)])
T = np.zeros((N,S,A,S))


# how precise to make probabilities when saving
round_precision = 4#int(np.ceil(np.log10(11)))

# for checking whether bs indexes are same as blp indexes
diff_tolerance = 1e-2





# setting up the probability space grid
N_GRID = par.N_GRID
prob_epsilon = par.prob_epsilon
lower_prob = 0 + prob_epsilon
upper_prob = 1 - prob_epsilon




# use to save results
wi_grid_binary_search = {}
wi_grid_blp_compute = {}
wi_grid_binary_search_list = np.zeros((N_GRID**4,6))
wi_grid_blp_compute_list = np.zeros((N_GRID**4,6))


no_differences = True

start = time.time()
i = 0
print("starting binary search")
for p01p in np.linspace(lower_prob, upper_prob, N_GRID):
    for p11p in np.linspace(lower_prob, upper_prob, N_GRID):
        for p01a in np.linspace(lower_prob, upper_prob, N_GRID):
            for p11a in np.linspace(lower_prob, upper_prob, N_GRID):

                key_tup = (p01p, p11p, p01a, p11a)
                # round these to make them more weildy for keying
                key_tup = tuple(list(map(lambda x: np.around(x,round_precision), key_tup)))

                wi_grid_binary_search_list[i, :4] = key_tup

                T[:,0,0,0] = 1 - p01p
                T[:,0,0,1] = p01p
                T[:,1,0,0] = 1 - p11p
                T[:,1,0,1] = p11p

                T[:,0,1,0] = 1 - p01a
                T[:,0,1,1] = p01a
                T[:,1,1,0] = 1 - p11a
                T[:,1,1,1] = p11a




                current_state = np.array([0,1])
                indexes = binary_search_all_arms(T, R, C, current_state, index_lb=index_lb, index_ub=index_ub, gamma=gamma, tolerance=bs_tolerance)

                wi_grid_binary_search[key_tup] = indexes
                wi_grid_binary_search_list[i, 4:] = indexes
                i+=1

                # L_vals, blp_indexes, z_vals, bina_vals = mathprog_methods.blp_to_compute_index(T, R, C, current_state,  gamma=gamma)
                # print('blp indexes\t',blp_indexes)

                # no_differences = no_differences & (abs(indexes - blp_indexes) < diff_tolerance).all()
                # if not no_differences:
                #     print(key_tup)
                #     raise ValueError()


end = time.time()
bs_time = end - start


i = 0
start = time.time()
print("starting BLP computation")
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
                wi_grid_blp_compute[key_tup] = blp_indexes
                wi_grid_blp_compute_list[i, 4:] = blp_indexes
                i+=1

end = time.time()
blp_time = end - start



for p01p in np.linspace(lower_prob, upper_prob, N_GRID):
    for p11p in np.linspace(lower_prob, upper_prob, N_GRID):
        for p01a in np.linspace(lower_prob, upper_prob, N_GRID):
            for p11a in np.linspace(lower_prob, upper_prob, N_GRID):

                key_tup = (p01p, p11p, p01a, p11a)
                # round these to make them more weildy for keying
                key_tup = tuple(list(map(lambda x: np.around(x,round_precision), key_tup)))

                bs_indexes = wi_grid_binary_search[key_tup]
                blp_indexes = wi_grid_blp_compute[key_tup]

                no_differences = no_differences & (abs(bs_indexes - blp_indexes) < diff_tolerance).all()
                if not no_differences:
                    print('indexes were not same:')
                    print(key_tup)
                    print(bs_indexes)
                    print(blp_indexes)
                    raise ValueError()



print("bs time:",bs_time)
print("blp time:",blp_time)
print("no differences:",no_differences)

# colnames = ['p01p','p11p','p01a','p11a','index0','index1']
# bs_df = pd.DataFrame(wi_grid_binary_search_list, columns=colnames)
# blp_df = pd.DataFrame(wi_grid_blp_compute_list, columns=colnames)

# with open('blp_indexes_df_NGRID%s.pickle'%N_GRID, 'wb') as handle:
#     pickle.dump(blp_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

# print(bs_df)
# print(blp_df)


