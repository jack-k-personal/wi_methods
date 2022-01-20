import numpy as np 
import pandas as pd
import mathprog_methods
import pickle
import time
import params

par = params.Params()

np.random.seed(2)
pd.set_option("display.max_rows", None, "display.max_columns", None)



N = 2
S = 2
A = 2
gamma = par.gamma
C = np.array([0, 1])
R = np.array([0, 1])
T = np.zeros((N,S,A,S))

N_GRID = par.N_GRID
prob_epsilon = par.prob_epsilon
lower_prob = 0 + prob_epsilon
upper_prob = 1 - prob_epsilon

wi_grid_df = None
with open('indexes/blp_indexes_df_NGRID%s.pickle'%N_GRID, 'rb') as handle:
    wi_grid_df = pickle.load(handle)


# for ensuring indexes are same between the grid and the bqp
diff_tolerance = 1e-2

# for avoiding annoying issues with filtering the pandas df
choice_eps = 1e-3
choices = np.linspace(lower_prob, upper_prob, N_GRID)

no_differences = True

N_TRIALS = 10
start = time.time()
for trial in range(N_TRIALS):


    p01p_range = np.sort(np.random.choice(choices,2,replace=True))
    p11p_range = np.sort(np.random.choice(choices,2,replace=True))
    p01a_range = np.sort(np.random.choice(choices,2,replace=True))
    p11a_range = np.sort(np.random.choice(choices,2,replace=True))



    current_state = np.random.choice([0,1])

    # get ranges from the pre-computed df
    sub_df = wi_grid_df[(wi_grid_df['p01p'] >= p01p_range[0]-choice_eps) & (wi_grid_df['p01p'] <= p01p_range[1]+choice_eps)]
    sub_df = sub_df[(sub_df['p11p'] >= p11p_range[0]-choice_eps) & (sub_df['p11p'] <= p11p_range[1]+choice_eps)]
    sub_df = sub_df[(sub_df['p01a'] >= p01a_range[0]-choice_eps) & (sub_df['p01a'] <= p01a_range[1]+choice_eps)]
    sub_df = sub_df[(sub_df['p11a'] >= p11a_range[0]-choice_eps) & (sub_df['p11a'] <= p11a_range[1]+choice_eps)]

    max_index_grid = sub_df['index%s'%current_state].max()
    min_index_grid = sub_df['index%s'%current_state].min()
    # print(wi_grid_df)
    print(sub_df)
    
    print("state",current_state)
    print('ranges')
    print('p01p_range',p01p_range)
    print('p11p_range',p11p_range)
    print('p01a_range',p01a_range)
    print('p11a_range',p11a_range)


    print('trying to max the wi')
    maxed_index_bqp, L_max, z_max, bina_max, T_return  = mathprog_methods.bqp_to_optimize_index(p01p_range, p11p_range, p01a_range, p11a_range,
                                                                R, C, current_state, maximize=True, gamma=gamma)

    print("transition settings")
    print('p01p',T_return[0,0,1])
    print('p11p',T_return[1,0,1])
    print('p01a',T_return[0,1,1])
    print('p11a',T_return[1,1,1])
    print()


    # check for just one arm
    Q = np.zeros((T.shape[0], T.shape[1]))
    for s in range(T.shape[0]):
        for a in range(T.shape[1]):
            Q[s,a] = R[s] - maxed_index_bqp*C[a] + gamma*L_max.dot(T_return[s,a])

    # print('BQP L vals')
    # print(L_max)
    # print('BQP Q vals')
    # print(Q)
    # print()
    print('max index (grid)', max_index_grid)
    print('max index (bqp)', maxed_index_bqp)
    print()
    print()
    no_differences = no_differences & (abs(max_index_grid - maxed_index_bqp) < diff_tolerance)

    print('trying to min the wi')
    minned_index_bqp, L_min, z_min, bina_min, T_return  = mathprog_methods.bqp_to_optimize_index(p01p_range, p11p_range, p01a_range, p11a_range,
                                                                R, C, current_state, maximize=False, gamma=gamma)


    # check for just one arm
    Q = np.zeros((T.shape[0], T.shape[1]))
    for s in range(T.shape[0]):
        for a in range(T.shape[1]):
            Q[s,a] = R[s] - minned_index_bqp*C[a] + gamma*L_min.dot(T_return[s,a])

    # print('BQP L vals')
    # print(L_min)
    # print('BQP Q vals')
    # print(Q)
    # print('z_min')
    # print(z_min)
    # print('bina_min')
    # print(bina_min)
    print("transition settings")
    print('p01p',T_return[0,0,1])
    print('p11p',T_return[1,0,1])
    print('p01a',T_return[0,1,1])
    print('p11a',T_return[1,1,1])
    print()


    print('min index (grid)', min_index_grid)
    print('min index (bqp)', minned_index_bqp)
    print()
    no_differences = no_differences & (abs(minned_index_bqp - min_index_grid) < diff_tolerance)



end = time.time()
bqp_time = end - start

print('time taken',bqp_time)
print("no differences:",no_differences)

# colnames = ['p01p','p11p','p01a','p11a','index0','index1']
# blp_df = pd.DataFrame(wi_grid_blp_compute_list, columns=colnames)

# with open('blp_indexes_df_NGRID%s.pickle'%N_GRID, 'wb') as handle:
#     pickle.dump(blp_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

# print(blp_df)





