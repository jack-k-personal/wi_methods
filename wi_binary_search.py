import numpy as np 
import mdptoolbox
import time


def binary_search_all_arms(T, R, C, current_state, index_lb=-1, index_ub=1, gamma=0.95, tolerance=1e-1):

	N = T.shape[0]

	indexes = np.zeros(N)


	for i in range(N):

		indexes[i] = binary_search_one_arm(T[i],R[i],C[i],current_state[i], 
						index_lb=index_lb, index_ub=index_ub, gamma=gamma, tolerance=tolerance)


	return indexes


# the next two functions handle annoyances around 0 w.r.t. 
# the whole "increase/decrease by factor of 2 depending on the bound" idea
def reduce_lb(lb):
	if lb <= -1:
		return lb*2
	elif lb > -1 and lb < 1:
		return lb - 1
	elif lb >= 1:
		return lb/2

def increase_ub(ub):
	if ub <= -1:
		return ub/2
	elif ub > -1 and ub < 1:
		return ub + 1
	elif ub >= 1:
		return ub*2	



# this version has the step 0 which searches for upper and lower bounds first
def binary_search_one_arm(T, R, C, current_state, index_lb=-1, index_ub=1, gamma=0.95, tolerance=1e-1):


	# Go from S,A,S to A,S,S
	T_i = np.swapaxes(T,0,1)

	C_i = C

	# rewards need to be A,S,S too, but R is only S (current state)
	# create the "base-case" reward matrix
	R_base = np.zeros(T_i.shape)
	for x in range(R_base.shape[0]):
		for y in range(R_base.shape[1]):
			R_base[x,:,y] += R

	# this is the reward matrix we will edit
	R_i = np.copy(R_base)


	#####
	# run once to see if we should go up or down
	#####

	upper = index_ub
	lower = index_lb

	index_estimate = (upper + lower)/2

	# adjust the rewards
	# change the reward along the A axis, to account for new index estimate
	R_i[1] = R_base[1] - index_estimate*C_i[1]


	# run value iteration

	mdp = mdptoolbox.mdp.ValueIteration(T_i, R_i, discount=gamma, stop_criterion='fast')
	mdp.run()
	policy = np.array(mdp.policy)

	action = policy[current_state]
	# if not acting, reduce the penalty for acting
	if action == 0:
		upper = index_estimate

	# if acting, increase the penalty for acting
	elif action == 1:
		lower = index_estimate


	# now loop until we are told to turn around (finds appropriate upper and lower bounds)
	previous_action = action	
	while action == previous_action:
		# print('lower',lower, 'upper',upper)

		# this needs to be here for loop logic
		previous_action = action

		# if not acting, need to go down
		if previous_action == 0:
			index_estimate = lower
		# if acting, need to go up
		elif previous_action == 1:
			index_estimate = upper

		# adjust the rewards
		# change the reward along the A axis, to account for new index estimate
		R_i[1] = R_base[1] - index_estimate*C_i[1]

		# run value iteration

		mdp = mdptoolbox.mdp.ValueIteration(T_i, R_i, discount=gamma, stop_criterion='fast')
		mdp.run()
		policy = np.array(mdp.policy)

		action = policy[current_state]

		# if we haven't found our flip point, then shift the bounds as needed
		if action == previous_action:
			# if not acting, reduce the penalty for acting and reduce the lower bound
			if action == 0:
				upper = lower
				lower = reduce_lb(lower)

			# if acting, increase the penalty for acting and increase the upper bound
			elif action == 1:
				lower = index_estimate
				upper = increase_ub(upper)

		# else, we have found good bounds for our index
		else:
			# if not acting, reduce the penalty for acting and reduce the lower bound
			if action == 0:
				upper = index_estimate

			# if acting, increase the penalty for acting and increase the upper bound
			elif action == 1:
				lower = index_estimate

			# loop condition should let us break after this


	# NOW  we should have upper and lower bounds that are guaranteed to contain our index value
	while (upper - lower) > tolerance:
		# print('lower',lower, 'upper',upper)
		index_estimate = (upper + lower)/2
	


		# adjust the rewards
		# change the reward along the A axis, to account for new index estimate
		R_i[1] = R_base[1] - index_estimate*C_i[1]

		# run value iteration
		# import pdb; pdb.set_trace()
		# s = time.time()

		mdp = mdptoolbox.mdp.ValueIteration(T_i, R_i, discount=gamma, stop_criterion='fast')

		mdp.run()
		policy = np.array(mdp.policy)


		action = policy[current_state]
		# print("action",action)


		# if not acting, reduce the penalty for acting
		if action == 0:
			upper = index_estimate
			# print("upper!")

		# if acting, increase the penalty for acting
		elif action == 1:
			lower = index_estimate
			# print("lower!")


	index_estimate = (upper + lower)/2

	return index_estimate


