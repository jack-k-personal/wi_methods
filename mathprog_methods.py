from gurobipy import *
import numpy as np 
import time

# F1

def blp_to_compute_index(T, R, C, start_state, lambda_lim=None, lambda_val=None, gamma=0.95):

	start = time.time()

	NPROCS = T.shape[0]
	NSTATES = T.shape[1]
	NACTIONS = T.shape[2]

	# Create a new model
	m = Model("BLP for Computing the Whittle indices")
	m.setParam( 'OutputFlag', False )

	L = np.zeros((NPROCS,NSTATES),dtype=object)
	
	mu = np.zeros((NPROCS,NSTATES),dtype=object)
	for i in range(NPROCS):
		mu[i, int(start_state[i])] = 1
		# mu[i] = np.ones(NSTATES)/NSTATES

	# Create variables
	lb = -GRB.INFINITY
	ub = GRB.INFINITY
	if lambda_lim is not None:
		ub = lambda_lim


	# going to compute indices in a decoupled manner
	index_variables = np.zeros(NPROCS,dtype=object)
	for i in range(NPROCS):
		index_variables[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=lb, ub=ub, name='index_%s'%i)


	for p in range(NPROCS):
		for i in range(NSTATES):
			L[p,i] = m.addVar(vtype=GRB.CONTINUOUS, lb=lb, ub=ub, name='L_%s_%s'%(p,i))
 

	# Problem with the LP to compute whittle index:
	# The whittle index constraint causes the V = max_a{Q[a]} construction of the LP to sometimes fail. Why?
	# Normally, the lp to compute value function relies on computing V = max_a{Q[a]}
	# by using typical LP construction to compute a max, i.e., min{V} where V >= Q[a] for all a.
	# however, when we add the additional whittle index constraint, that constraint may be more restrictive
	# than any of the Q[a], so our V is not guaranteed to be equal to any of the Q[a] when minimized!
	# Instead, it may settle on the more restrictive Whittle index constraint.
	#
	# Solution: 
	#      we use binary variables + bigM construction to enforce that V = max{Q[a]}
	# 
	# Note:
	#      Solving a binary linear program is obviously slower than solving an LP, so this won't scale 
	#      as well as other potential approaches. However, this should be a good base for a QBP to max/min
	#      the Whittle index!


	z = np.zeros((NPROCS,NSTATES,NACTIONS),dtype=object)
	bina = np.zeros((NPROCS,NSTATES,NACTIONS),dtype=object)
	for p in range(NPROCS):
		for i in range(NSTATES):
			for j in range(NACTIONS):
				z[p,i,j] = m.addVar(vtype=GRB.CONTINUOUS, lb=lb, ub=ub, name='z_%s_%s_%s'%(p,i,j))
				bina[p,i,j] = m.addVar(vtype=GRB.BINARY, name='binary_%s_%s_%s'%(p,i,j))



	L = np.array(L)


	# print('Variables added in %ss:'%(time.time() - start))
	start = time.time()


	m.modelSense=GRB.MINIMIZE

	# Set objective
	# m.setObjectiveN(obj, index, priority) -- larger priority = optimize first

	M = 1e2
	# in the BLP construction, we don't need to minimize over Q constraints -- binary vars will ensure tightness
	# m.setObjectiveN(1, 0, 1)
	# m.setObjectiveN(sum([L[i].dot(mu[i]) for i in range(NPROCS)]) + index_variables[i]*B*((1-gamma)**-1), 0, 1)
	m.setObjectiveN(sum([L[i].dot(mu[i]) for i in range(NPROCS)]), 0, 1)

	if lambda_val is not None:
		# m.addConstr(index_variables[0]==lambda_val)
		1==1

	bigM = M
	# import pdb; pdb.set_trace()
	# set constraints to figure out what value SHOULD be in value-iteration land
	for p in range(NPROCS):
		for i in range(NSTATES):
			for j in range(NACTIONS):
				# m.addConstr( L[p][i] >= R[p][i] - index_variable*c[j] + gamma*L[p].dot(T[p,i,j]) )
				m.addConstr( L[p,i] >= R[p,i] - index_variables[p]*C[p,j] + gamma*LinExpr(T[p,i,j], L[p])) 
				m.addConstr( z[p,i,j] == R[p,i] - index_variables[p]*C[p,j] + gamma*LinExpr(T[p,i,j], L[p]))
				m.addConstr( L[p,i] <= z[p,i,j] + bina[p,i,j]*bigM)

			# ensure that only one of the upper bound constraints on the value function is tight
			# by construction, the only way to satisfy this in conjunction with the above constraints
			# is if the largest lower bound constraint is also the only active upper bound constraint
			m.addConstr(bina[p,i].sum() == NACTIONS - 1) # any binary var set to 1 will be a loose upper bound via bigM


	# enforce the whittle index constraint on all p arms (note that they are all independent)
	for p in range(NPROCS):
		m.addConstr( index_variables[p] == -gamma*LinExpr(T[p, start_state[p], 0], L[p]) + gamma*LinExpr(T[p, start_state[p], 1], L[p]) ) 


	start = time.time()

	# Optimize model
	m.optimize()
	# print("model status",m.status)
	# m.printStats()
	# obj = m.getObjective()
	# print("obj",obj.getValue())

	start = time.time()


	L_vals = np.zeros((NPROCS,NSTATES))
	index_solved_values = np.zeros(NPROCS)
	z_vals = np.zeros((NPROCS,NSTATES,NACTIONS))
	bina_vals = np.zeros((NPROCS,NSTATES,NACTIONS))
	for v in m.getVars():
		if 'index' in v.varName:
			i = int(v.varName.split('_')[1])
			index_solved_values[i] = v.x

		if 'L_' in v.varName:
			i = int(v.varName.split('_')[1])
			j = int(v.varName.split('_')[2])

			L_vals[i,j] = v.x

		if 'z_' in v.varName:
			p = int(v.varName.split('_')[1])
			i = int(v.varName.split('_')[2])
			j = int(v.varName.split('_')[3])

			z_vals[p,i,j] = v.x

		if 'binary_' in v.varName:
			p = int(v.varName.split('_')[1])
			i = int(v.varName.split('_')[2])
			j = int(v.varName.split('_')[3])
			bina_vals[p,i,j] = v.x


	# print('Variables extracted in %ss:'%(time.time() - start))
	start = time.time()

	return L_vals, index_solved_values, z_vals, bina_vals







# F2



def bqp_to_optimize_index(p01p_range, p11p_range, p01a_range, p11a_range, R, C, start_state, maximize=True, gamma=0.95, lambda_lim=None):


	start = time.time()

	NSTATES = 2
	NACTIONS = 2

	# Create a new model
	m = Model("BQP for Maximizing/Minimizing the Whittle index")
	m.setParam( 'OutputFlag', False )

	L = np.zeros(NSTATES,dtype=object)
	
	mu = np.zeros(NSTATES, dtype=float)
	mu[int(start_state)] = 1


	# Create variables
	lb = -R.max()/(1-gamma)*10
	ub = R.max()/(1-gamma)*10
	if lambda_lim is not None:
		ub = lambda_lim


	index_variable = m.addVar(vtype=GRB.CONTINUOUS, lb=lb, ub=ub, name='index')

	for i in range(NSTATES):
		L[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=lb, ub=ub, name='L_%s'%i)

	p01p = m.addVar(vtype=GRB.CONTINUOUS, lb=p01p_range[0], ub=p01p_range[1], name='p01p')
	p00p = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name='p00p')
	m.addConstr(p00p == 1 - p01p)

	p11p = m.addVar(vtype=GRB.CONTINUOUS, lb=p11p_range[0], ub=p11p_range[1], name='p11p')
	p10p = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name='p10p')
	m.addConstr(p10p == 1 - p11p)

	p01a = m.addVar(vtype=GRB.CONTINUOUS, lb=p01a_range[0], ub=p01a_range[1], name='p01a')
	p00a = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name='p00a')
	m.addConstr(p00a == 1 - p01a)

	p11a = m.addVar(vtype=GRB.CONTINUOUS, lb=p11a_range[0], ub=p11a_range[1], name='p11a')
	p10a = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name='p10a')
	m.addConstr(p10a == 1 - p11a)

	T = np.zeros((NSTATES,NACTIONS,NSTATES),dtype=object)

	T[0,0,0] = p00p
	T[0,0,1] = p01p
	T[1,0,0] = p10p
	T[1,0,1] = p11p

	T[0,1,0] = p00a
	T[0,1,1] = p01a
	T[1,1,0] = p10a
	T[1,1,1] = p11a

	# Make each row of T sum to 1
	# for i in range(NSTATES):
	# 	for j in range(NACTIONS):
	# 		m.addConstr(T[i,j].sum() == 1)
 

	z = np.zeros((NSTATES,NACTIONS),dtype=object)
	bina = np.zeros((NSTATES,NACTIONS),dtype=object)
	for i in range(NSTATES):
		for j in range(NACTIONS):
			z[i,j] = m.addVar(vtype=GRB.CONTINUOUS, lb=lb, ub=ub, name='z_%s_%s'%(i,j))
			bina[i,j] = m.addVar(vtype=GRB.BINARY, name='binary_%s_%s'%(i,j))

	L = np.array(L)


	# print('Variables added in %ss:'%(time.time() - start))
	start = time.time()


	m.modelSense=GRB.MINIMIZE

	# Set objective
	# m.setObjectiveN(obj, index, priority) -- larger priority = optimize first

	M = 1e2
	# in the BLP construction, we don't need to minimize over Q constraints -- binary vars will ensure tightness
	# m.setObjectiveN(1, 0, 1)
	# m.setObjectiveN(sum([L[i].dot(mu[i]) for i in range(NPROCS)]) + index_variables[i]*B*((1-gamma)**-1), 0, 1)

	# maximize objective first of all
	if maximize:
		m.setObjectiveN(-index_variable, 0, 1)
	else:
		m.setObjectiveN(index_variable, 0, 1)
	# then, need to solve the value functions
	m.setObjectiveN(L.dot(mu), 1, 0)

	# obj1 = m.getObjective(0)
	# obj2 = m.getObjective(1)

	# print("obj1")
	# print('priority',obj1.ObjNPriority)
	# print('weight',obj1.ObjNWeight)
	# print('rel tol',obj1.ObjNRelTol)
	# print('abs tol',obj1.ObjNAbsTol)
	# print()
	# print("obj2")
	# print('priority',obj2.ObjNPriority)
	# print('weight',obj2.ObjNWeight)
	# print('rel tol',obj2.ObjNRelTol)
	# print('abs tol',obj2.ObjNAbsTol)
	# print()
	# 1/0


	bigM = M
	eq_epsilon = 0#1e-2
	# import pdb; pdb.set_trace()
	# set constraints to figure out what value SHOULD be in value-iteration land
	for i in range(NSTATES):
		for j in range(NACTIONS):
			# m.addConstr( L[p][i] >= R[p][i] - index_variable*c[j] + gamma*L[p].dot(T[p,i,j]) )
			# m.addConstr( L[i] >= R[i] - index_variable*C[j] + gamma*(T[i,j,0]*L[0] + T[i,j,1]*L[1]) )
			# m.addConstr( L[i] >= R[i] - index_variable*C[j] + gamma* sum([T[i,j,s]*L[s] for s in range(NSTATES) ]) )  
			# m.addConstr( z[i,j] == R[i] - index_variable*C[j] + gamma*sum([T[i,j,s]*L[s] for s in range(NSTATES) ]))
			m.addConstr( L[i] >= R[i] - index_variable*C[j] + gamma*L.dot(T[i,j]) )  
			m.addConstr( z[i,j] == R[i] - index_variable*C[j] + gamma*L.dot(T[i,j]) )
			# m.addConstr( z[i,j] - (R[i] - index_variable*C[j] + gamma*sum([T[i,j,s]*L[s] for s in range(NSTATES) ])) <= eq_epsilon )
			# m.addConstr( z[i,j] - (R[i] - index_variable*C[j] + gamma*sum([T[i,j,s]*L[s] for s in range(NSTATES) ])) >= -eq_epsilon )
			m.addConstr( L[i] <= z[i,j] + bina[i,j]*bigM + eq_epsilon)

		# ensure that only one of the upper bound constraints on the value function is tight
		# by construction, the only way to satisfy this in conjunction with the above constraints
		# is if the largest lower bound constraint is also the only active upper bound constraint
		m.addConstr(bina[i].sum() == NACTIONS - 1) # any binary var set to 1 will be a loose upper bound via bigM


	# enforce the whittle index constraint 
	
	m.addConstr( index_variable == -gamma*L.dot(T[start_state,0]) + gamma*L.dot(T[start_state,1]) ) 
	# m.addConstr( index_variable - (-gamma*sum([T[i,0,s]*L[s] for s in range(NSTATES) ]) + gamma*sum([T[i,1,s]*L[s] for s in range(NSTATES) ])) <= eq_epsilon )
	# m.addConstr( index_variable - (-gamma*sum([T[i,0,s]*L[s] for s in range(NSTATES) ]) + gamma*sum([T[i,1,s]*L[s] for s in range(NSTATES) ])) >= -eq_epsilon ) 


	start = time.time()

	m.params.NonConvex=2

	# Optimize model
	m.optimize()
	# print("model status",m.status)
	# m.printStats()
	# obj = m.getObjective()
	# print("obj",obj.getValue())

	start = time.time()


	L_vals = np.zeros(NSTATES)
	index_solved_value = None
	z_vals = np.zeros((NSTATES,NACTIONS))
	bina_vals = np.zeros((NSTATES,NACTIONS))
	for v in m.getVars():
		if 'index' in v.varName:
			index_solved_value = v.x

		if 'L_' in v.varName:
			i = int(v.varName.split('_')[1])

			L_vals[i] = v.x

		if 'z_' in v.varName:
			i = int(v.varName.split('_')[1])
			j = int(v.varName.split('_')[2])

			z_vals[i,j] = v.x

		if 'binary_' in v.varName:
			i = int(v.varName.split('_')[1])
			j = int(v.varName.split('_')[2])
			bina_vals[i,j] = v.x


	T_return = np.zeros(T.shape)
	for s in range(T.shape[0]):
		for a in range(T.shape[1]):
			for sp in range(T.shape[2]):
				T_return[s,a,sp] = T[s,a,sp].x



	# print('Variables extracted in %ss:'%(time.time() - start))
	start = time.time()

	return index_solved_value, L_vals, z_vals, bina_vals, T_return














# F3

# key difference between this method and the one that maxs/mins for only one state 
# at a time, is that we need to create separate copies of the value function
# for each index to solve for each index value
# however, we want to return a single copy of the transition probabilities
# that give us the indexes we want so, only make one copy of those
def bqp_to_optimize_index_both_states(p01p_range, p11p_range, p01a_range, p11a_range, R, C, 
											senses=['min','min'], gamma=0.95, lambda_lim=None, time_limit=None):


	start = time.time()

	NSTATES = 2
	NACTIONS = 2

	# convert min and max to their respective 'senses', i.e.,
	# 1 for min and -1 for max, since the model will minimize by default
	senses = [ -1 if i=='max' else 1 for i in senses]

	# Create a new model
	m = Model("BQP for Maximizing/Minimizing the Whittle index")
	m.setParam( 'OutputFlag', False )
	if time_limit is not None:
		m.setParam('TimeLimit', time_limit)

	# need to create one copy per index
	L = np.zeros((NSTATES,NSTATES),dtype=object)
	mu = np.zeros((NSTATES,NSTATES), dtype=float)

	# looping over number of indexes -- need a mu for each copy of the value function
	# np.eye would be faster, but this is maybe slightly more interpretable
	for s in range(NSTATES):
		mu[s, s] = 1


	# Create variables
	# TODO: these bounds are too loose for the index, but also technically too tight for the value function
	# so we should find better bounds -- tight bounds are crucial for faster solving
	lb = -R.max()/(1-gamma)
	ub = R.max()/(1-gamma)
	if lambda_lim is not None:
		ub = lambda_lim


	index_variables = np.zeros(NSTATES,dtype=object)
	for s in range(NSTATES):
		index_variables[s] = m.addVar(vtype=GRB.CONTINUOUS, lb=lb, ub=ub, name='index_%s'%s)


	lb = 0#lb*5
	ub = ub*5
	# create one copy for each index
	for index in range(index_variables.shape[0]):
		for s in range(NSTATES):
			L[index,s] = m.addVar(vtype=GRB.CONTINUOUS, lb=lb, ub=ub, name='L_%s%s'%(index,s))


	# need one 
	p01p = m.addVar(vtype=GRB.CONTINUOUS, lb=p01p_range[0], ub=p01p_range[1], name='p01p')
	p00p = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name='p00p')
	m.addConstr(p00p == 1 - p01p)

	p11p = m.addVar(vtype=GRB.CONTINUOUS, lb=p11p_range[0], ub=p11p_range[1], name='p11p')
	p10p = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name='p10p')
	m.addConstr(p10p == 1 - p11p)

	p01a = m.addVar(vtype=GRB.CONTINUOUS, lb=p01a_range[0], ub=p01a_range[1], name='p01a')
	p00a = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name='p00a')
	m.addConstr(p00a == 1 - p01a)

	p11a = m.addVar(vtype=GRB.CONTINUOUS, lb=p11a_range[0], ub=p11a_range[1], name='p11a')
	p10a = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name='p10a')
	m.addConstr(p10a == 1 - p11a)

	T = np.zeros((NSTATES,NACTIONS,NSTATES),dtype=object)

	T[0,0,0] = p00p
	T[0,0,1] = p01p
	T[1,0,0] = p10p
	T[1,0,1] = p11p

	T[0,1,0] = p00a
	T[0,1,1] = p01a
	T[1,1,0] = p10a
	T[1,1,1] = p11a

	# "Natural" constraints

	# better to be in the good state already
	m.addConstr(p11a >= p01a)
	m.addConstr(p11p >= p01p)

	# better to act than not
	m.addConstr(p11a >= p11p)
	m.addConstr(p01a >= p01p)

	# p11a needs to have highest boundary
	# p01a is between p11a and p01p
	# p01p needs to go no higher than the lower of p01a and p11p
	# p11p is betwen p01p and p11a
	#

	# so: 
	#    p11a:
	#        upper bound should be equal or higher than all ubs 
	#        lower bound can be anything
	#    p01a:
	#        upper bound should be higher than p01p
	#        lower bound should be 


	# Make each row of T sum to 1
	# for i in range(NSTATES):
	# 	for j in range(NACTIONS):
	# 		m.addConstr(T[i,j].sum() == 1)
 

	# need a copy of these for each index too
	z = np.zeros((NSTATES, NSTATES, NACTIONS),dtype=object)
	bina = np.zeros((NSTATES, NSTATES, NACTIONS),dtype=object)
	for index in range(index_variables.shape[0]):
		for s in range(NSTATES):
			for a in range(NACTIONS):
				z[index,s,a] = m.addVar(vtype=GRB.CONTINUOUS, lb=lb, ub=ub, name='z_%s_%s_%s'%(index,s,a))
				bina[index,s,a] = m.addVar(vtype=GRB.BINARY, name='binary_%s_%s_%s'%(index,s,a))




	# print('Variables added in %ss:'%(time.time() - start))
	start = time.time()


	m.modelSense=GRB.MINIMIZE

	

	M = 1e3
	# in the BLP construction, we don't need to minimize over Q constraints -- binary vars will ensure tightness
	# m.setObjectiveN(1, 0, 1)
	# m.setObjectiveN(sum([L[i].dot(mu[i]) for i in range(NPROCS)]) + index_variables[i]*B*((1-gamma)**-1), 0, 1)


	# Set objective
	# m.setObjectiveN(obj, index, priority) -- larger priority = optimize first

	# the signs in sense will take care of min/maxing as needed -- give this highest priority
	print('senses',senses)
	m.setObjectiveN(index_variables.dot(senses), 0, 1)

	# then, need to solve the value functions, give lower priority
	m.setObjectiveN(sum(L[index].dot(mu[index]) for index in range(NSTATES)), 1, 0)

	# obj1 = m.getObjective(0)
	# obj2 = m.getObjective(1)

	# print("obj1")
	# print('priority',obj1.ObjNPriority)
	# print('weight',obj1.ObjNWeight)
	# print('rel tol',obj1.ObjNRelTol)
	# print('abs tol',obj1.ObjNAbsTol)
	# print()
	# print("obj2")
	# print('priority',obj2.ObjNPriority)
	# print('weight',obj2.ObjNWeight)
	# print('rel tol',obj2.ObjNRelTol)
	# print('abs tol',obj2.ObjNAbsTol)
	# print()
	# 1/0


	bigM = M
	eq_epsilon = 0#1e-2
	# import pdb; pdb.set_trace()
	# set constraints to figure out what value SHOULD be in value-iteration land
	for index in range(NSTATES):
		for i in range(NSTATES):
			for j in range(NACTIONS):
				# m.addConstr( L[p][i] >= R[p][i] - index_variable*c[j] + gamma*L[p].dot(T[p,i,j]) )
				# m.addConstr( L[i] >= R[i] - index_variable*C[j] + gamma*(T[i,j,0]*L[0] + T[i,j,1]*L[1]) )
				# m.addConstr( L[i] >= R[i] - index_variable*C[j] + gamma* sum([T[i,j,s]*L[s] for s in range(NSTATES) ]) )  
				# m.addConstr( z[i,j] == R[i] - index_variable*C[j] + gamma*sum([T[i,j,s]*L[s] for s in range(NSTATES) ]))
				m.addConstr( L[index,i] >= R[i] - index_variables[index]*C[j] + gamma*L[index].dot(T[i,j]) )  
				m.addConstr( z[index,i,j] == R[i] - index_variables[index]*C[j] + gamma*L[index].dot(T[i,j]) )
				# m.addConstr( z[i,j] - (R[i] - index_variable*C[j] + gamma*sum([T[i,j,s]*L[s] for s in range(NSTATES) ])) <= eq_epsilon )
				# m.addConstr( z[i,j] - (R[i] - index_variable*C[j] + gamma*sum([T[i,j,s]*L[s] for s in range(NSTATES) ])) >= -eq_epsilon )
				m.addConstr( L[index,i] <= z[index,i,j] + bina[index,i,j]*bigM + eq_epsilon)

			# ensure that only one of the upper bound constraints on the value function is tight
			# by construction, the only way to satisfy this in conjunction with the above constraints
			# is if the largest lower bound constraint is also the only active upper bound constraint
			m.addConstr(bina[index,i].sum() == NACTIONS - 1) # any binary var set to 1 will be a loose upper bound via bigM


	# enforce the whittle index constraint 
	for index in range(NSTATES):
		m.addConstr( index_variables[index] == -gamma*L[index].dot(T[index,0]) + gamma*L[index].dot(T[index,1]) ) 
		# m.addConstr( index_variable - (-gamma*sum([T[i,0,s]*L[s] for s in range(NSTATES) ]) + gamma*sum([T[i,1,s]*L[s] for s in range(NSTATES) ])) <= eq_epsilon )
		# m.addConstr( index_variable - (-gamma*sum([T[i,0,s]*L[s] for s in range(NSTATES) ]) + gamma*sum([T[i,1,s]*L[s] for s in range(NSTATES) ])) >= -eq_epsilon ) 


	start = time.time()

	m.params.NonConvex=2

	# Optimize model
	m.optimize()
	print("model status",m.status)
	# m.printStats()
	# obj = m.getObjective()
	# print("obj",obj.getValue())

	start = time.time()


	# index_solved_values = np.zeros(NSTATES)
	# L_vals = np.zeros((NSTATES, NSTATES))
	# z_vals = np.zeros((NSTATES, NSTATES, NACTIONS))
	# bina_vals = np.zeros((NSTATES, NSTATES, NACTIONS))
	# for v in m.getVars():
	# 	if 'index' in v.varName:
	# 		i = int(v.varName.split('_')[1])

	# 		index_solved_value = v.x

	# 	if 'L_' in v.varName:
	# 		i = int(v.varName.split('_')[1])

	# 		L_vals[i] = v.x

	# 	if 'z_' in v.varName:
	# 		i = int(v.varName.split('_')[1])
	# 		j = int(v.varName.split('_')[2])

	# 		z_vals[i,j] = v.x

	# 	if 'binary_' in v.varName:
	# 		i = int(v.varName.split('_')[1])
	# 		j = int(v.varName.split('_')[2])
	# 		bina_vals[i,j] = v.x

	index_solved_values = np.zeros(NSTATES)
	for index in range(NSTATES):
		index_solved_values[index] = index_variables[index].x

	L_vals = np.zeros((NSTATES, NSTATES))
	for index in range(NSTATES):
		for s in range(NSTATES):
			L_vals[index,s] = L[index,s].x


	z_vals = np.zeros((NSTATES, NSTATES, NACTIONS))
	for index in range(NSTATES):
		for s in range(NSTATES):
			for a in range(NACTIONS):
				z_vals[index,s,a] = z[index,s,a].x

	bina_vals = np.zeros((NSTATES, NSTATES, NACTIONS))
	for index in range(NSTATES):
		for s in range(NSTATES):
			for a in range(NACTIONS):
				bina_vals[index,s,a] = bina[index,s,a].x


	T_return = np.zeros(T.shape)
	for s in range(T.shape[0]):
		for a in range(T.shape[1]):
			for sp in range(T.shape[2]):
				T_return[s,a,sp] = T[s,a,sp].x



	# print('Variables extracted in %ss:'%(time.time() - start))
	start = time.time()

	return index_solved_values, L_vals, z_vals, bina_vals, T_return







# F4

def check_feasible_range(p01p_range, p11p_range, p01a_range, p11a_range):


	start = time.time()

	# Create a new model
	m = Model("Quick program for checking feasibility")
	m.setParam( 'OutputFlag', False )

	# need one 
	p01p = m.addVar(vtype=GRB.CONTINUOUS, lb=p01p_range[0], ub=p01p_range[1], name='p01p')
	p00p = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name='p00p')
	m.addConstr(p00p == 1 - p01p)

	p11p = m.addVar(vtype=GRB.CONTINUOUS, lb=p11p_range[0], ub=p11p_range[1], name='p11p')
	p10p = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name='p10p')
	m.addConstr(p10p == 1 - p11p)

	p01a = m.addVar(vtype=GRB.CONTINUOUS, lb=p01a_range[0], ub=p01a_range[1], name='p01a')
	p00a = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name='p00a')
	m.addConstr(p00a == 1 - p01a)

	p11a = m.addVar(vtype=GRB.CONTINUOUS, lb=p11a_range[0], ub=p11a_range[1], name='p11a')
	p10a = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name='p10a')
	m.addConstr(p10a == 1 - p11a)

	T = np.zeros((2,2,2),dtype=object)

	T[0,0,0] = p00p
	T[0,0,1] = p01p
	T[1,0,0] = p10p
	T[1,0,1] = p11p

	T[0,1,0] = p00a
	T[0,1,1] = p01a
	T[1,1,0] = p10a
	T[1,1,1] = p11a

	# "Natural" constraints

	# better to be in the good state already
	m.addConstr(p11a >= p01a)
	m.addConstr(p11p >= p01p)

	# better to act than not
	m.addConstr(p11a >= p11p)
	m.addConstr(p01a >= p01p)

	# p11a needs to have highest boundary
	# p01a is between p11a and p01p
	# p01p needs to go no higher than the lower of p01a and p11p
	# p11p is betwen p01p and p11a
	#

	# so: 
	#    p11a:
	#        upper bound should be equal or higher than all ubs 
	#        lower bound can be anything
	#    p01a:
	#        upper bound should be higher than p01p
	#        lower bound should be 

	# print('Variables added in %ss:'%(time.time() - start))



	# Optimize model
	m.optimize()
	print("model status",m.status)

	feasible = int(m.status) != 3

	return feasible









