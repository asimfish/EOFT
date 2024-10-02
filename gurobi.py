import scipy as sp
import numpy as np
import ot
import gurobipy as gp
from gurobipy import GRB
import time 

def gurobi_EN(r0,c0,M_combined,a_sample,b_sample,c_sample, S0=None, N0=None,need_S = False,need_N=False):

    a_samples = a_sample
    b_samples = b_sample
    c_samples = c_sample
    n_samples = a_samples + b_samples + c_samples  # nb samples
    r = r0
    c = c0
    np.random.seed(0)

    r_c = np.zeros(n_samples)
    r_c[:a_samples] = r
    r_c[a_samples + b_samples:] = c
    r_c[a_samples + b_samples:] = [-x for x in r_c[a_samples + b_samples:]]


    P_ = {}  
    t1 =time.time()
    model = gp.Model()

    P = model.addVars(n_samples, n_samples, lb=0, vtype=GRB.CONTINUOUS, name="P")
    q = model.addVars(n_samples, name="q")

    M_combined_ = M_combined.clone()
    obj_expr = gp.quicksum(M_combined_[i, j] * P[i, j] for i in range(n_samples) for j in range(n_samples))
    model.setObjective(obj_expr, GRB.MINIMIZE)

    for i in range(n_samples):
        model.addConstr(gp.quicksum(P[i, j] for j in range(n_samples)) == (q[i]))
        model.addConstr(gp.quicksum(P[j, i] for j in range(n_samples)) == (q[i] - r_c[i]))
        model.addConstr(q[i] >= 0)
        model.addConstr((q[i] - r_c[i]) >= 0)
        if N0 != None:
            model.addConstr(q[i] <= N0[i])
        if S0 != None:
            for j_ in range(n_samples):
                model.addConstr(P[i,j_] <= S0[i,j_])  

    model.optimize()
    PM = np.zeros((n_samples, n_samples))
    optim_v = -1
    if model.status == GRB.OPTIMAL:
        for i in range(n_samples):
            for j in range(n_samples):
                P_[(i, j)] = P[i, j].x  
        for i in range(n_samples):
            for j in range(n_samples):
                PM[i, j] = P_[(i, j)]
        optim_v = model.objVal

    else:
        print("do not find optimal solution")
    t2 = time.time()
    return PM, optim_v,t2-t1