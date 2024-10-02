import time
import argparse
import numpy as np
import ot
import os
import torch
import sys
import json
import gzip
import matplotlib.pyplot as plt
from EOFT.alg import eoft_EN,eoft_WO
from EOFT.util import Relu, read_uniform, read_netgen
from EOFT.gurobi import  gurobi_EN
np.set_printoptions(threshold=np.inf)



def main(args):
    # set_type = args.name
    data_type = args.data_type ## including the 'netgen', 'uniform_WO', 'uniform_EN'
    data_path = args.data_path ## the path of your data 
    a_samples = args.a_samples
    b_samples = args.b_samples
    c_samples = args.c_samples
    n_samples = a_samples+b_samples+c_samples##total number of data node
    eps =args.eps

    torch.manual_seed(args.seed)
    

     
    print(f'Start to read from the data {data_path} which is the {data_type} Dataset')
    # data_name += name_set + "_.json"
    
    
    
    if data_type == 'netgen':
        print(f'----------now we are solving netgen problem----------')
        r_ = []         ## the source marginals
        c_ = []         ## the tart marginals
        M_combined = [] ## the cost matrix
        S_netgen = []   ## the edge  capacity matrix
        MAX_INT = []    ## the scaling factor of the cost matrix
        
        for i in range(args.batch_size):
            data_name_i = data_path+"_"+f'{args.batch_size}'+f'_{i}'
            print(f'read from the {data_name_i} of batch {args.batch_size}')
            r_i, c_i, M_combined_i, S_netgen_i, MAX_INT_i = read_netgen(data_name_i)
            r_.append(r_i)
            c_.append(c_i)
            np.fill_diagonal(M_combined_i, 0.0)
            M_combined.append(M_combined_i)
            S_netgen.append(S_netgen_i)
            MAX_INT.append(MAX_INT_i)
           
        r_ = np.array(r_)
        c_ = np.array(c_)
        print(f'----------the total flow  of source set is {(np.sum(r_))} and the total flow of target set is {(np.sum(c_))}----------')
        M_combined = np.array(M_combined)
        S_netgen = np.array(S_netgen)
        MAX_INT = np.array(MAX_INT)
        print(f'the shape of Dist matrix is {M_combined.shape} and the shape of Capacity matrix is {S_netgen.shape}')
        Edge_C_matrix = torch.from_numpy(S_netgen)
        print(f'the NETGEN CAPACITY max {torch.max(Edge_C_matrix)} and min {torch.min(Edge_C_matrix)} and mean {torch.mean(Edge_C_matrix)}')
                
        
    else:
        r_, c_, M_combined, xs_, xm_, xt_, MAX_INT = read_uniform(data_path,False,True, True)


    M_combined = torch.from_numpy(M_combined)
    r = r_
    c = c_
    print(f'the shape of Dist matrix .Source marginals.Target marginals is {M_combined.shape} and {r.shape} and {c.shape}')
    MAX_INT = torch.tensor(MAX_INT)
    M_tmp = M_combined.clone()                  ## save the original dist matrix

    for i in range(args.batch_size):

        M_combined[i] /= MAX_INT[i]             ## scale the dist matrix


    if data_type == "uniform_WO" :
        print(f'----------now we are solving uniform WO constraints problem----------')
        INTMAX = 99999
        S0 = torch.ones_like(M_tmp) * INTMAX
        N0 = torch.ones_like(M_tmp) * INTMAX
        Gurobi_sol = []                         ## the list to store the gurobi solution
        Gurobi_cost = []                        ## the list to store the gurobi cost

        Node_array = torch.ones(args.batch_size,n_samples) * INTMAX
        Edge_C_matrix = torch.ones_like(M_combined) * INTMAX

        if args.need_gurobi: ##calculate gurobi solution
            tg1 = time.time()
            for i in range(args.batch_size):
                
                PM, cost_gurobi, _ = gurobi_EN(r[i], c[i], M_tmp[i], a_samples, b_samples, c_samples,  Edge_C_matrix[i], Node_array[i])       
                Gurobi_sol.append(PM)
                Gurobi_cost.append(cost_gurobi)
            
            Gurobi_cost = np.stack(Gurobi_cost)
            Gurobi_sol = np.stack(Gurobi_sol)
            tg2 = time.time()
            
            time_gurobi = tg2 - tg1
            Gurobi_cost = torch.tensor(Gurobi_cost)
            Gurobi_tensor = torch.tensor(Gurobi_sol)

        P, time_eoft = eoft_WO(args.batch_size,M_combined.to('cuda'), eps,r,c ,args.err, args.iters, a_samples, b_samples, args.d0)

        P_eoft =  []        
        ## P - P.T to avoid self loop
        for batch_ in range(args.batch_size):  
            P_eoft.append(Relu(P[batch_] - P[batch_].T).cpu())
        P_eoft = torch.stack(P_eoft)
        
        cost_eoft = torch.sum(P_eoft * M_tmp, dim=(1,2))
        


        
        print(f'----------------------------------------------------------')
        print(f'EOFT Obj is {cost_eoft.mean().item()}')
        print(f'EOFT Time(s) is {time_eoft}')
        if args.need_gurobi :
            print(f'Gurobi Obj {Gurobi_cost.mean().item()}') 
            print(f'gurobi Time(s) is {time_gurobi}')
        print(f'----------------------------------------------------------')
        
        
    # elif type_1 == "mcf_W_EN_con":             
    elif data_type == "netgen":#专门给netge数据集准备的算法（主要的区别是不存在节点的约束，另外容量约束是node wise的）
        INTMAX = 99999
        Node_array = torch.ones(args.batch_size, n_samples) * INTMAX
        Gurobi_cost = []
        Gurobi_sol = []
        if args.need_gurobi:
            tg1 = time.time()
            for i in range(args.batch_size):## the NEGEN only have capacity constraint on edges
                PM, cost_gurobi, _ = gurobi_EN(r[i], c[i], M_tmp[i], a_samples, b_samples, c_samples,  Edge_C_matrix[i], None)
                Gurobi_sol.append(PM)
                Gurobi_cost.append(cost_gurobi)        
            tg2 = time.time()
            Gurobi_cost = np.stack(Gurobi_cost)
            Gurobi_sol = np.stack(Gurobi_sol)
            time_gurobi = tg2 - tg1
            print(f'------------the time of gurobi is {time_gurobi}---------------')
            Gurobi_cost = torch.tensor(Gurobi_cost)
            Gurobi_tensor = torch.tensor(Gurobi_sol)
            


        P, time_eoft = eoft_EN(args.batch_size,M_combined.to('cuda'), eps,r,c ,args.err, args.iters, a_samples, b_samples, args.d0, Edge_C=Edge_C_matrix, Node_C= None)
       
        P_eoft =  [] 
        for batch_ in range(args.batch_size):  
            P_eoft.append(Relu(P[batch_] - P[batch_].T).cpu())
        P_eoft = torch.stack(P_eoft)
        cost_eoft = torch.sum(P_eoft * M_tmp, dim=(1,2))

        
        
        print(f'----------------------------------------------------------')
        print(f'EOFT Obj is {cost_eoft.mean().item()}')
        print(f'EOFT Time(s) is {time_eoft}')
        if args.need_gurobi :
            print(f'Gurobi Obj {Gurobi_cost.mean().item()}') 
            print(f'gurobi Time(s) is {time_gurobi}')
        print(f'----------------------------------------------------------')
        
        
    else:## means it is uniform_EN
        print(f'----------now we are solving uniform W Edge and Node constraints problem----------')
 
        Edge_C_matrix = torch.ones_like(M_combined) * args.Edge_C
        Node_array = torch.ones(args.batch_size, n_samples) * args.Node_C
        Gurobi_cost = []
        Gurobi_sol = []
        
        if args.need_gurobi:
            tg1 = time.time()
            for i in range(args.batch_size):
                # if args.need_gurobi:
                    # tg1 = time.time()
                    # PM, cost_gurobi, _ = gurobi_SN(r, c, M_tmp, a_samples, b_samples, c_samples, d0, S_max, N_max, need_S = True,need_N = True , P_output= False)
                PM, cost_gurobi, _ = gurobi_EN(r[i], c[i], M_tmp[i], a_samples, b_samples, c_samples,  Edge_C_matrix[i], Node_array[i], need_S = True,need_N = True , P_output= False)
                    
                Gurobi_sol.append(PM)
                Gurobi_cost.append(cost_gurobi)
                Gurobi_cost = np.stack(Gurobi_cost)
                Gurobi_sol = np.stack(Gurobi_sol)
            tg2 = time.time()
            time_gurobi = tg2 - tg1
            print(f'------------the time of gurobi is {time_gurobi}---------------')
            Gurobi_cost = torch.tensor(Gurobi_cost)
            Gurobi_tensor = torch.tensor(Gurobi_sol)
            
        Node_C = torch.tensor(args.Node_C)
        P, time_eoft= eoft_EN(args.batch_size,M_combined.to('cuda'), eps,r,c ,args.err, args.iters, a_samples, b_samples, args.d0, Edge_C=Edge_C_matrix, Node_C= Node_C)
        

        
        P_eoft =  []
        for batch_ in range(args.batch_size):  
            P_eoft.append(Relu(P[batch_] - P[batch_].T).cpu())
        P_eoft = torch.stack(P_eoft)
        cost_eoft = torch.sum(P_eoft * M_tmp, dim=(1,2))
        

        print(f'----------------------------------------------------------')
        print(f'EOFT Obj is {cost_eoft.mean().item()}')
        print(f'EOFT Time(s) is {time_eoft}')
        if args.need_gurobi :
            print(f'Gurobi Obj {Gurobi_cost.mean().item()}') 
            print(f'gurobi Time(s) is {time_gurobi}')
        print(f'----------------------------------------------------------')



def parse_arguments(args_to_parse):
    """Parse the command line arguments.
    Parameters
    ----------
    args_to_parse: list of str
        Arguments to parse (splitted on whitespaces).
    """
    description = "PyTorch implementation and evaluation of EOFT-Sinkhorn."
    parser = argparse.ArgumentParser(description=description)
    # General options
    general = parser.add_argument_group('General options')
    general.add_argument('--data_type', type=str,
                    help="the type of dataset, including the netgen dataset, uniform dataset with no constraint, uniform dataset with edge and node constraint.",
                    choices=['netgen', 'uniform_WO', 'uniform_EN']) 
    
    general.add_argument('--data_path', type=str,
                    help="The path to read data. Please replace with the path to your own data.") 
    
    general.add_argument('--seed', type=int, default=112,
                         help='Random seed. Can be `None` for stochastic behavior.')
    # Model Options
    model = parser.add_argument_group('Model specfic options')
    model.add_argument('-a', '--a_samples', type=int, default=20,
                          help="The node num of source points.")
    model.add_argument('-b', '--b_samples', type=int, default=20,
                          help="The node num of mid points.")
    model.add_argument('-c', '--c_samples', type=int, default=20,
                          help="The node num of target points.")
    model.add_argument('--err', type=float, default=1e-8,
                          help="The stop err of in the algorithm ")

    model.add_argument('--batch_size', type=int, default=1,
                          help="The batchsize of data")
    
    model.add_argument('--Node_C', type=float, default=1.0,
                          help="The capacity constraint for eace node, mainly for uniform dataset, ")
    model.add_argument('--Edge_C', type=float, default=1.0,
                          help="The capacity constrint for each edge, mainly for uniform dataset ")
    
    model.add_argument('--iters', type=int, default=1000,
                          help="The  maximum iters of in the algorithm ")
    
    model.add_argument('--need_gurobi', action='store_true',
                    help="Whether you want to compare with gurobi.")
       
    # Learning option`
    training = parser.add_argument_group('Training specific options')
    training.add_argument('--eps', type=float, default=0.003,
                          help="The eplision of the entropic item P.")
    training.add_argument('--d0', type=float, default=1e-8,
                          help='The num virtual flow')

    args = parser.parse_args(args_to_parse)
    return args

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
