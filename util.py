import torch
import numpy as np
import json
import gzip
def Relu(x):
    return torch.maximum(x, torch.tensor(0))
    
def read_netgen(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        
    print(f'Total NETGEN lines num is {len(lines)}')
    #read in the parameters
    num_sources = int(lines[8].split(':')[1].strip())
    num_targets = int(lines[9].split(':')[1].strip())
    num_total = int(lines[7].split(':')[1].strip())
    num_mid = num_total - num_sources - num_targets
    
    minimum_cost = int(lines[11].split(':')[1].strip())
    maximum_cost = int(lines[12].split(':')[1].strip())
    total_supply = int(lines[13].split(':')[1].strip())
    min_capacity = int(lines[20].split(':')[1].strip())
    max_capacity = int(lines[21].split(':')[1].strip())
    
    trans_source = int(lines[15].split(':')[1].strip())
    trans_sink = int(lines[16].split(':')[1].strip())
    
    total_edges = int(lines[10].split(':')[1].strip())
    
    # the node information
    node_info = lines[26:26 + num_sources + num_targets]
    source_flows = np.array([float(line.split()[2]) / total_supply for line in node_info[:num_sources]])
    target_flows = np.array([np.abs(float(line.split()[2])) / total_supply for line in node_info[num_sources:]])
    print(f'the source flows are {np.sum(source_flows)}')
    print(f'the target flows are {np.sum(target_flows)}')
    # the edge information
    edge_infos = lines[26 + num_sources + num_targets:]
    num_edges = len(edge_infos)
    print(f'The edge num is {num_edges}')
    
    distance_matrix = -1 * np.ones((num_total, num_total))
    capacity_matrix = np.ones((num_total, num_total))
    
    for i, edge_info in enumerate(edge_infos):
        parts = edge_info.split()
        from_node = int(parts[1]) - 1
        to_node = int(parts[2]) - 1
        min_cap = float(parts[3])
        max_cap = float(parts[4])
        cost = float(parts[5])
        distance_matrix[from_node][to_node] = cost
        capacity_matrix[from_node][to_node] = max_cap / total_supply
    
    unreachable_cost = max([float(line.split()[5]) for line in edge_infos]) * 10
    distance_matrix = np.where(distance_matrix == -1, unreachable_cost, distance_matrix)
    return source_flows, target_flows, distance_matrix, capacity_matrix, unreachable_cost


def read_uniform(file_name, comprese_data = False, multi_seed = False, dir = False):

    if not comprese_data:
        with open(file_name, 'r') as json_file:
            data_dict = json.load(json_file)
    else:
        with gzip.open(file_name+".gz", 'rt', encoding='utf-8') as file:
            data_dict = json.load(file)
    # file_name = data_dict["name"]
    # if multi_seed == False:
    #     prefix, node_a, node_b, node_c, _ = file_name[:-5].split('_')
    # else:
    #     if not dir:
    #         prefix, node_a, node_b, node_c,seed = file_name[:-5].split('_')
    #     else:
    #         prefix, node_a, node_b, node_c,seed,_ = file_name[:-5].split('_')

    # print(f'source point nums:{int(node_a)} \n mid point nums :{int(node_b)} \n target point nums : {int(node_c)}')
    ## the source marginals
    a = np.array(data_dict["a"])
    ## the target marginals
    b = np.array(data_dict["b"])
    ## the distance matrix
    M = np.array(data_dict["M"])
    ##the double precision point 
    xs = np.array(data_dict["xs"])
    xm = np.array(data_dict["xm"])
    xt = np.array(data_dict["xt"])
    ##the scaling factor
    MAX_INT = np.array(data_dict["MAXINT"])
    # print("have read the file that name is :", file_name)
    return a, b, M, xs, xm, xt, MAX_INT

