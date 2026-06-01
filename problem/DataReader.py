import pickle
from pathlib import Path

import numpy as np
import torch

def _get_saved_data_loaders():
    return {
        'tsp': {
            'pt': use_saved_problems_tsp_pt,
            'pkl': use_saved_problems_tsp_pkl,
            'txt': use_saved_problems_tsp_txt,
        },
        'cvrp': {
            'pt': use_saved_problems_cvrp_pt,
            'pkl': use_saved_problems_cvrp_pkl,
            'txt': use_saved_problems_cvrp_txt,
        },
    }


def get_saved_data(problem, filename, total_episodes, device, start=0, solution_file=None):
    problem = str(problem).lower()
    data_type = Path(filename).suffix.lstrip(".").lower()

    data_loaders = _get_saved_data_loaders()

    if problem not in data_loaders:
        raise ValueError(f"Unsupported problem: {problem}. Supported problems are: {list(data_loaders.keys())}")

    data_loader = data_loaders[problem]
    if data_type not in data_loader:
        raise ValueError(f"Unsupported file type: {data_type}. Supported types are: {list(data_loader.keys())}")

    return data_loader[data_type](filename, total_episodes, device, start, solution_file)

def use_saved_problems_tsp_pt(filename, total_episodes, device, start=0, solution_file=None):

    loaded_dict = torch.load(filename, map_location=device)
    problems = loaded_dict['node_xy'][start:start+total_episodes]
    optimal = None
    if 'optimal' in loaded_dict.keys():
        optimal = loaded_dict['optimal']
    if 'solutions' in loaded_dict.keys():
        solution = loaded_dict['solutions']
        gathering_index =  solution.unsqueeze(2).expand(-1, -1, 2)
        # shape: (batch, problem, 2)
        ordered_seq = problems.gather(dim=1, index=gathering_index)
        rolled_seq = ordered_seq.roll(dims=1, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(2).sqrt()
        # shape: (batch, problem)
        travel_distances = segment_lengths.sum(1)
        # shape: (batch,)
        optimal = travel_distances.mean().item()
    else:
        solution = None

    assert optimal is not None, 'optimal score is not given'
    
    dataset_dict = {
        'node_xy': problems,
    }

    return dataset_dict, optimal

def use_saved_problems_tsp_pkl(filename, total_episodes, device, start=0, solution_file=None):
    with open(filename, 'rb') as f:
        out_1 = pickle.load(f)[start:start+total_episodes]
        problems = torch.tensor(out_1, dtype=torch.float32,device=device)
        # shape: (batch, problem, 2)
        problem_size = problems.size(1)
    if solution_file is not None:
        with open(solution_file, 'rb') as f2:
            out_2 = pickle.load(f2)[start:start+total_episodes]
            out_2 = np.array(out_2, dtype=object)[:, 0].tolist()
            optimal_score_all = torch.tensor(out_2, dtype=torch.float32, device=device)
            optimal = optimal_score_all.mean().item()
    else:
        # A dummy value, since the optimal score is not given, 
        # but we need to return something here to avoid assertion error in the calling function. 
        # Please provide the solution file or set the optimal score to a meaningful value.
        optimal = 1.0 
        
    dataset_dict = {
        'node_xy': problems,
    }
    return dataset_dict, optimal

def use_saved_problems_tsp_txt(filename, total_episodes, device, start=0, solution_file=None):
    nodes_coords = []
    solution = []

    for line in open(filename, "r").readlines()[start:start+total_episodes]:
        line = line.split(" ")
        num_nodes = int(line.index('output') // 2)
        nodes_coords.append(
            [[float(line[idx]), float(line[idx + 1])] for idx in range(0, 2 * num_nodes, 2)]
        )
        tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]]
        solution.append(tour_nodes)

    problems = torch.tensor(nodes_coords,device=device) # shape: (batch, problem, 2)

    solution = torch.tensor(solution,device=device)  # shape: (batch, problem)

    if problems.size(1) == 1000:
        optimal = 23.1182
    elif problems.size(1) == 10000:
        optimal = 71.7778
    else:
        gathering_index = solution.unsqueeze(2).expand(-1, -1, 2)
        # shape: (batch, problem, 2)
        ordered_seq = problems.gather(dim=1, index=gathering_index)
        rolled_seq = ordered_seq.roll(dims=1, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(2).sqrt()
        # shape: (batch, problem)
        travel_distances = segment_lengths.sum(1)
        # shape: (batch,)
        optimal = travel_distances.mean().item()
    
    dataset_dict = {
        'node_xy': problems,
    }

    return dataset_dict, optimal



def use_saved_problems_cvrp_pt(filename, total_episodes,device,start=0, solution_file=None):

    loaded_dict = torch.load(filename, map_location=device)
    dataset_dict = {
        'depot_xy': loaded_dict['depot_xy'][start:start + total_episodes],
        'node_xy': loaded_dict['node_xy'][start:start + total_episodes],
        'node_demand': loaded_dict['node_demand'][start:start + total_episodes],
        'capacity': 50, # only for dataset from INViT
    }
    optimal = loaded_dict["optimal"]
    return dataset_dict, optimal

def use_saved_problems_cvrp_pkl(filename, total_episodes, device,start=0, solution_file=None):
    with open(filename, 'rb') as f:
        out = np.array(pickle.load(f)[start:start + total_episodes], dtype=object)
        raw_data_depot = torch.tensor(out[:, 0].tolist(), dtype=torch.float32,device=device)
        if raw_data_depot.shape == (total_episodes, 2):
            raw_data_depot = raw_data_depot[:, None, :]
        # shape: (batch, 1, 2)
        raw_data_nodes = torch.tensor(out[:, 1].tolist(), dtype=torch.float32,device=device)
        # shape: (batch, problem, 2)
        raw_data_demand = torch.tensor(out[:, 2].tolist(), dtype=torch.float32,device=device)
        # shape: (batch, problem)
        capacity = float(out[0, 3])
        raw_data_demand = (raw_data_demand / capacity).to(device)
        problem_size = raw_data_nodes.shape[1]
        optimal_dict = {
            1000: 41.2,
            2000: 57.2,
            5000: 126.2,
            7000: 172.1,
            10000: 227.2,
        }
        optimal = optimal_dict[problem_size]

        dataset_dict = {
            'depot_xy': raw_data_depot,
            'node_xy': raw_data_nodes,
            'node_demand': raw_data_demand,
            'capacity': capacity,
        }
        return dataset_dict, optimal

def use_saved_problems_cvrp_txt(filename, total_episodes, device, start=0, solution_file=None):
    raw_data_nodes = []
    raw_data_depot = []
    raw_data_demand = []
    raw_cost = []
    capacity = 0

    for line in open(filename, "r").readlines()[start:start + total_episodes]:
        line = line.split(",")

        depot_index = int(line.index('depot'))
        customer_index = int(line.index('customer'))
        capacity_index = int(line.index('capacity'))
        demand_index = int(line.index('demand'))
        cost_index = int(line.index('cost'))

        depot = [[float(line[depot_index + 1]), float(line[depot_index + 2])]]
        customer = [[float(line[idx]), float(line[idx + 1])] for idx in
                    range(customer_index + 1, capacity_index, 2)]
        raw_data_nodes.append(customer)
        raw_data_depot.append(depot)

        if capacity == 0:
            capacity = float(line[capacity_index + 1])

        demand = [int(line[idx]) for idx in range(demand_index + 2, cost_index)] # remove the first demand which is 0 for depot
        raw_data_demand.append(demand)
        raw_cost.append(float(line[cost_index + 1]))

    raw_data_depot = torch.tensor(raw_data_depot, device=device)
    # shape: (batch, 1, 2)
    raw_data_nodes = torch.tensor(raw_data_nodes, device=device)
    # shape: (batch, problem, 2)
    raw_data_demand = torch.tensor(raw_data_demand, device=device) / capacity
    # shape: (batch, problem)
    optimal_score = np.mean(raw_cost)

    dataset_dict = {
        'depot_xy': raw_data_depot,
        'node_xy': raw_data_nodes,
        'node_demand': raw_data_demand,
        'capacity': capacity,
    }
    return dataset_dict, optimal_score
