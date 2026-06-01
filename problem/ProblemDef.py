import pickle

import math
import torch
import numpy as np

def get_random_instances(problem, batch_size, problem_size, device, capacity=50):

    data_generator = {
        'TSP': get_random_problems_tsp,
        'CVRP': get_random_problems_cvrp,
    }

    if problem not in data_generator.keys():
        assert False, f"Unsupported problem type: {problem}. Supported types are: {list(data_generator.keys())}"

    return data_generator[problem](batch_size, problem_size, device, capacity)

def get_random_problems_tsp(batch_size, problem_size, device, capacity=None):
    problems = torch.rand(size=(batch_size, problem_size, 2),device=device)
    # problems.shape: (batch, problem, 2)
    dataset_dict = {
        'node_xy': problems
    }
    return dataset_dict

def get_random_problems_cvrp(batch_size, problem_size, device, capacity=50):

    depot_xy = torch.rand(size=(batch_size, 1, 2), device=device)
    # shape: (batch, 1, 2)

    node_xy = torch.rand(size=(batch_size, problem_size, 2), device=device)
    # shape: (batch, problem, 2)

    demand = torch.randint(1, 10, size=(batch_size, problem_size), device=device)
    # shape: (batch, problem)

    node_demand = demand / float(capacity)
    # shape: (batch, problem)
    
    dataset_dict = {
        'depot_xy': depot_xy,
        'node_xy': node_xy,
        'node_demand': node_demand
    }

    return dataset_dict


def augment_xy_data_by_8_fold(problems):
    # problems.shape: (batch, problem, 2)

    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, problem, 2)

    return aug_problems


