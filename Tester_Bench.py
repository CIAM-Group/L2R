import random
import time

import numpy as np
import torch

from logging import getLogger

from tqdm import tqdm

import env
import model

from utils.utils import *
from problem.LibReader import TSPLIBReader, CVRPLIBReader, tsplib_cost


class Tester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='tester')
        self.result_folder = get_result_folder()
        
        self.problem = self.env_params['problem'].upper()

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        self.env_params['device'] = device
        self.model_params['device'] = device
        
        # ENV and MODEL
        self.upper_model = getattr(model, f"{self.problem}UpperModel")(**self.model_params)
        self.lower_model = getattr(model, f"{self.problem}LowerModel")(**self.model_params)
        self.env = getattr(env, f"{self.problem}Env")(**self.env_params)

        # Restore
        checkpoint_fullname = tester_params['model_load']['path'] 
        self.logger.info("Load model from: {0}".format(checkpoint_fullname))
        
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.upper_model.load_state_dict(checkpoint['upper_model_state_dict'])
        self.lower_model.load_state_dict(checkpoint['lower_model_state_dict'])
        total_params = list(self.upper_model.parameters()) + list(self.lower_model.parameters())

        total = sum([param.nelement() for param in total_params])
        self.logger.info("Number of parameters: %.2fM" % (total / 1e6))

        # utility
        self.time_estimator = TimeEstimator()
        self.time_estimator_2 = TimeEstimator()
    

    def run(self):
        self.get_sorted_instances(self.tester_params['test_data_load']['filename'])
        self.time_estimator.reset()

        result_dict = {}
        result_dict["instances"] = []
        result_dict['optimal'] = []
        result_dict['problem_size'] = []
        result_dict['score'] = []
        result_dict['gap'] = []
        result_dict['time'] = []
        
        start_time = time.time()
        solved_count = 0

        for name, instance_info in self.data.items():  
            instance_start_time = time.time() 
            optimal = instance_info["optimal"]  # optimal value of the instance
            problem_size = instance_info["problem_size"]  # node number of the instance,not including the depot
            edge_weight_type = instance_info["edge_weight_type"]  # edge weight type of the instance
            capacity = instance_info["capacity"]  # capacity,shape:(1,)
            node_coord = instance_info["locations"].unsqueeze(0) # node coordinates, including the depot
            # shape:(1,problem_size+1,2)
            
            instance_info['original_xy_lib'] = node_coord
            instance_info['normalized_demand'] = instance_info["demand"].unsqueeze(0) / float(capacity) # shape:(problem_size+1,)
            instance_info['optimal'] = optimal
            
            # normalize coordinates to [0,1] 
            ################################################################
            xy_max = torch.max(node_coord, dim=1, keepdim=True).values
            xy_min = torch.min(node_coord, dim=1, keepdim=True).values
            # shape: (1, 1, 2)
            ratio = torch.max((xy_max - xy_min), dim=-1, keepdim=True).values
            ratio[ratio == 0] = 1
            # shape: (1, 1, 1)
            normalized_xy = (node_coord - xy_min) / ratio.expand(-1, 1, 2)
            # shape: (1, problem_size+1,2)
            instance_info["normalized_xy"] = normalized_xy
            
            self.logger.info("=" * 80)
            self.logger.info("Instance name: {0}, problem_size: {1}, edge_weight_type: {2}, optimal: {3}".format(name, problem_size, edge_weight_type, optimal))
            self.logger.info("Instance path: {0}".format(instance_info["file_path"]))
            try:
                score = self._test_one_instance(batch_size=1, instance_info=instance_info)
                solved_count += 1
                gap = (score - optimal) * 100 / optimal
                instance_end_time = time.time()
                during_instance_time = instance_end_time - instance_start_time
                
                self.logger.info("Instance name: {}, optimal score: {:.4f}".format(name, optimal))
                self.logger.info("Score:{:.3f}, Gap:{:.3f}%".format(score, gap))
                self.logger.info("Time: {:.2f}s, {:.2f}m".format(during_instance_time, during_instance_time / 60))
                self.logger.info("Solved {}/{} instances.".format(solved_count, len(self.data)))
            except Exception as e:
                self.logger.info("Error occurred in instance {0}, dimension: {1}, skip it!".format(name, problem_size))
                self.logger.info("Error message: {0}".format(e))
                continue
            
            ############################
            # Logs
            ############################
            result_dict["instances"].append(name)
            result_dict['optimal'].append(optimal)
            result_dict['problem_size'].append(problem_size)
            result_dict['score'].append(score)
            result_dict['gap'].append(gap)
            result_dict['time'].append(during_instance_time)
            
        end_time = time.time()
        assert solved_count > 0, "No instance is solved successfully."

        self.logger.info("=" * 80)
        if self.tester_params["detailed_log"]:
            self.logger.info("instance: {0}".format(result_dict['instances']))
            self.logger.info("optimal: {0}".format(result_dict['optimal']))
            self.logger.info("problem_size: {0}".format(result_dict['problem_size']))
            self.logger.info("score: {0}".format(result_dict['score']))
            self.logger.info("gap: {0}".format(result_dict['gap']))
            self.logger.info("=" * 80)

        self.logger.info("=" * 80)
        self.logger.info("=" * 80)
        assert solved_count == len(result_dict['instances'])
        avg_all_gap = np.mean(result_dict['gap'])
        max_dimension = max(result_dict['problem_size']) 
        min_dimension = min(result_dict['problem_size'])
        
        ranges_list = {
            "tsp": [(1000, 5000), (5001, 100000)],
            "cvrp": [(1000, 7000), (7001, 100000)],
        }
        range_1, range_2 = ranges_list[self.env_params['problem']]
        gap_set_range_1 = [gap for gap, size in zip(result_dict['gap'], result_dict['problem_size']) if range_1[0] <= size <= range_1[1]]
        gap_set_range_2 = [gap for gap, size in zip(result_dict['gap'], result_dict['problem_size']) if range_2[0] <= size <= range_2[1]]
        self.logger.info("size {}~{}, number: {}, avg_gap: {:.3f}%".format(range_1[0], range_1[1], len(gap_set_range_1), np.mean(gap_set_range_1)))
        self.logger.info("size {}~{}, number: {}, avg_gap: {:.3f}%".format(range_2[0], range_2[1], len(gap_set_range_2), np.mean(gap_set_range_2)))
        
        self.logger.info("Solved {0}/{1} instances, with dimension range: [{2}, {3}] ==> avg gap: {4:.3f}%".format(
            solved_count, len(self.data), min_dimension, max_dimension, avg_all_gap))
        self.logger.info("Avg time per instance: {0:.2f}s".format((end_time - start_time) / solved_count))
        
        


    def _test_one_instance(self, batch_size,instance_info):

        # Augmentation
        ###############################################
        problem_size = instance_info['problem_size']
        # Ready
        ###############################################
        self.upper_model.eval()
        self.lower_model.eval()
        self.upper_model.set_decoder_method('greedy')
        self.lower_model.set_decoder_method('greedy')

        
        self.env.load_problems(batch_size, 
                                problem_size,
                                lib_data=instance_info,
                                device=self.device)
        
        # reset peak memory stats to get correct memory usage for each batch
        torch.cuda.reset_peak_memory_stats(device=self.device)
        
        reset_state, _, _ = self.env.reset()
        
        with torch.no_grad():
            self.upper_model.pre_forward(reset_state)
            
            # AM Rollout
            ###############################################
            state, reward, done = self.env.pre_step()
            with tqdm(total=0) as pbar:
                while not done:
                    if state.current_node is not None:
                        state = self.env.get_upper_input()
                        upper_scores,_,_ = self.upper_model(state)
                        self.env.update_cur_scores(upper_scores=upper_scores)
                        
                    state = self.env.get_lower_transformed_neighbors()
                    low_selected, _ = self.lower_model(state)
                    # shape: (batch,)
                    state, reward, done = self.env.step(low_selected,lib_mode=True)
                    # shape: (batch,)
                    pbar.total += 1
                    pbar.update(1)
                    
        batch_memory = torch.cuda.max_memory_allocated(device=self.device) / 1024 / 1024
        self.logger.info("batch_memory: {:.2f}MB, {:.2f}GB, avg_memory:{:.2f}MB".format(
        batch_memory, batch_memory / 1024, batch_memory / batch_size))
        
        # Return
        ###############################################
        avg_score = -reward.float().mean().item()  # negative sign to make positive value
        
        return avg_score
    
    def get_sorted_instances(self, data_dir):
        min_problem_size = self.env_params['problem_size']
        max_problem_size = self.env_params['max_problem_size']
        
        self.logger.info("Reading instances from data_dir: {}, with scale_range: {}-{}".format(data_dir, min_problem_size, max_problem_size))
        self.data = {}
        num_sample = 0
        if self.env_params['problem'] == "tsp":
            for root, _, files in os.walk(data_dir):
                for f in files:
                    file_path = os.path.join(root, f)
                    if f.endswith(".tsp") or f.startswith("E"):
                        name, problem_size, locs, edge_weight_type = TSPLIBReader(file_path)
                        if name is None:
                            continue
                        if not (min_problem_size <= problem_size <= max_problem_size):
                            continue
                        if f.startswith("E") and "DIMACS" in file_path:
                            name = f
                        optimal = tsplib_cost.get(name, None)
                        if optimal is None:
                            raise ValueError(f"Optimal value for TSP instance {name} not found in tsplib_cost dict.")
                        self.data[name] = {
                                        "problem_size": problem_size, 
                                        "locations": torch.as_tensor(locs, dtype=torch.float32),
                                        "edge_weight_type": edge_weight_type,
                                        "demand": torch.zeros(problem_size, dtype=torch.float32), # dummy demand for TSP
                                        "capacity": 1.0, # dummy capacity for TSP, avoid potential division by zero when normalizing demand
                                        "optimal": optimal,
                                        "file_name": f,
                                        "file_path": file_path
                                           }
                        num_sample += 1
        elif self.env_params['problem'] == "cvrp":
            for root, _, files in os.walk(data_dir):
                for f in files:
                    file_path = os.path.join(root, f)
                    if f.endswith(".vrp"):
                        name, problem_size, locs, demand, capacity, optimal, edge_weight_type = CVRPLIBReader(
                            file_path
                        )
                        if name is None:
                            continue
                        if not (min_problem_size <= problem_size <= max_problem_size):
                            continue
                        self.data[name] = {
                                    "problem_size": problem_size,
                                    "locations": torch.as_tensor(locs, dtype=torch.float32),
                                    "edge_weight_type": edge_weight_type,
                                    "demand": torch.as_tensor(demand, dtype=torch.float32),
                                    "capacity": capacity,
                                    "optimal": optimal,
                                    "file_name": f,
                                    "file_path": file_path
                        }
                        num_sample += 1
        else:
            raise ValueError(f"Unsupported problem type: {self.env_params['problem']}")
        
        if num_sample == 0:
                raise ValueError(f"No {self.env_params['problem'].upper()} files found in {data_dir} within the specified scale range {min_problem_size}-{max_problem_size}")
        else:
            self.data = dict(sorted(self.data.items(), key=lambda item: item[1]["problem_size"]))

        self.logger.info("The instances are sorted according to their problem size, and the total number of instances is {}".format(len(self.data)))
        
