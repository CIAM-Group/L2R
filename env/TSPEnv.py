import math
from dataclasses import dataclass

import torch

from problem.ProblemDef import get_random_problems_tsp


@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem, 2)
    log_scale: float = None

@dataclass
class Step_State:
    batch_size: int
    problem_size: int
    current_node: torch.Tensor = None
    # shape: (batch, )
    selected_count: int = 0
    ninf_mask: torch.Tensor = None
    # shape: (batch, node)

    upper_cur_dist: torch.Tensor = None
    upper_cur_ninf_mask: torch.Tensor = None
    upper_unvisited_index: torch.Tensor = None

    lower_xy: torch.Tensor = None
    lower_neighbors_index: torch.Tensor = None
    lower_pairwise_dist: torch.Tensor = None
    lower_cur_ninf_mask: torch.Tensor = None
    neighbors_num_list: torch.Tensor = None


class TSPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = None

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.problems = None
        # shape: (batch, node, node)
        self.dataset_dict = None

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, )
        self.selected_node_list = None
        # shape: (batch, 0~problem)

        self.dist = None
        self.ninf_mask = None

        self.cur_dist = None
        self.cur_dist_clone = None
        self.first_xy = None
        self.cur_xy = None
        self.cur_sorted_idx = None
        self.first_last_xy = None
        self.unselected_count = None

        self.FLAG__use_saved_problems = False
        self.device = env_params['device']
        self.saved_node_xy = None
        self.saved_index = None

        self.original_xy_lib = None # for lib data

        self.nodes_score_whole = None
        self.cur_unvisited_num = None
        self.unvisited_index_sorted = None

        self.nearest_unvisited_nodes = None
        self.nearest_unvisited_distance = None

        self.reset_state = None
        self.step_state = None
        
    def input_saved_data(self,dataset_dict,device):
        self.FLAG__use_saved_problems = True
        self.saved_node_xy = dataset_dict["node_xy"]
         # shape: (total_num, problem, 2)
        self.saved_index = 0
        self.device = device
        self.problem_size = self.saved_node_xy.size(1)

    def load_problems(self, batch_size,
                            problem_size,
                            lib_data=None,
                            validation_data=None,
                            device=None,
                            capacity=50):
        self.batch_size = batch_size
        self.problem_size = problem_size
        if device is not None:
            self.device = device

        if lib_data is not None:
            self.edge_weight_type = lib_data.get('edge_weight_type', None)
            self.problems = lib_data["normalized_xy"].to(device)  # shape: (1, problem, 2)
            # shape: (1, problem_size, 2)
            self.original_xy_lib = lib_data['original_xy_lib'].to(device)
            # shape: (1, problem_size, 2)
            assert self.original_xy_lib.size(0) == 1, 'The original_xy_lib should be a single instance.'
        elif validation_data is not None:
            self.problems = validation_data["node_xy"].to(self.device)
            # shape: (batch, problem_size, 2)
        else:
            if not self.FLAG__use_saved_problems:
                self.dataset_dict = get_random_problems_tsp(batch_size, self.problem_size, self.device)
                self.problems = self.dataset_dict['node_xy']
                # shape: (batch, problem_size, 2)
            else:
                self.problems = self.saved_node_xy[self.saved_index:self.saved_index + batch_size].to(self.device)
                # shape: (batch, problem_size, 2)
                self.saved_index += batch_size
        
        assert self.problems.shape == (self.batch_size, self.problem_size, 2), \
            f"problems shape is {self.problems.shape}, but expected {(self.batch_size, self.problem_size, 2)}."

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch,)

        self.selected_node_list = torch.zeros((self.batch_size, 0), dtype=torch.long,device=self.device)
        # shape: (batch, 0~problem)

        # CREATE STEP STATE
        self.step_state = Step_State(batch_size=self.batch_size,problem_size=self.problem_size)
        self.ninf_mask = torch.zeros((self.batch_size,self.problem_size),device=self.device)
        # shape: (batch, problem)

        self.reset_state = Reset_State(problems=self.problems,
                                       log_scale=math.log2(self.problem_size))

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected,lib_mode=False):
        # selected.shape: (batch,)

        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, )
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, None]), dim=-1)
        # shape: (batch, 0~problem)

        # UPDATE STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (batch,)
        self.step_state.selected_count = self.selected_count
        self.unselected_count = self.problem_size - self.selected_count
        self.ninf_mask.scatter_(dim=-1, index=self.current_node[:, None], value=float('-inf'))
        # shape: (batch, node)
        self.cur_xy = self.problems.gather(dim=1, index=selected[:,None,None].expand(-1, 1, 2))
        # shape: (batch, 1, 2)
        if self.selected_count == 1:
            self.first_xy = self.cur_xy.clone()
        self.first_last_xy = torch.cat((self.first_xy,self.cur_xy),dim=1)
        # shape: (batch, 2, 2)
        self.cur_dist = torch.cdist(self.cur_xy, self.problems, p=2,compute_mode='donot_use_mm_for_euclid_dist').squeeze(1)
        # shape: (batch, problem)
        assert self.cur_dist.shape == (self.batch_size, self.problem_size), \
            f"cur_dist shape is {self.cur_dist.shape}, but expected {(self.batch_size, self.problem_size)}."

        self.cur_dist_clone = self.cur_dist.clone() # To prevent a situation where there are no selectable nodes
        # mask the farthest node to be -inf, percentage is a hyperparameter.
        reduction_percentage = int(self.env_params['reduction_percentage'] * self.problem_size)
        if reduction_percentage > 0:
            farthest_index = self.cur_dist.argsort(dim=-1,descending=True)[:, :reduction_percentage]
            self.cur_dist.scatter_(dim=-1, index=farthest_index, value=float('inf'))           
        # change the visited node's distance to be inf, so that it will not be selected.
        self.cur_dist[self.ninf_mask < 0] = float('inf')  # including the current node!!!
        self.cur_dist_clone[self.ninf_mask < 0] = float('inf')  # including the current node!!!
        # shape: (batch, problem)

        self.cur_sorted_idx = self.cur_dist.argsort(dim=-1,descending=False) # ascending order
        # shape: (batch, problem)
        nearest_dist, nearest_idx = self.cur_dist_clone.topk(k=1, dim=-1, largest=False)
        # shape: (batch, 1)
        self.nearest_unvisited_distance = nearest_dist.squeeze(1)
        self.nearest_unvisited_nodes = nearest_idx.squeeze(1)
        # shape: (batch, )

        # returning values
        done = (self.selected_count == self.problem_size)

        if done:
            # judge whether solution is valid.
            assert (self.ninf_mask == float('-inf')).all(), \
                'some solutions are infeasible in current batch: ninf_mask should be all -inf when done is True.'
            reward = -self._get_travel_distance(lib_mode)  # note the minus sign!
            assert reward.shape == (self.batch_size, ), f"reward shape is {reward.shape}, but expected {(self.batch_size, )}."
        else:
            reward = None
        return self.step_state, reward, done

    def _get_travel_distance(self,lib_mode):
        gathering_index = self.selected_node_list.unsqueeze(2).expand(self.batch_size, self.problem_size, 2)
        # shape: (batch, problem, 2)
        if not lib_mode:
            ordered_seq = self.problems.gather(dim=1, index=gathering_index)
            # shape: (batch, problem, 2)
            rolled_seq = ordered_seq.roll(dims=1, shifts=-1)
            segment_lengths = ((ordered_seq-rolled_seq)**2).sum(2).sqrt()
            # shape: (batch, problem)
            travel_distances = segment_lengths.sum(1)
            # shape: (batch,)
        else:
            ordered_seq = self.original_xy_lib.gather(dim=1, index=gathering_index)
            # shape: (batch, problem, 2)
            rolled_seq = ordered_seq.roll(dims=1, shifts=-1)
            # shape: (batch, problem)
            segment_lengths_raw = ((ordered_seq-rolled_seq)**2).sum(2).sqrt()
            # shape: (batch, pomo, problem)
            if self.edge_weight_type == 'CEIL_2D':
                segment_lengths = torch.ceil(segment_lengths_raw)
            elif self.edge_weight_type == 'EUC_2D':
                segment_lengths = torch.floor(segment_lengths_raw + 0.5)
            else:
                raise NotImplementedError(f"edge_weight_type {self.edge_weight_type} is not supported in lib data.")
            travel_distances = segment_lengths.sum(1)
            # shape: (batch,)
        
        return travel_distances

    def get_upper_input(self):
        # self.cur_xy.shape: (batch, 1, 2)
        if self.current_node is None:
            return self.step_state
        self.cur_unvisited_num = torch.sum((self.cur_dist < 2).long(), dim=-1)
        # shape: (batch, )
        
        # if all nodes are visited or prepruning, we set the unvisited num to be 1 and select the nearest node.
        cur_all_masked = (self.cur_unvisited_num == 0)
        self.cur_unvisited_num[cur_all_masked] = 1
        nearest_unvisited_node = self.nearest_unvisited_nodes[cur_all_masked]
        self.cur_sorted_idx[:, 0][cur_all_masked] = nearest_unvisited_node
        max_cur_unvisited_num = self.cur_unvisited_num.max()
        self.unvisited_index_sorted = self.cur_sorted_idx[:, :max_cur_unvisited_num]
        # shape: (batch, max_cur_unvisited_num)

        NEIGHBOR_IDX = torch.arange(max_cur_unvisited_num, device=self.device)[None, :].expand(self.batch_size, -1)
        # shape: (batch, max_cur_unvisited_num)
        cur_dist_unvisited = self.cur_dist.gather(dim=-1, index=self.unvisited_index_sorted)
        # shape: (batch, max_cur_unvisited_num)
        nearest_unvisited_distance = self.nearest_unvisited_distance[cur_all_masked]
        cur_dist_unvisited[:, 0][cur_all_masked] = nearest_unvisited_distance

        cur_dist_unvisited[NEIGHBOR_IDX >= self.cur_unvisited_num[:, None]] = 2 # can be any value which is larger than sqrt(2)
        assert (cur_dist_unvisited <= 2).all(), "cur_dist_unvisited is expected <= 2"
        assert (torch.sum((cur_dist_unvisited < 2).long(), dim=-1) == self.cur_unvisited_num).all(), "cur_unvisited_num is not correct."

        cur_ninf_mask_unvisited = self.ninf_mask.gather(dim=-1, index=self.unvisited_index_sorted)
        # shape: (batch, max_cur_unvisited_num)
        cur_ninf_mask_unvisited[NEIGHBOR_IDX >= self.cur_unvisited_num[:, None]] = float('-inf')
        # check correctness
        unvisited_num = torch.sum((cur_ninf_mask_unvisited >= 0).long(), dim=-1)  # shape: (batch,)
        assert (unvisited_num == self.cur_unvisited_num).all(), f"unvisited_num is {unvisited_num}, but expected {self.cur_unvisited_num}."


        self.step_state.upper_cur_dist = cur_dist_unvisited.unsqueeze(1)
        # shape: (batch, 1, max_cur_unvisited_num)
        self.step_state.upper_unvisited_index = self.unvisited_index_sorted
        self.step_state.upper_cur_ninf_mask = cur_ninf_mask_unvisited.unsqueeze(1)
        # shape: (batch, 1, max_cur_unvisited_num)
        return self.step_state

    def update_cur_scores(self, upper_scores):
        # upper_score.shape: (batch, problem_size)
        self.nodes_score_whole = upper_scores + self.ninf_mask.clone()
        self.cur_sorted_idx = self.nodes_score_whole.argsort(dim=-1,descending=True)  # descending order, note that the score >=0
        # shape: (batch, problem)
        
        # if all nodes are visited, we set the unvisited num to be 1 and select the nearest node.
        cur_all_masked = (self.cur_dist == float('inf')).all(dim=-1)  # shape: (batch,)
        nearest_unvisited_node = self.nearest_unvisited_nodes[cur_all_masked]  # shape: (batch,)
        self.cur_sorted_idx[:, 0][cur_all_masked] = nearest_unvisited_node

    def get_lower_transformed_neighbors(self):
        # nodes_score_whole.shape: (batch,problem)
        # current_node.shape: (batch,)
        # self.problem.shape: (batch,problem,2)
        # self.cur_xy.shape: (batch,1,2) , as same as self.first_xy
        # neighbors_num_list.shape: (batch,)
        # self.cur_sorted_idx.shape: (batch, problem)
        if self.current_node is None:
            return self.step_state

        # step1: get the number of unvisited nodes that after reducing operation.
        # If the number of remaining nodes is less than the number of neighbors, take the minimum of the two.
        ######################################################
        lower_neighbors_num = self.env_params['lower_neighbors_num'] * torch.ones((self.batch_size,), device=self.device)
        neighbors_num_list = torch.min(self.cur_unvisited_num, lower_neighbors_num)
        # shape: (batch, )
        self.step_state.neighbors_num_list = neighbors_num_list
        max_neighbors_num = neighbors_num_list.max().int().item()
        
        # step2: get the coordinates of the first node, the current node and the neighbor nodes.
        # note that we use the current node's coordinates to replace the padding node's coordinates.
        ######################################################
        neighbors_index = self.cur_sorted_idx[:, :max_neighbors_num]
        # shape: (batch, max_neighbor_k)
        NEIGHBOR_IDX = torch.arange(max_neighbors_num, device=self.device)[None, :].expand(self.batch_size, max_neighbors_num)
        # shape: (batch, max_neighbors_num)
        current_node_expand = self.current_node[:, None].expand(-1, max_neighbors_num)[NEIGHBOR_IDX >= neighbors_num_list[:, None]]
        # use the current node to replace the padding node, and its ninf_mask is -inf
        neighbors_index[NEIGHBOR_IDX >= neighbors_num_list[:, None]] = current_node_expand
        # shape: (batch, max_neighbor_k)

        self.step_state.lower_neighbors_index = neighbors_index
        neighbors_xy = self.problems.gather(dim=1, index=neighbors_index.unsqueeze(-1).expand(-1, -1, 2))
        # shape: (batch, max_neighbor_k, 2)
        first_last_neighbors_xy = torch.cat((self.first_last_xy, neighbors_xy), dim=1)
        # shape: (batch, 1+1+max_neighbor_k, 2)
        
        # step3: coordinate transformation
        ######################################################
        first_last_neighbors_xy = self.data_transform(first_last_neighbors_xy)
        # shape: (batch, 1+1+max_neighbor_k, 2)
        self.step_state.lower_xy = first_last_neighbors_xy

        # step4: check correctness and get the ninf_mask state of the neighbors
        ######################################################
        cur_ninf_mask = self.ninf_mask.gather(dim=1, index=neighbors_index)
        # check correctness
        unvisited_num = torch.sum((cur_ninf_mask >= 0).long(), dim=-1) # shape: (batch,)
        assert (unvisited_num == neighbors_num_list).all(), f"unvisited_num is {unvisited_num}, but expected {neighbors_num_list}."

        # make the padding distance to be inf
        padding_zero_mask = torch.zeros((self.batch_size, 2), device=self.device)
        # shape: (batch, 2)
        cur_ninf_mask = torch.cat((padding_zero_mask, cur_ninf_mask), dim=-1).unsqueeze(-2)
        # shape: (batch, 1, 1+1+max_neighbor_k)
        self.step_state.lower_cur_ninf_mask = cur_ninf_mask

        # step5: calculate the pairwise distance.
        ######################################################
        pairwise_dist = torch.cdist(first_last_neighbors_xy, first_last_neighbors_xy, p=2,compute_mode='donot_use_mm_for_euclid_dist')
        # shape: (batch, 1+1+max_neighbor_k, 1+1+max_neighbor_k)
        assert pairwise_dist.shape == (self.batch_size, 2+max_neighbors_num, 2+max_neighbors_num), \
            f"pairwise_dist shape is {pairwise_dist.shape}, but expected {(self.batch_size, 2+max_neighbors_num, 2+max_neighbors_num)}."
        self.step_state.lower_pairwise_dist = pairwise_dist

        return self.step_state

    def data_transform(self,first_last_neighbors_xy):
        # first_last_neighbors_xy.shape: (batch, 1+1+k,2)
        # Transform the coordinates of the current node and the neighbor nodes to [0,1].
        ######################################################
        last_neighbors_xy = first_last_neighbors_xy[:, 1:, :]
        # shape: (batch, 1+neighbor_k, 2)
        xy_max = torch.max(last_neighbors_xy, dim=1, keepdim=True).values
        xy_min = torch.min(last_neighbors_xy, dim=1, keepdim=True).values
        # shape: (batch, 1, 2)
        ratio = torch.max((xy_max - xy_min), dim=-1, keepdim=True).values
        ratio[ratio == 0] = 1
        # shape: (batch, 1, 1)
        first_last_neighbors_xy_transformed = torch.clip((first_last_neighbors_xy - xy_min) / ratio.expand(-1, 1, 2), 0, 1)
        # shape: (batch, 1 + 1 + neighbor_k, 2)
        return first_last_neighbors_xy_transformed
