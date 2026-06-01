from dataclasses import dataclass

import math
import torch

from problem.ProblemDef import get_random_problems_cvrp


@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)
    log_scale: float = None


@dataclass
class Step_State:
    batch_size: torch.Tensor = None
    problem_size: torch.Tensor = None
    current_node: torch.Tensor = None
    # shape: (batch, )
    selected_count: int = None
    load: torch.Tensor = None
    # shape: (batch, )
    ninf_mask: torch.Tensor = None
    # shape: (batch, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, )

    upper_cur_dist: torch.Tensor = None
    upper_cur_ninf_mask: torch.Tensor = None
    upper_unvisited_index: torch.Tensor = None

    lower_xy: torch.Tensor = None
    lower_demand: torch.Tensor = None
    lower_neighbors_index: torch.Tensor = None
    lower_pairwise_dist: torch.Tensor = None
    lower_cur_ninf_mask: torch.Tensor = None
    neighbors_num_list: torch.Tensor = None


class CVRPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = None

        self.FLAG__use_saved_problems = False
        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None
        self.saved_index = None
        self.device = env_params['device']

        self.original_xy_lib = None # for lib data
        self.dataset_dict = None

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        # shape: (batch, problem+1)
        self.round_error_epsilon = 0.00001  # for precision stability

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, pomo)
        self.load = None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = None
        # shape: (batch, pomo, problem+1)
        self.finished = None
        # shape: (batch, pomo)

        self.first_xy = None
        self.first_demand = None
        self.cur_xy = None
        self.cur_dist = None
        self.cur_dist_clone = None
        self.cur_demand = None
        self.cur_sorted_idx = None
        self.valid_index_sorted = None
        self.nodes_score_whole = None
        self.cur_valid_num = None
        self.nearest_valid_nodes = None
        self.nearest_valid_distance = None

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def input_saved_data(self,dataset_dict,device):
        self.FLAG__use_saved_problems = True
        self.saved_depot_xy = dataset_dict["depot_xy"]
        self.saved_node_xy = dataset_dict["node_xy"]
        self.saved_node_demand = dataset_dict["node_demand"]
        self.saved_index = 0
        self.device = device
        self.problem_size = dataset_dict["node_xy"].shape[1]

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
            normalized_xy = lib_data["normalized_xy"].to(device)  # shape: (1, problem+1, 2)
            depot_xy = normalized_xy[:, :1, :]  # shape: (1, 1, 2)
            node_xy = normalized_xy[:, 1:, :]  # shape: (1, problem, 2)
            node_demand = lib_data['normalized_demand'][:, 1:].to(device)  # not including the depot node
            self.original_xy_lib = lib_data['original_xy_lib'].to(device)  # shape: (1, problem+1, 2)
            assert self.original_xy_lib.size(0) == 1, 'The original_xy_lib should be a single instance.'
        elif validation_data is not None:
            depot_xy = validation_data["depot_xy"].to(device) # shape: (batch, 1, 2)
            node_xy = validation_data["node_xy"].to(device) # shape: (batch, problem, 2)
            node_demand = validation_data["node_demand"].to(device) # shape: (batch, problem)
           
        else:
            if not self.FLAG__use_saved_problems:
                self.dataset_dict = get_random_problems_cvrp(batch_size, self.problem_size, device, capacity)
                depot_xy = self.dataset_dict['depot_xy'] # depot_xy.shape: (batch, 1, 2)
                node_xy = self.dataset_dict['node_xy'] # node_xy.shape: (batch, problem, 2)
                node_demand = self.dataset_dict['node_demand'] # node_demand.shape: (batch, problem)  
            else:
                depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index+batch_size]
                node_xy = self.saved_node_xy[self.saved_index:self.saved_index+batch_size]
                node_demand = self.saved_node_demand[self.saved_index:self.saved_index+batch_size]
                self.saved_index += batch_size

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1), device=device)
        # shape: (batch, 1)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, problem+1)
        assert self.depot_node_xy.shape == (self.batch_size, self.problem_size+1, 2), "depot_node_xy size error"
        assert self.depot_node_demand.shape == (self.batch_size, self.problem_size+1), "depot_node_demand size error"

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand
        self.reset_state.log_scale = math.log2(self.problem_size)

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, )

        self.selected_node_list = torch.zeros((self.batch_size, 0), dtype=torch.long, device=self.device)
        # shape: (batch, 0~)

        self.at_the_depot = torch.ones(size=(self.batch_size, ), dtype=torch.bool, device=self.device)
        # shape: (batch, )
        self.load = torch.ones(size=(self.batch_size, ), device=self.device)
        # shape: (batch, )
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.problem_size+1), device=self.device)
        # shape: (batch, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.problem_size+1), device=self.device)
        # shape: (batch, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, ), dtype=torch.bool, device=self.device)
        # shape: (batch, )

        self.first_xy = self.reset_state.depot_xy
        # shape: (batch, 1, 2)
        self.first_demand = torch.zeros(size=(self.batch_size, 1), device=self.device)
        # shape: (batch, 1)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.batch_size = self.batch_size
        self.step_state.problem_size = self.problem_size
        self.step_state.current_node = self.current_node
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected,lib_mode=False):
        # selected.shape: (batch, )

        # Dynamic-1
        ####################################
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, )
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, None]), dim=-1)
        # shape: (batch, 0~)

        self.step_state.current_node = self.current_node
        self.step_state.selected_count = self.selected_count

        self.cur_xy = self.depot_node_xy.gather(dim=1, index=selected[:, None, None].expand(-1, 1, 2))
        # shape: (batch, 1, 2)
        self.cur_dist = torch.cdist(self.cur_xy, self.depot_node_xy, p=2,compute_mode='donot_use_mm_for_euclid_dist').squeeze(dim=1)
        # shape: (batch, problem+1)
        assert self.cur_dist.size() == (self.batch_size, self.problem_size+1), "cur_dist size error"
        self.cur_dist_clone = self.cur_dist.clone() # To prevent a situation where there are no selectable nodes

        # mask the farthest node to be -inf, percentage is a hyperparameter.
        reduction_percentage = int(self.env_params['reduction_percentage'] * self.problem_size)
        if reduction_percentage > 0:
            # only mask the customer nodes, not including the depot node.
            farthest_index = self.cur_dist[:,1:].argsort(dim=-1, descending=True)[:, :reduction_percentage] + 1
            self.cur_dist.scatter_(dim=-1, index=farthest_index, value=float('inf'))

        # Dynamic-2
        ####################################
        self.at_the_depot = (selected == 0)
        gathering_index = selected[:, None]
        # shape: (batch, 1)
        selected_demand = self.depot_node_demand.gather(dim=1, index=gathering_index).squeeze(dim=1)
        # shape: (batch, )
        self.load -= selected_demand
        self.cur_demand = selected_demand[:, None] # shape: (batch, 1)

        # check the load
        assert (self.load >= -self.round_error_epsilon).all(), "load cannot be negative"
        self.load[self.at_the_depot] = 1 # refill loaded at the depot

        # check the mask state of the current node
        current_node_state = self.ninf_mask.gather(dim=1, index=gathering_index).squeeze(dim=1)
        # shape: (batch, )
        assert (current_node_state == 0).all(), "the mask state of current nodes should be 0"

        self.visited_ninf_flag.scatter_(dim=1, index=gathering_index, value=float('-inf'))
        # shape: (batch, problem+1)
        self.visited_ninf_flag[:, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        self.ninf_mask = self.visited_ninf_flag.clone()

        demand_too_large = self.load[:, None] + self.round_error_epsilon < self.depot_node_demand
        # shape: (batch, problem+1)
        self.ninf_mask[demand_too_large] = float('-inf')
        # shape: (batch, problem+1)

        # if the instance is finished, we need to set the subsequent nodes to be the depot node.
        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=-1)
        # shape: (batch,)
        self.finished = self.finished + newly_finished
        # shape: (batch,)
        # do not mask depot for finished episode.
        self.ninf_mask[:, 0][self.finished] = 0

        self.cur_dist[self.ninf_mask < 0] = float('inf')  # including the current node!!!
        self.cur_dist_clone[self.ninf_mask < 0] = float('inf')  # note that it must be a valid node.
        self.cur_sorted_idx = self.cur_dist.argsort(dim=-1, descending=False)  # ascending order
        # shape: (batch, problem+1)

        nearest_dist, nearest_idx = self.cur_dist_clone.topk(k=1, dim=-1, largest=False)
        # shape: (batch, 1)
        self.nearest_valid_nodes = nearest_idx.squeeze(dim=1)
        self.nearest_valid_distance = nearest_dist.squeeze(dim=1)
        # shape: (batch, )

        self.step_state.load = self.load
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        # returning values
        done = self.finished.all()
        if done:
            reward = -self._get_travel_distance(lib_mode)  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self,lib_mode):
        gathering_index = self.selected_node_list.unsqueeze(2).expand(-1, -1, 2)
        # shape: (batch, selected_list_length, 2)
        if not lib_mode:
            ordered_seq = self.depot_node_xy.gather(dim=1, index=gathering_index)
            # shape: (batch, problem, 2)
            rolled_seq = ordered_seq.roll(dims=1, shifts=-1)
            segment_lengths = ((ordered_seq-rolled_seq)**2).sum(2).sqrt()
            # shape: (batch, selected_list_length)
            travel_distances = segment_lengths.sum(1)
            # shape: (batch,)
        else:
            ordered_seq = self.original_xy_lib.gather(dim=1, index=gathering_index)
            # shape: (batch, selected_list_length, 2)
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
        self.cur_valid_num = torch.sum((self.cur_dist < 2).long(), dim=-1)
        # shape: (batch, )
        # if all nodes are masked, we need to set the next node to be the depot node.
        cur_all_masked = (self.cur_valid_num == 0)  # shape: (batch,)
        self.cur_valid_num[cur_all_masked] = 1
        nearest_valid_node = self.nearest_valid_nodes[cur_all_masked]  # shape: (batch,)
        self.cur_sorted_idx[:, 0][cur_all_masked] = nearest_valid_node

        max_cur_valid_nodes = self.cur_valid_num.max()
        self.valid_index_sorted = self.cur_sorted_idx[:, :max_cur_valid_nodes]
        # shape: (batch, max_cur_valid_nodes)

        NEIGHBOR_IDX = torch.arange(max_cur_valid_nodes, device=self.device)[None, :].expand(self.batch_size, -1)
        # shape: (batch, max_cur_valid_nodes)
        cur_dist_valid = self.cur_dist.gather(dim=-1, index=self.valid_index_sorted)
        # shape: (batch, max_cur_valid_nodes)
        nearest_unvisited_distance = self.nearest_valid_distance[cur_all_masked]
        cur_dist_valid[:, 0][cur_all_masked] = nearest_unvisited_distance
        cur_dist_valid[NEIGHBOR_IDX >= self.cur_valid_num[:, None]] = 2  # can be any value which is larger than sqrt(2)
        assert (cur_dist_valid <= 2).all(), "cur_dist_unvisited is expected <= 2"
        assert (torch.sum((cur_dist_valid < 2).long(), dim=-1) == self.cur_valid_num).all(), "cur_unvisited_num is not correct."

        cur_ninf_mask_valid = self.ninf_mask.gather(dim=-1, index=self.valid_index_sorted)
        # shape: (batch, max_cur_unvisited_num)
        cur_ninf_mask_valid[NEIGHBOR_IDX >= self.cur_valid_num[:, None]] = float('-inf')
        # check correctness
        valid_num = torch.sum((cur_ninf_mask_valid >= 0).long(), dim=-1)  # shape: (batch,)
        assert (valid_num == self.cur_valid_num).all(), f"unvisited_num is {valid_num}, but expected {self.cur_valid_num}."

        self.step_state.upper_cur_dist = cur_dist_valid.unsqueeze(1)
        # shape: (batch, 1, max_cur_valid_nodes)
        self.step_state.upper_unvisited_index = self.valid_index_sorted
        # shape: (batch, max_cur_valid_nodes)
        self.step_state.upper_cur_ninf_mask = cur_ninf_mask_valid.unsqueeze(1)
        # shape: (batch, 1, max_cur_valid_nodes)
        return self.step_state

    def update_cur_scores(self, upper_scores):
        # upper_score.shape: (batch, max_cur_valid_nodes)
        # valid_index_sorted.shape: (batch, max_cur_valid_nodes)
        self.nodes_score_whole = upper_scores + self.ninf_mask.clone()
        self.cur_sorted_idx = self.nodes_score_whole.argsort(dim=-1, descending=True)  # descending order, note that the score >=0
        # shape: (batch, problem+1)
        
        # if all nodes are masked, we set the unvisited num to be 1 and select the nearest valid node.
        cur_all_masked = (self.cur_dist == float('inf')).all(dim=-1)  # shape: (batch,)
        nearest_valid_node = self.nearest_valid_nodes[cur_all_masked]  # shape: (batch,)
        self.cur_sorted_idx[:,0][cur_all_masked] = nearest_valid_node

    def get_lower_transformed_neighbors(self):
        # cur_dist.shape: (batch,problem)
        # current_node.shape: (batch,)
        # self.problem.shape: (batch,problem+1,2)
        # self.cur_xy.shape: (batch,1,2)
        # neighbors_num_list.shape: (batch,)
        # self.cur_sorted_idx.shape: (batch, problem+1)
        if self.current_node is None:
            return self.step_state

        # step1: get the number of valid nodes that after reducing operation.
        # If the number of remaining nodes is less than the number of neighbors, take the minimum of the two.
        ######################################################
        lower_neighbors_num = self.env_params['lower_neighbors_num'] * torch.ones((self.batch_size,), device=self.device)

        neighbors_num_list = torch.min(self.cur_valid_num, lower_neighbors_num)
        # shape: (batch, )
        self.step_state.neighbors_num_list = neighbors_num_list
        max_neighbors_num = neighbors_num_list.max().int().item()

        # step2: get the coordinates of the first node, the current node and the neighbor nodes.
        # note that we use the current node's coordinates to replace the padding node's coordinates.
        ######################################################
        neighbors_index = self.cur_sorted_idx[:, :max_neighbors_num]
        # shape: (batch, max_neighbor_k)
        NEIGHBOR_IDX = torch.arange(max_neighbors_num, device=self.device)[None, :].expand(self.batch_size, max_neighbors_num)
        # shape: (batch, max_neighbor_k)
        current_node_expand = self.current_node[:, None].expand(-1, max_neighbors_num)[NEIGHBOR_IDX >= neighbors_num_list[:, None]]
        # use the current node to replace the padding node, and its ninf_mask is -inf.
        # the padding node will be replaced by a padding_embedded, the current node is temporary.
        neighbors_index[NEIGHBOR_IDX >= neighbors_num_list[:, None]] = current_node_expand
        # shape: (batch, max_neighbor_k)

        neighbors_xy = self.depot_node_xy.gather(dim=1, index=neighbors_index.unsqueeze(-1).expand(-1, -1, 2))
        # shape: (batch, max_neighbor_k, 2)
        first_last_neighbors_xy = torch.cat((self.first_xy, self.cur_xy, neighbors_xy), dim=1)
        # shape: (batch, 1+1+max_neighbor_k, 2)
        
        # step3: coordinate transformation
        ######################################################
        first_last_neighbors_xy = self.data_transform(first_last_neighbors_xy)
        # shape: (batch, 1+1+max_neighbor_k, 2)

        # step4: get the demand of the neighbors, and set the demand of the padding neighbors to be 0.
        ########################################################
        neighbors_demand = self.depot_node_demand.gather(dim=1, index=neighbors_index)
        # shape: (batch, max_neighbor_k)
        neighbors_demand[NEIGHBOR_IDX >= neighbors_num_list[:, None]] = 0
        self.step_state.lower_demand = neighbors_demand.unsqueeze(-1)
        # shape: (batch, max_neighbor_k, 1)
        
        
        # step5: check correctness and get the ninf_mask state of the neighbors
        ######################################################
        cur_ninf_mask = self.ninf_mask.gather(dim=1, index=neighbors_index)
        valid_num = torch.sum((cur_ninf_mask >= 0).long(), dim=-1)  # shape: (batch,)
        assert (valid_num > 0).all(), "valid_num is expected > 0"
        # the below assertion is not correct in cvrp
        # because if the instance is finished, the unique valid node is the depot node, and the valid_num is 1, but the neighbors are all depot nodes.
        # assert (valid_num == neighbors_num_list).all(), f"valid_num is {valid_num}, but expected {neighbors_num_list}."
        
        padding_zero_mask = torch.zeros((self.batch_size, 2), device=self.device)
        # shape: (batch, 2)
        cur_ninf_mask = torch.cat((padding_zero_mask, cur_ninf_mask), dim=-1).unsqueeze(1)
        # shape: (batch, 1, 1+1+max_neighbor_k)
        assert cur_ninf_mask.shape == (self.batch_size, 1, 2+max_neighbors_num), \
            f"cur_ninf_mask shape is {cur_ninf_mask.shape}, but expected {(self.batch_size, 1, 2+max_neighbors_num)}."

        # step6: calculate the pairwise distance.
        ######################################################
        pairwise_dist = torch.cdist(first_last_neighbors_xy, first_last_neighbors_xy, p=2,compute_mode='donot_use_mm_for_euclid_dist')
        # shape: (batch, 1+1+max_neighbor_k, 1+1+max_neighbor_k)
        assert pairwise_dist.shape == (self.batch_size, 2+max_neighbors_num, 2+max_neighbors_num), \
            f"pairwise_dist shape is {pairwise_dist.shape}, but expected {(self.batch_size, 2+max_neighbors_num, 2+max_neighbors_num)}."

        self.step_state.lower_neighbors_index = neighbors_index
        self.step_state.lower_xy = first_last_neighbors_xy
        self.step_state.lower_cur_ninf_mask = cur_ninf_mask
        self.step_state.lower_pairwise_dist = pairwise_dist

        return self.step_state

    def data_transform(self,first_last_neighbors_xy):
        # last_neighbors_xy.shape: (batch, 1+k,2)
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


