import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Model_Lib import *

class TSPUpperModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = self.model_params['embedding_dim']
        self.embedding = nn.Linear(2, self.embedding_dim)

        # first node & last node
        self.Wq_first = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.Wq_last = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        
        # for attention
        self.Wk = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.Wv = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

        self.alpha_attn = nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        self.alpha_com = nn.Parameter(torch.Tensor([1.]), requires_grad=True)

        self.encoded_nodes = None
        self.log_scale = None
        self.k = None
        self.v = None
        self.single_head_key = None
        self.q_first = None
        
        # initialize parameters with Xavier initialization
        # experimentally, we find that Xavier initialization can lead to more stable convergence compared to default initialization.
        self.reset_initial_params()
    
    def reset_initial_params(self):
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        torch.nn.init.xavier_uniform_(self.Wq_first.weight)
        torch.nn.init.xavier_uniform_(self.Wq_last.weight)
        torch.nn.init.xavier_uniform_(self.Wk.weight)
        torch.nn.init.xavier_uniform_(self.Wv.weight)

    def set_decoder_method(self,decoder_type):
        self.model_params['eval_type'] = decoder_type

    def pre_forward(self, reset_state):
        self.encoded_nodes = self.embedding(reset_state.problems)
        # shape: (batch, problem, embedding_dim)
        self.set_kv(self.encoded_nodes)
        self.log_scale = reset_state.log_scale
        # shape: (batch, problem, embedding_dim)

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem, embedding)
        self.k = self.Wk(encoded_nodes)
        self.v = self.Wv(encoded_nodes)
        self.single_head_key = encoded_nodes
        # shape: (batch, problem, embedding)

    def set_q1(self, state):
        # encoded_q.shape: (batch, 1, embedding)
        encoded_first_node = get_encoding(self.encoded_nodes, state.current_node[:, None])
        self.q_first = self.Wq_first(encoded_first_node)
        # shape: (batch, 1, embedding)

    def forward(self, state):
        # unvisited_index_sorted.shape:(batch, unvisited_num)
        batch_size = state.batch_size
        problem_size = state.problem_size
        if state.selected_count == 1:
            self.set_q1(state)
            # shape: (batch, 1, embedding)
        if state.current_node is None:
            return None, None, None
        else:
            unvisited_index_sorted = state.upper_unvisited_index
            # shape: (batch, max_cur_unvisited_num)
            cur_dist_unvisited = state.upper_cur_dist
            # shape: (batch, 1, max_cur_unvisited_num)
            cur_ninf_mask = state.upper_cur_ninf_mask
            # shape: (batch, 1, max_cur_unvisited_num)

            encoded_last_node = get_encoding(self.encoded_nodes, state.current_node[:, None])
            # shape: (batch, 1, embedding)
            q_last = self.Wq_last(encoded_last_node)
            # shape: (batch, 1, embedding_dim)

            #  AAFM
            #######################################################
            q = self.q_first + q_last
            # shape: (batch, 1, embedding_dim)
            k = get_encoding(self.k, unvisited_index_sorted)
            v = get_encoding(self.v, unvisited_index_sorted)
            # shape: (batch, unvisited_num, embedding_dim)
            alpha_adaptation_bias_attn = -1 * self.log_scale * self.alpha_attn * cur_dist_unvisited
            # shape: (batch, 1, unvisited)
            AAFM_OUT = adaptation_attention_free_module(q, k, v, alpha_adaptation_bias_attn,ninf_mask=cur_ninf_mask)
            # shape: (batch, 1, embedding_dim)

            #######################################################
            # obtain unvisited nodes' scores
            alpha_adaptation_bias_com = -1 * self.log_scale * self.alpha_com * cur_dist_unvisited
            # shape: (batch, 1, unvisited)
            single_k = get_encoding(self.single_head_key, unvisited_index_sorted)
            # shape: (batch, unvisited_num, embedding_dim)

            upper_score_k = compatibility(self.model_params, AAFM_OUT, single_k, alpha_adaptation_bias_com, cur_ninf_mask)
            # shape: (batch, unvisited_num)
            
            if self.training:
                assert self.model_params['eval_type'] == 'sampling', "During training, only sampling is allowed"
            selected, selected_score = select_next_node(upper_score_k, decoding_strategy=self.model_params['eval_type'])

            upper_scores = torch.zeros(size=(batch_size, problem_size), device=upper_score_k.device)
            upper_scores.scatter_(dim=1, index=unvisited_index_sorted, src=upper_score_k)
            # shape: (batch, problem_size)
            true_selected = torch.gather(unvisited_index_sorted, dim=-1, index=selected.unsqueeze(-1)).squeeze(-1)
            # shape: (batch,)

            return upper_scores, true_selected, selected_score

class TSPLowerModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = self.model_params['embedding_dim']
        self.device = self.model_params['device']
        self.embedding = nn.Linear(2, self.embedding_dim,device=self.device)

        decoder_layer_num = self.model_params['decoder_layer_num']
        self.layers_unvisited = nn.ModuleList([TSP_Decoder(**self.model_params) for _ in range(decoder_layer_num)])

        # first node & last node
        self.Wq_first = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False,device=self.device)
        self.Wq_last = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False,device=self.device)

        self.padding_embedded = torch.zeros(size=(1, self.embedding_dim), device=self.device)

        self.alpha_com = nn.Parameter(torch.Tensor([1.]), requires_grad=True)  # for compatibility

    def set_decoder_method(self,decoder_type):
        self.model_params['eval_type'] = decoder_type


    def forward(self, state):
        # pairwise_dist.shape: (batch, 1+1+k, 1+1+k)
        # ninf_mask.shape: (batch, 1, 1+1+max_neighbor_k)
        # neighbors_index.shape: (batch, max_neighbor_k)
        batch_size = state.batch_size

        if state.current_node is None:
            true_selected = torch.zeros(size=(batch_size, ), dtype=torch.long)
            prob = torch.ones(size=(batch_size,))
        else:
            neighbors_ninf_mask = state.lower_cur_ninf_mask[..., 2:]
            # shape: (batch, 1, k)

            first_last_neighbor_embedded = self.embedding(state.lower_xy)
            # shape: (batch,1+1+k, embedding)
            q_first = self.Wq_first(first_last_neighbor_embedded[:, [0],:])
            q_last = self.Wq_last(first_last_neighbor_embedded[:, [1],:])
            # shape: (batch, 1, embedding)

            neighbor_nodes = first_last_neighbor_embedded[:, 2:]
            # shape: (batch, k, embedding)
            # use placeholder to represent the padding embeddings of the nodes that do not exist
            neighbor_nodes[neighbors_ninf_mask.squeeze(-2) < 0] = self.padding_embedded

            decoder_out = torch.cat((q_first, q_last, neighbor_nodes), dim=1)
            # shape: (batch, 1+1+k, embedding)
            log_neighbors = torch.log2(state.neighbors_num_list.float())[:, None, None]
            # shape: (batch, 1, 1)
            scale_pairwise_dist = -1 * log_neighbors * state.lower_pairwise_dist
            # shape: (batch, 1+1+k, 1+1+k)

            for layer in self.layers_unvisited:
                decoder_out = layer(decoder_out, scale_pairwise_dist,
                                    state.lower_cur_ninf_mask.expand(-1,scale_pairwise_dist.size(1),-1))
                # shape: (batch, 1+1+k, embedding)

            q_out = decoder_out[:, [0]] + decoder_out[:, [1]]
            # shape: (batch, 1, embedding)
            neighbor_nodes_out = decoder_out[:, 2:]
            # shape: (batch, k, embedding)
            adaptation_bias_com = self.alpha_com * scale_pairwise_dist[:, 1, 2:][:, None, :]
            # shape: (batch, 1, k)
            output_probs = compatibility(self.model_params,q_out, neighbor_nodes_out,
                                         adaptation_bias_com,neighbors_ninf_mask)
            # shape: (batch,k)

            # AAFM may lead to NAN
            if torch.isnan(output_probs).any():
                probs_clone = output_probs.clone()
                flag = torch.isnan(probs_clone)  # shape: (batch, k)
                row_indices = flag.any(dim=1).nonzero(as_tuple=True)[0]  # shape: (batch,)
                probs_clone[flag] = 0.0  # replace nan with a small value
                probs_clone[row_indices, 0] = 1  # if probs are nan, replace the first one with 1 and select it
            else:
                probs_clone = output_probs.clone()

            if self.training:
                assert self.model_params['eval_type'] == 'sampling', "During training, only sampling is allowed"
                
            selected, prob = select_next_node(probs_clone, self.model_params['eval_type'])
            # shape: (batch, )
            # revise the selected node
            true_selected = torch.gather(state.lower_neighbors_index, dim=-1, index=selected.unsqueeze(-1)).squeeze(-1)
            # shape: (batch,)

        return true_selected, prob

########################################
# DECODER
########################################
class TSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        device = self.model_params['device']

        self.Wq = nn.Linear(embedding_dim, embedding_dim, bias=False, device=device)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False, device=device)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False, device=device)

        self.feedForward = Feed_Forward_Module(**model_params)

        self.alpha_attn = nn.Parameter(torch.Tensor([1.]), requires_grad=True)

        self.layer_norm_1 = AddAndLayerNormalization(**model_params)
        self.layer_norm_2 = AddAndLayerNormalization(**model_params)

    def forward(self,data,scale_pairwise_dist,ninf_mask):
        # data.shape: (batch, 1+1+k, embedding)
        # pairwise_dist.shape: (batch, 1+1+k, 1+1+k)
        # log_neighbors.shape: (batch, 1, 1)

        #  We use AAFM to replace the multi-head attention
        #######################################################
        q = self.Wq(data)
        k = self.Wk(data)
        v = self.Wv(data)
        # shape: (batch, 1+1+k, embedding)

        AAFM_OUT = adaptation_attention_free_module(q, k, v, self.alpha_attn * scale_pairwise_dist, ninf_mask=ninf_mask)
        # shape: (batch, 1+1+k, embedding)

        out1 = self.layer_norm_1(data, AAFM_OUT)
        out2 = self.feedForward(out1)
        out3 = self.layer_norm_2(out1, out2)
        return out3








