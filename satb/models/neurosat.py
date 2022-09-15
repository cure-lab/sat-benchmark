"""PyTorch NeuroSAT

This started as a reimplementation of https://github.com/ryanzhangfan/NeuroSAT with 
PyTorch Geometric lib.

It required customized data structure for inputs: literals lie at logic level 0; clauses lie at logic level 1.

Copyright 2022, Lee Man
"""

import torch
from torch import nn
from torch_scatter import scatter_sum
from .gat_conv import AGNNConv
from .gcn_conv import AggConv
from .deepset_conv import DeepSetConv
from .gated_sum_conv import GatedSumConv
from .mlp import MLP
from .layernorm_gru import LayerNormGRU
from .layernorm_lstm import LayerNormLSTM

from torch.nn import LSTM, GRU


_aggr_function_factory = {
    'agnnconv': AGNNConv,
    'deepset': DeepSetConv,
    'gated_sum': GatedSumConv,
    'conv_sum': AggConv,
}

_update_function_factory = {
    'lstm': LSTM,
    'gru': GRU,
    'layernorm_lstm': LayerNormLSTM,
    'layernorm_gru': LayerNormGRU,
}


class NeuronSAT(nn.Module):
    '''
    Recurrent Graph Neural Networks of NeuroSAT.
    The structure follows the pytorch version: https://github.com/ryanzhangfan/NeuroSAT
    '''
    def __init__(self, args):
        super(NeuronSAT, self).__init__()
        
        self.args = args

         # configuration
        self.num_rounds = self.args.num_rounds
        assert self.num_rounds == 26, '# rounds in NeuroSAT is 26'
        self.device = self.args.device

        # dimensions
        self.dim_hidden = args.dim_hidden
        assert self.dim_hidden == 128, 'Size of hidden state in NeuroSAT is 128'
        self.dim_mlp = args.dim_mlp
        self.dim_pred = args.dim_pred
        assert self.dim_pred == 1, 'Size of output in NeuroSAT is 1'
        self.num_fc = args.num_fc
        assert self.num_fc == 3, '# layers in NeuroSAT is 3'

        # 1. message/aggr-related
        assert self.args.aggr_function == 'deepset', 'The aggregation function used in NeuroSAT is deepset.'
        L_msg = MLP(self.dim_hidden, self.dim_hidden, self.dim_hidden, num_layer=self.num_fc, p_drop=0.)
        C_msg = MLP(self.dim_hidden, self.dim_hidden, self.dim_hidden, num_layer=self.num_fc, p_drop=0.)

        self.aggr_forward = _aggr_function_factory[self.args.aggr_function](self.dim_hidden, mlp=L_msg)
        self.aggr_backward = _aggr_function_factory[self.args.aggr_function](self.dim_hidden, mlp=C_msg, reverse=True)
        

        # 2. update-related
        assert self.args.update_function == 'lstm', 'The update function used in NeuroSAT is LSTM'
        self.L_update = _update_function_factory[self.args.update_function](self.dim_hidden*2, self.dim_hidden)
        self.C_update = _update_function_factory[self.args.update_function](self.args.dim_hidden, self.dim_hidden)

        
        # consider the embedding for the LSTM/GRU model initialized by non-zeros
        self.one = torch.ones(1).to(self.device)
        self.L_init = nn.Linear(1, self.dim_hidden)
        self.C_init = nn.Linear(1, self.dim_hidden)
        self.one.requires_grad = False


        # 3. predictor-related
        self.L_vote = MLP(self.dim_hidden, self.dim_hidden, self.dim_pred, num_layer=self.num_fc, p_drop=0.)
    
    def forward_features(self, G, num_rounds=None):
        num_nodes = G.num_nodes

        node_state = self._lstm_forward(G, num_nodes, num_rounds)
        return node_state

    def forward_head(self, G, node_state):
        num_nodes = G.num_nodes
        x, edge_index = G.x, G.edge_index
        l_mask = (x == 0)
        l_index = torch.arange(0, G.x.size(0)).to(self.device)[l_mask.squeeze(1)]

        logits = torch.index_select(node_state[0].squeeze(0), dim=0, index=l_index)  
        vote = torch.zeros((num_nodes, 1)).to(self.device)
        vote[l_index, :] = self.L_vote(logits)
        vote_mean = scatter_sum(vote, G.batch, dim=0).squeeze(1) / (G.n_vars * 2)

        return vote_mean


    def forward(self, G, num_rounds=None):

        node_state = self.forward_features(G, num_rounds)
        pred = self.forward_head(G, node_state)
        return pred
            
    
    def _lstm_forward(self, G, num_nodes, num_rounds=None):
        x, edge_index = G.x, G.edge_index

        l_mask = (x == 0)
        c_mask = (x == 1)
        l_init = self.L_init(self.one).view(1, 1, -1) # (1 x 1 x dim_hidden)
        l_init = l_init.repeat(1, num_nodes, 1) # (1 x num_nodes x dim_hidden)
        c_init = self.C_init(self.one).view(1, 1, -1) # (1 x 1 x dim_hidden)
        c_init = c_init.repeat(1, num_nodes, 1) # (1 x num_nodes x dim_hidden)
        h_init = l_init * l_mask + c_init * c_mask

        c_index = torch.arange(0, G.x.size(0)).to(self.device)[c_mask.squeeze(1)]
        l_index = torch.arange(0, G.x.size(0)).to(self.device)[l_mask.squeeze(1)]

        node_state = (h_init, torch.zeros(1, num_nodes, self.dim_hidden).to(self.device)) # (h_0, c_0). here we only initialize h_0.
        
        _num_rounds = num_rounds if num_rounds else self.num_rounds   
       
        for _ in range(_num_rounds):
            # forward layer
            c_state = (torch.index_select(node_state[0], dim=1, index=c_index), 
                        torch.index_select(node_state[1], dim=1, index=c_index))
                            
            msg = self.aggr_forward(node_state[0].squeeze(0), edge_index)
            c_msg = torch.index_select(msg, dim=0, index=c_index)
            
            _, c_state = self.C_update(c_msg.unsqueeze(0), c_state)

            node_state[0][:, c_index, :] = c_state[0]
            node_state[1][:, c_index, :] = c_state[1]

            # backward layer
            l_state = (torch.index_select(node_state[0], dim=1, index=l_index), 
                        torch.index_select(node_state[1], dim=1, index=l_index))
            msg = self.aggr_backward(node_state[0].squeeze(0), edge_index)
            l_msg = torch.index_select(msg, dim=0, index=l_index)
            
            l_neg = self.flip(G, node_state[0].squeeze(0))
                
            _, l_state = self.L_update(torch.cat([l_msg, l_neg], dim=1).unsqueeze(0), l_state)
            
            node_state[0][:, l_index, :] = l_state[0]
            node_state[1][:, l_index, :] = l_state[1]

        return node_state
        
    
    def flip(self, G, state):
        offset = 0
        select_index = []
        for idx_g in range(G.num_graphs):
            n_vars = G.n_vars[idx_g]
            n_nodes = G.n_nodes[idx_g]
            select_index.append(torch.arange(offset+n_vars, offset+2*n_vars, dtype=torch.long))
            select_index.append(torch.arange(offset, offset+n_vars, dtype=torch.long))
            offset += n_nodes
        select_index = torch.cat(select_index, dim=0).to(self.device)

        flip_state = torch.index_select(state, dim=0, index=select_index)

        return flip_state


    def decode_assignment(self, G, num_rounds=None):
        # self.forward_features(G, num_rounds)
        raise NotImplementedError

