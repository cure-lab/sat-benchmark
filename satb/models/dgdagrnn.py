from typing import Optional

import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_geometric.utils import softmax
from .gat_conv import AGNNConv
from .gcn_conv import AggConv
from .deepset_conv import DeepSetConv
from .gated_sum_conv import GatedSumConv
from .mlp import MLP
from .layernorm_lstm import LayerNormLSTM
from .layernorm_gru import LayerNormGRU

from torch.nn import LSTM, GRU



def subgraph(target_idx, edge_index, edge_attr=None, dim=0):
    '''
    function from DAGNN
    '''
    le_idx = []
    for n in target_idx:
        ne_idx = edge_index[dim] == n
        le_idx += [ne_idx.nonzero().squeeze(-1)]
    le_idx = torch.cat(le_idx, dim=-1)
    lp_edge_index = edge_index[:, le_idx]
    if edge_attr is not None:
        lp_edge_attr = edge_attr[le_idx, :]
    else:
        lp_edge_attr = None
    return lp_edge_index, lp_edge_attr


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

class SoftEvaluator(MessagePassing):
    '''
    AND node => Soft Min;
    OR node => Soft max;
    Not node => 1 - z;
    '''
    def __init__(self, temperature=5.0, use_aig=False):
        super(SoftEvaluator, self).__init__(aggr='add', flow='source_to_target')

        self.temperature = temperature
        self.use_aig = use_aig



    def forward(self, x, edge_index, node_attr=None):
        return self.propagate(edge_index, x=x, node_attr=node_attr)

    def message(self, x_j, node_attr_i, index: Tensor, ptr: OptTensor, size_i: Optional[int]):
        # x_j has shape [E, out_channels], where out_channel is jut one-dimentional value in range of (0, 1)
        # softmax
        if self.use_aig:
            softmin_j = softmax(-x_j/self.temperature, index, ptr, size_i)
            and_mask = (node_attr_i[:, 1] == 1.0).unsqueeze(1)
            not_mask = (node_attr_i[:, 2] == 1.0).unsqueeze(1)
            t = and_mask * (softmin_j * x_j) + not_mask * (1 - x_j)
        else:
            softmax_j = softmax(x_j/self.temperature, index, ptr, size_i)
            softmin_j = softmax(-x_j/self.temperature, index, ptr, size_i)

            # mask
            and_mask = (node_attr_i[:, 1] == 1.0).unsqueeze(1)
            or_mask = (node_attr_i[:, 2] == 1.0).unsqueeze(1)
            not_mask = (node_attr_i[:, 3] == 1.0).unsqueeze(1)

            t = and_mask * (softmin_j * x_j) + or_mask * (softmax_j * x_j) + not_mask * (1 - x_j)

        return t

    def update(self, aggr_out):
        return aggr_out

class HardEvaluator(MessagePassing):
    '''
    AND node => Soft Min with low temperature;
    OR node => Soft max with low temperature;
    Not node => 1 - z;
    '''
    def __init__(self, temperature=0.01, use_aig=False):
        super(HardEvaluator, self).__init__(aggr='add', flow='source_to_target')

        self.temperature = temperature
        self.use_aig = use_aig



    def forward(self, x, edge_index, node_attr=None):
        # TODO: Check how to do the hard evaluation. 
        # x = (x > 0.5).float()
        return self.propagate(edge_index, x=x, node_attr=node_attr)

    def message(self, x_j, node_attr_i, index: Tensor, ptr: OptTensor, size_i: Optional[int]):
        # x_j has shape [E, out_channels], where out_channel is jut one-dimentional value in range of (0, 1)
        # softmax
        if self.use_aig:
            softmin_j = softmax(-x_j/self.temperature, index, ptr, size_i)
            and_mask = (node_attr_i[:, 1] == 1.0).unsqueeze(1)
            not_mask = (node_attr_i[:, 2] == 1.0).unsqueeze(1)
            t = and_mask * (softmin_j * x_j) + not_mask * (1 - x_j)
        else:
            softmax_j = softmax(x_j/self.temperature, index, ptr, size_i)
            softmin_j = softmax(-x_j/self.temperature, index, ptr, size_i)

            # mask
            and_mask = (node_attr_i[:, 1] == 1.0).unsqueeze(1)
            or_mask = (node_attr_i[:, 2] == 1.0).unsqueeze(1)
            not_mask = (node_attr_i[:, 3] == 1.0).unsqueeze(1)

            t = and_mask * (softmin_j * x_j) + or_mask * (softmax_j * x_j) + not_mask * (1 - x_j)

        return t

    def update(self, aggr_out):
        return aggr_out





class DGDAGRNN(nn.Module):
    '''
    DGDARRNN.
    The model used in this paper named Deep-Gated DAG Recursive Neural Networks (DG-DARGNN).
    https://openreview.net/forum?id=BJxgz2R9t7
    '''
    def __init__(self, use_aig=False, num_rounds=10, temperature=0.01, eplison=0.4, dim_hidden=100, dim_mlp=30, activation_layer='relu', dim_pred=1, num_fc=2):
        super(DGDAGRNN, self).__init__()

        self.use_aig = use_aig

        # configuration
        self.num_rounds = num_rounds
        self.temperature = torch.tensor(temperature, dtype=torch.float).cuda()
        self.eplison = eplison


        # dimensions
        self.dim_node_feature = 3 if self.use_aig else 4
        # assert self.dim_node_feature == 4, 'The dimension for node feature is 4.'
        self.dim_hidden = dim_hidden
        assert self.dim_hidden == 100, 'The dimension for GRU is 100.'
        self.dim_mlp = dim_mlp
        assert self.dim_mlp == 30, 'The dimension for classifier is 30.'
        self.activation_layer = activation_layer
        assert self.activation_layer == 'relu', 'The default activation function for MLP is ReLU.'
        self.dim_pred = dim_pred
        self.num_fc = num_fc
        assert self.num_fc == 2, 'The number of layers for FC is 2.'
        # assert self.args.aggr_function == 'deepset', 'The aggregation function used in circuit-sat is Deepset.'
        # assert self.args.update_function == 'gru', 'The update function used in circuit-sat is GRU.'

        # 1. message/aggr-related
        # TODO: add the difinition for dim of hidden state for deepset
        aggr_forward_pre = MLP(self.dim_hidden, 50, self.dim_hidden, num_layer=2, p_drop=0.2)
        aggr_forward_post = MLP(self.dim_hidden, 50, self.dim_hidden, num_layer=2, p_drop=0.2)
        self.aggr_forward = _aggr_function_factory['deepset'](self.dim_hidden, mlp=aggr_forward_pre, mlp_post=aggr_forward_post)

        aggr_backward_pre = MLP(self.dim_hidden, 50, self.dim_hidden, num_layer=2, p_drop=0.2)
        aggr_backward_post = MLP(self.dim_hidden, 50, self.dim_hidden, num_layer=2, p_drop=0.2)
        self.aggr_backward = _aggr_function_factory['deepset'](self.dim_hidden, mlp=aggr_backward_pre, mlp_post=aggr_backward_post)

        # 2. update-related
        self.update_forward = _update_function_factory['gru'](self.dim_node_feature, self.dim_hidden)
        self.update_backward = _update_function_factory['gru'](self.dim_node_feature, self.dim_hidden)

        # 3. projector
        self.projector = nn.Linear(self.dim_hidden, self.dim_node_feature)

        # 4. classifer-related
        self.literal_classifier = MLP(self.dim_hidden, self.dim_mlp, self.dim_pred, num_layer=2, act_layer='relu', sigmoid=True)
        
        # 5. evaluator
        self.soft_evaluator = SoftEvaluator(temperature=self.temperature, use_aig=self.use_aig)
        self.hard_evaluator = HardEvaluator(use_aig=self.use_aig)


    def forward_features(self, G, num_rounds=None):
        num_nodes = G.num_nodes
        num_layers_f = max(G.forward_level).item() + 1
        num_layers_b = max(G.backward_level).item() + 1

        node_embedding = self._gru_forward(G, num_layers_f, num_layers_b, num_nodes, num_rounds)
        return node_embedding

    def forward_head(self, node_embedding):
        pred = self.literal_classifier(node_embedding)
        return pred

    # def forward(self, G, num_rounds=None):
    #     node_embedding = self.forward_features(G)
    #     pred = self.forward_head(node_embedding)
    #     return pred
    
    def forward(self, G, num_rounds=None):
        node_embedding = self.forward_features(G, num_rounds)
        pred = self.forward_head(node_embedding)
        sat = self.evaluate(G, pred)

        return sat

    def _gru_forward(self, G, num_layers_f, num_layers_b, num_nodes, num_rounds=None):
        # the solving procedure in circuitsat
        x, edge_index = G.x.clone().detach(), G.edge_index
        node_state = torch.zeros(1, num_nodes, self.dim_hidden).cuda() # (h_0). here we initialize h_0 as all zeros. TODO: option of  initializing the hidden state of GRU.

        _num_rounds = num_rounds if num_rounds else self.num_rounds
        for round_idx in range(_num_rounds):
            if round_idx > 0:
                x = self.projector(node_state.squeeze(0))
            for l_idx in range(1, num_layers_f):
                # forward layer
                layer_mask = G.forward_level == l_idx
                l_node = G.forward_index[layer_mask]

                l_edge_index, _ = subgraph(l_node, edge_index, dim=1)
                msg = self.aggr_forward(node_state.squeeze(0), l_edge_index)
                l_msg = torch.index_select(msg, dim=0, index=l_node)
                l_x = torch.index_select(x, dim=0, index=l_node)
                
                _, l_state = self.update_forward(l_x.unsqueeze(0), l_msg.unsqueeze(0))
                l_state = l_state.to(dtype=node_state.dtype)
                node_state = node_state.scatter(dim=1, index=l_node.unsqueeze(0).unsqueeze(2).expand(-1, -1, self.dim_hidden), src=l_state)
                
            for l_idx in range(1, num_layers_b):
                # backward layer
                layer_mask = G.backward_level == l_idx
                l_node = G.backward_index[layer_mask]

                l_edge_index, _ = subgraph(l_node, edge_index, dim=0)
                msg = self.aggr_backward(node_state.squeeze(0), l_edge_index)
                l_msg = torch.index_select(msg, dim=0, index=l_node)
                l_x = torch.index_select(x, dim=0, index=l_node)
                
                _, l_state = self.update_backward(l_x.unsqueeze(0), l_msg.unsqueeze(0))
                
                l_state = l_state.to(dtype=node_state.dtype)
                node_state = node_state.scatter(dim=1, index=l_node.unsqueeze(0).unsqueeze(2).expand(-1, -1, self.dim_hidden), src=l_state)


        node_embedding = node_state.squeeze(0)
        
        return node_embedding

    
    def evaluate(self, G, pred):
        # if self.soft:
        #     evaluator = self.soft_evaluator
        # else:
        evaluator = self.hard_evaluator
        x, edge_index = G.x, G.edge_index
        num_nodes = G.num_nodes

        # literal index
        layer_mask = G.forward_level == 0
        l_node = G.forward_index[layer_mask]

        literal_mask = torch.zeros(num_nodes).cuda()
        literal_mask = literal_mask.scatter(dim=0, index=l_node, src=torch.ones(len(l_node)).cuda()).unsqueeze(1)
        literal_mask.requires_grad = False

        pred = pred * literal_mask

        num_layers_f = max(G.forward_level).item() + 1
        for l_idx in range(1, num_layers_f):
            # forward layer
            layer_mask = G.forward_level == l_idx
            l_node = G.forward_index[layer_mask]
            
            l_edge_index, _ = subgraph(l_node, edge_index, dim=1)
            msg = evaluator(pred, l_edge_index, x)
            l_msg = torch.index_select(msg, dim=0, index=l_node)
            
            pred[l_node, :] = l_msg
        
        # sink index
        layer_mask = G.backward_level == 0
        sink_node = G.backward_index[layer_mask]

        sat = torch.index_select(pred, dim=0, index=sink_node)
            
        return sat

    def update_temperature(self, epoch):
        print('\nAnneal temperature from: ', self.temperature.item(), 'to: ', 0.01 * torch.tensor(epoch+1, dtype=torch.float).pow(-self.eplison))
        self.temperature = 0.01 * torch.tensor((epoch+1), dtype=torch.float).cuda().pow(-self.eplison)
        self.soft_evaluator = SoftEvaluator(temperature=self.temperature, use_aig=self.use_aig)

    
    def hard_evaluate(self):
        raise NotImplementedError


    def decode_assignment(self, g):
        return self.forward(g)


    
        


