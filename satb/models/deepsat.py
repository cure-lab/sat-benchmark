from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
# from utils.dag_utils import subgraph, custom_backward_subgraph

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


class DeepSAT(nn.Module):
    '''
    DeepSAT Graph Neural Networks for Satifiability problems.
    '''
    def __init__(self, args):
        super(DeepSAT, self).__init__()
        
        self.args = args

        # configuration
        self.num_rounds = args.num_rounds
        self.device = args.device
        self.reverse = args.reverse
        self.mask = args.mask

        # dimensions
        self.num_aggr = args.num_aggr
        self.dim_node_feature = args.dim_node_feature
        self.dim_hidden = args.dim_hidden
        self.dim_mlp = args.dim_mlp
        self.dim_pred = args.dim_pred
        self.num_fc = args.num_fc
        self.wx_update = args.wx_update
        self.wx_mlp = args.wx_mlp
        self.dim_edge_feature = args.dim_edge_feature

        # 1. message/aggr-related
        dim_aggr = self.dim_hidden
        if self.args.aggr_function in _aggr_function_factory.keys():
            self.aggr_forward = _aggr_function_factory[self.args.aggr_function](dim_aggr, self.dim_hidden)
            if self.reverse:
                self.aggr_backward = _aggr_function_factory[self.args.aggr_function](dim_aggr, self.dim_hidden, reverse=True)
        else:
            raise KeyError('no support {} aggr function.'.format(self.args.aggr_function))


        # 2. update-related
        if self.args.update_function in _update_function_factory.keys():
            # Here only consider the inputs as the concatenated vector from embedding and feature vector.
            if self.wx_update:
                self.update_forward = _update_function_factory[self.args.update_function](self.dim_node_feature+self.dim_hidden, self.dim_hidden)
                if self.reverse:
                    self.update_backward = _update_function_factory[self.args.update_function](self.dim_node_feature+self.dim_hidden, self.dim_hidden)
            else:
                self.update_forward = _update_function_factory[self.args.update_function](self.dim_hidden, self.dim_hidden)
                if self.reverse:
                    self.update_backward = _update_function_factory[self.args.update_function](self.dim_hidden, self.dim_hidden)
        else:
            raise KeyError('no support {} update function.'.format(self.args.update_function))
        # consider the embedding for the LSTM/GRU model initialized by non-zeros
        self.one = torch.ones(1).to(self.device)
        self.emd_int = nn.Linear(1, self.dim_hidden)
        self.one.requires_grad = False


        # 3. predictor-related
        # TODO: support multiple predictors. Use a nn.ModuleList to handle it.
        self.norm_layer = args.norm_layer
        self.activation_layer = args.activation_layer
        if self.wx_mlp:
            self.predictor = MLP(self.dim_hidden+self.dim_node_feature, self.dim_mlp, self.dim_pred, 
            num_layer=self.num_fc, norm_layer=self.norm_layer, act_layer=self.activation_layer, sigmoid=False, tanh=False)
        else:
            self.predictor = MLP(self.dim_hidden, self.dim_mlp, self.dim_pred, 
            num_layer=self.num_fc, norm_layer=self.norm_layer, act_layer=self.activation_layer, sigmoid=False, tanh=False)

    def forward_features(self, G):
        num_nodes = G.num_nodes
        num_layers_f = max(G.forward_level).item() + 1
        num_layers_b = max(G.backward_level).item() + 1
        one = self.one
        h_init = self.emd_int(one).view(1, 1, -1) # (1 x 1 x dim_hidden)
        h_init = h_init.repeat(1, num_nodes, 1) # (1 x num_nodes x dim_hidden)
        # h_init = torch.empty(1, num_nodes, self.dim_hidden).to(self.device)
        # nn.init.normal_(h_init)

        if self.mask:
            h_true = torch.ones_like(h_init).to(self.device)
            h_false = -torch.ones_like(h_init).to(self.device)
            h_true.requires_grad = False
            h_false.requires_grad = False
            h_init = self.imply_mask(G, h_init, h_true, h_false)
        else:
            h_true = None
            h_false = None

        if 'lstm' in self.args.update_function:
            node_embedding = self._lstm_forward(G, h_init, num_layers_f, num_layers_b, num_nodes, h_true, h_false)
        elif 'gru' in self.args.update_function:
            node_embedding = self._gru_forward(G, h_init, num_layers_f, num_layers_b, h_true, h_false)
        else:
            raise NotImplementedError('The update function should be specified as one of lstm and gru.')
        
        return node_embedding

    def forward_head(self, G, node_embedding):

        if self.wx_mlp:
            pred = self.predictor(torch.cat([node_embedding, x], dim=1))
        else:
            pred = self.predictor(node_embedding)

        return pred

    def forward(self, G):
        node_embedding = self.forward_features(G)
        pred = self.forward_head(G, node_embedding)
        return pred
            
    
    def _lstm_forward(self, G, h_init, num_layers_f, num_layers_b, num_nodes, h_true=None, h_false=None):
        x, edge_index = G.x, G.edge_index
        edge_attr = None
        
        node_state = (h_init, torch.zeros(1, num_nodes, self.dim_hidden).to(self.device)) # (h_0, c_0). here we only initialize h_0. TODO: option of not initializing the hidden state of LSTM.
        

        for _ in range(self.num_rounds):
            for l_idx in range(1, num_layers_f):
                # forward layer
                layer_mask = G.forward_level == l_idx
                l_node = G.forward_index[layer_mask]

                l_state = (torch.index_select(node_state[0], dim=1, index=l_node), 
                            torch.index_select(node_state[1], dim=1, index=l_node))

                l_edge_index, l_edge_attr = subgraph(l_node, edge_index, edge_attr, dim=1)
                msg = self.aggr_forward(node_state[0].squeeze(0), l_edge_index, l_edge_attr)
                l_msg = torch.index_select(msg, dim=0, index=l_node)
                l_x = torch.index_select(x, dim=0, index=l_node)
                
                if self.args.wx_update:
                    _, l_state = self.update_forward(torch.cat([l_msg, l_x], dim=1).unsqueeze(0), l_state)
                else:
                    _, l_state = self.update_forward(l_msg.unsqueeze(0), l_state)

                node_state[0][:, l_node, :] = l_state[0]
                node_state[1][:, l_node, :] = l_state[1]

                if self.mask:
                    node_state[0][:] = self.imply_mask(G, node_state[0], h_true, h_false)

            if self.reverse:
                for l_idx in range(1, num_layers_b):
                    # backward layer
                    layer_mask = G.backward_level == l_idx
                    l_node = G.backward_index[layer_mask]
                    
                    l_state = (torch.index_select(node_state[0], dim=1, index=l_node), 
                                torch.index_select(node_state[1], dim=1, index=l_node))
                    
                    l_edge_index, l_edge_attr = subgraph(l_node, edge_index, edge_attr, dim=0)
                    msg = self.aggr_backward(node_state[0].squeeze(0), l_edge_index, l_edge_attr)
                    l_msg = torch.index_select(msg, dim=0, index=l_node)
                    l_x = torch.index_select(x, dim=0, index=l_node)
                    
                    if self.args.wx_update:
                        _, l_state = self.update_backward(torch.cat([l_msg, l_x], dim=1).unsqueeze(0), l_state)
                    else:
                        _, l_state = self.update_backward(l_msg.unsqueeze(0), l_state)
                    
                    node_state[0][:, l_node, :] = l_state[0]
                    node_state[1][:, l_node, :] = l_state[1]

                    if self.mask:
                        node_state[0][:] = self.imply_mask(G, node_state[0], h_true, h_false)
               

        node_embedding = node_state[0].squeeze(0)

        return node_embedding
    
    def _gru_forward(self, G, h_init, num_layers_f, num_layers_b, h_true=None, h_false=None):
        x, edge_index = G.x, G.edge_index
        edge_attr = None

        node_state = h_init # (h_0). here we initialize h_0. TODO: option of not initializing the hidden state of GRU.


        for _ in range(self.num_rounds):
            for l_idx in range(1, num_layers_f):
                # forward layer
                layer_mask = G.forward_level == l_idx
                l_node = G.forward_index[layer_mask]
                
                l_state = torch.index_select(node_state, dim=1, index=l_node)

                l_edge_index, l_edge_attr = subgraph(l_node, edge_index, edge_attr, dim=1)
                msg = self.aggr_forward(node_state.squeeze(0), l_edge_index, l_edge_attr)
                l_msg = torch.index_select(msg, dim=0, index=l_node)
                l_x = torch.index_select(x, dim=0, index=l_node)
                
                if self.args.wx_update:
                    _, l_state = self.update_forward(torch.cat([l_msg, l_x], dim=1).unsqueeze(0), l_state)
                else:
                    _, l_state = self.update_forward(l_msg.unsqueeze(0), l_state)
                node_state[:, l_node, :] = l_state
                
                if self.mask:
                    node_state = self.imply_mask(G, node_state, h_true, h_false)
            
            if self.reverse:
                for l_idx in range(1, num_layers_b):
                    # backward layer
                    layer_mask = G.backward_level == l_idx
                    l_node = G.backward_index[layer_mask]
                    
                    l_state = torch.index_select(node_state, dim=1, index=l_node)

                    l_edge_index, l_edge_attr = subgraph(l_node, edge_index, edge_attr, dim=0)
                    msg = self.aggr_backward(node_state.squeeze(0), l_edge_index, l_edge_attr)
                    l_msg = torch.index_select(msg, dim=0, index=l_node)
                    l_x = torch.index_select(x, dim=0, index=l_node)
                    
                    if self.args.wx_update:
                        _, l_state = self.update_backward(torch.cat([l_msg, l_x], dim=1).unsqueeze(0), l_state)
                    else:
                        _, l_state = self.update_backward(l_msg.unsqueeze(0), l_state)                
                    
                    node_state[:, l_node, :] = l_state

                    if self.mask:
                        node_state = self.imply_mask(G, node_state, h_true, h_false)


        node_embedding = node_state.squeeze(0)

        return node_embedding

    
    def imply_mask(self, G, h, h_true, h_false):
        # logic implication using masking
        true_mask = (G.mask == 1.0).unsqueeze(0)
        false_mask = (G.mask == 0.0).unsqueeze(0)
        normal_mask = (G.mask == -1.0).unsqueeze(0)
        h_mask = h * normal_mask + h_true * true_mask + h_false * false_mask
        return h_mask

    def decode_assignment(self, g):
        # create dict_state
        out = {}

        # get the solution (assigned during data generation)
        layer_mask = g.forward_level == 0
        l_node = g.forward_index[layer_mask]
        out['sol'] = g.y[l_node]

        # set PO as 1.
        layer_mask = g.backward_level == 0
        l_node = g.backward_index[layer_mask]
        g.mask[l_node] = torch.tensor(1.0)

        # check # PIs
        # literal index
        layer_mask = g.forward_level == 0
        l_node = g.forward_index[layer_mask]
        print('# PIs: ', len(l_node))


        # for backtracking
        ORDER = []
        change_ind = -1
        mask_backup = g.mask.clone().detach()


        for i in range(len(l_node)):
            print('==> # ', i+1, 'solving..')
            output = self.forward(g).cpu()

            # mask
            one_mask = torch.zeros(g.y.size(0))
            one_mask = one_mask.scatter(dim=0, index=l_node, src=torch.ones(len(l_node))).unsqueeze(1)
            
            max_val, max_ind = torch.max(output * one_mask, dim=0)
            min_val, min_ind = torch.min(output + (1 - one_mask), dim=0)

            ext_val, ext_ind = (max_val, max_ind) if (max_val > (1 - min_val)) else (min_val, min_ind)
            # ext_val, ext_ind = torch.min(torch.abs(output * one_mask - 0.5), dim=0)
            print('Assign No. ', ext_ind.item(), 'with prob: ', ext_val.item(), 'as value: ', 1.0 if ext_val > 0.5 else 0.0)
            g.mask[ext_ind] = torch.tensor(1.0) if ext_val > 0.5 else torch.tensor(0.0)
            # push the current index to Q
            ORDER.append(ext_ind)
            
            l_node_new = []
            for i in l_node:
                if i != ext_ind:
                    l_node_new.append(i)
            l_node = torch.tensor(l_node_new)
        


        # literal index
        layer_mask = g.forward_level == 0
        l_node = g.forward_index[layer_mask]
        print('Prob: ', output[l_node])
        
        sol = g.mask[l_node]
        print('Solution: ', sol)
        # check the satifiability
        sat = self.pyg_simulation(g, sol)[0]
        out['mask_0'] = sol
        out['pred_0'] = output[l_node]
        if sat:
            torch.save(out, 'out/{}_s.pth'.format(g.name))
            return sol, sat
        
        # index for saving
        ith = 1
        print('=====> Step into the backtracking...')
        # do the backtracking
        while ORDER:
            # renew the mask
            g.mask = mask_backup.clone().detach()
            change_ind = ORDER.pop()
            print('Change the values when solving No. ', change_ind.item(), 'PIs')
            # literal index
            layer_mask = g.forward_level == 0
            l_node = g.forward_index[layer_mask]

            for i in range(len(l_node)):
                # print('==> # ', i+1, 'solving..')
                output = self.forward(g).cpu()
                # mask
                one_mask = torch.zeros(g.y.size(0))
                one_mask = one_mask.scatter(dim=0, index=l_node, src=torch.ones(len(l_node))).unsqueeze(1)
                
                max_val, max_ind = torch.max(output * one_mask, dim=0)
                min_val, min_ind = torch.min(output + (1 - one_mask), dim=0)

                ext_val, ext_ind = (max_val, max_ind) if (max_val > (1 - min_val)) else (min_val, min_ind)
                # ext_val, ext_ind = torch.min(torch.abs(output * one_mask - 0.5), dim=0)
                g.mask[ext_ind] = torch.tensor(1.0) if ext_val > 0.5 else torch.tensor(0.0)
                # push the current index to Q
                if ext_ind == change_ind:
                    g.mask[ext_ind] = 1 - g.mask[ext_ind]
                print('Assign No. ', ext_ind.item(), 'with prob: ', ext_val.item(), 'as value: ', g.mask[ext_ind].item())
                
                l_node_new = []
                for i in l_node:
                    if i != ext_ind:
                        l_node_new.append(i)
                l_node = torch.tensor(l_node_new)

            # literal index
            layer_mask = g.forward_level == 0
            l_node = g.forward_index[layer_mask]
            print('Prob: ', output[l_node])
            
            sol = g.mask[l_node]
            # check the satifiability
            sat = self.pyg_simulation(g, sol)[0]
            print('Solution: ', sol)
            out['mask_{}'.format(ith)] = sol
            out['pred_{}'.format(ith)] = output[l_node]
            ith += 1
            if sat:
                print('====> Hit the correct solution during the backtracking...')
                torch.save(out, 'out/{}_s.pth'.format(g.name))
                return sol, sat
            else:
                print('Wrong..')
        
        torch.save(out, 'out/{}_f.pth'.format(g.name))

        return None, 0

    def pyg_simulation(self, g, pattern=[]):
        # PI, Level list
        max_level = 0
        PI_indexes = []
        fanin_list = []
        for idx, ele in enumerate(g.forward_level):
            level = int(ele)
            fanin_list.append([])
            if level > max_level:
                max_level = level
            if level == 0:
                PI_indexes.append(idx)
        level_list = []
        for level in range(max_level + 1):
            level_list.append([])
        for idx, ele in enumerate(g.forward_level):
            level_list[int(ele)].append(idx)
        # Fanin list 
        for k in range(len(g.edge_index[0])):
            src = g.edge_index[0][k]
            dst = g.edge_index[1][k]
            fanin_list[dst].append(src)
        
        ######################
        # Simulation
        ######################
        y = [0] * len(g.x)
        # if len(pattern) == 0:
        #     pattern = random_pattern_generator(len(PI_indexes))
        j = 0
        for i in PI_indexes:
            y[i] = pattern[j]
            j = j + 1
        for level in range(1, len(level_list), 1):
            for node_idx in level_list[level]:
                source_signals = []
                for pre_idx in fanin_list[node_idx]:
                    source_signals.append(y[pre_idx])
                if len(source_signals) > 0:
                    if int(g.x[node_idx][1]) == 1:
                        gate_type = 1
                    elif int(g.x[node_idx][2]) == 1:
                        gate_type = 5
                    else:
                        raise("This is PI")
                    y[node_idx] = self.logic(gate_type, source_signals)

        # Output
        if len(level_list[-1]) > 1:
            raise('Too many POs')
        return y[level_list[-1][0]], pattern

    def logic(self, gate_type, signals):
        if gate_type == 1:  # AND
            for s in signals:
                if s == 0:
                    return 0
            return 1

        elif gate_type == 2:  # NAND
            for s in signals:
                if s == 0:
                    return 1
            return 0

        elif gate_type == 3:  # OR
            for s in signals:
                if s == 1:
                    return 1
            return 0

        elif gate_type == 4:  # NOR
            for s in signals:
                if s == 1:
                    return 0
            return 1

        elif gate_type == 5:  # NOT
            for s in signals:
                if s == 1:
                    return 0
                else:
                    return 1

        elif gate_type == 6:  # XOR
            z_count = 0
            o_count = 0
            for s in signals:
                if s == 0:
                    z_count = z_count + 1
                elif s == 1:
                    o_count = o_count + 1
            if z_count == len(signals) or o_count == len(signals):
                return 0
            return 1