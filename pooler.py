import torch
import torch.nn as nn
import torch.nn.functional as F
from aggregator import SAGELayer


class BatchedDiffPool(nn.Module):
    def __init__(self, nfeat, nhid, nnext, link_pred=True):
        '''
        nfeat: input dimenstion of x
        nhid: hidden dimenstion for calculation
        nnext: the number of node in the next layer
        > USAGE: xnext, anext, s_l = pool_layer(adj, x)
                 loss = pool_layer.link_pred_loss + pool_layer.entropy_loss*(0.1) + loss
        > TODO, Problem: All Nodes tend to be clustered into the same cluster
        '''
        super(BatchedDiffPool, self).__init__()
        self.link_pred = link_pred
        # TODO, the layer can be relaced
        self.embed = SAGELayer(nfeat, nhid, n_heads=1, activation=F.relu, dropout=0.0)  # [D1 D2]
        self.assign_mat = SAGELayer(nfeat, nnext, n_heads=1, activation=F.relu, dropout=0.0)  # [D1 N2]
        
        self.link_pred_loss = 0
        self.entropy_loss = 0

    def forward(self, adj, x, mask=None):
        '''
        adj [B N1 N1]
        x [B N1 D1]
        mask [B N1]
        '''
        z_l = self.embed(adj, x)  # [B N1 D2]
        z_l = z_l*mask.unsqueeze(-1) if mask else z_l
        s_l = F.softmax(self.assign_mat(adj, x), dim=-1)  # [B N1 N2]
        
        xnext = torch.matmul(s_l.transpose(-1, -2), z_l)  # [B N2 D2]
        anext = (s_l.transpose(-1, -2)).matmul(adj).matmul(s_l)  # # [B N2 N2]
        
        if self.link_pred:
            # TODO: Masking padded s_l, check below
            self.link_pred_loss = (adj - s_l.matmul(s_l.transpose(-1, -2))).norm(dim=(1, 2))
            self.entropy_loss = torch.distributions.Categorical(probs=s_l).entropy()
            if mask is not None:
                self.entropy_loss = self.entropy_loss * mask.expand_as(self.entropy_loss)
            self.entropy_loss = self.entropy_loss.sum(-1)
        return xnext, anext, s_l



class SAGPool(torch.nn.Module):
    def __init__(self, in_channels, ratio=0.5):
        super(SAGPool, self).__init__()
        
        self.in_channels = in_channels
        self.ratio = ratio
        
        self.linear = torch.nn.Linear(in_channels, 1, bias=False)
        
    def forward(self, x, edge_index):
        # Compute attention scores
        scores = self.linear(x)
        scores = torch.sigmoid(scores)
        
        # Apply attention scores to edges
        edge_weight = scores[edge_index[0]] * scores[edge_index[1]]
        
        # Apply softmax normalization to edge weights
        edge_weight = F.softmax(edge_weight, dim=1)
        
        # Apply pooling operation to nodes
        x_pool = torch.zeros((x.shape[0], self.in_channels)).to(x.device)
        x_pool = torch.scatter_add(x_pool, 0, edge_index[1].unsqueeze(1).repeat(1, self.in_channels), 
                                   x[edge_index[0]] * edge_weight.unsqueeze(1).repeat(1, self.in_channels))
        
        # Select top-k nodes
        num_nodes = int(x.shape[0] * self.ratio)
        _, indices = torch.topk(scores.view(-1), num_nodes)
        x_pool = x_pool[indices]
        
        # Update edge index for coarser graph
        edge_index, edge_weight = torch_geometric.utils.dense_to_sparse(edge_index, edge_weight)
        edge_index, _ = torch_geometric.utils.simplify_edge_index(edge_index)
        
        return x_pool, edge_index
