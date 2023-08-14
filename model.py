# • The item information is disrupted by enriching modal nodes. (ok, trans-modal)
# • The graph structure would make nodes too similar. It mostly focus on the graph output, hard to distinguish the small differnce with or without some nodes.
# • The sequential relations are attenuated. (region-amplify + relation-amplify)
# • Graph convolution is less efficient.

# • Model isolation (ok, multi-modal graph) 
# • Modal missing (ok, cross-attention)

import datetime
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from aggregator import RGATLayer, GCNLayer, SAGELayer, GATLayer, KVAttentionLayer, HeteAttenLayer
from torch.nn import Module, Parameter
import torch.nn.functional as F

from torch.nn.init import xavier_uniform_, xavier_normal_

from utils import trans_to_cpu, trans_to_cuda


# def _(item_modality_list):
#     for ms in item_modality_list:

class VLGraph(Module):
    def __init__(self, config, img_cluster_feature, txt_cluster_feature, item_image_list, item_text_list):
        '''
        img_cluster_feature: [I_N D]
        txt_cluster_feature: [T_N D]
        node_embedding: [V_N+I_N+T_N D]
        '''
        super(VLGraph, self).__init__()
        self.config = config
        self.batch_size = config['batch_size']
        self.num_item = config['num_node'][config['dataset']]
        self.num_image = config['cluster_num'][config['dataset']]
        self.num_text = config['cluster_num'][config['dataset']]
        self.num_node = self.num_item+self.num_image+self.num_text

        self.dim = config['embedding_size']
        self.auxiliary_info = config['auxiliary_info']
        self.dropout_local = config['dropout_local']
        self.dropout_atten = config['dropout_atten']
        self.n_layer = config['n_layer']

        self.aggregator = config['aggregator']
        self.max_relid = config['max_relid'] # 4/10

        # Aggregator
        if self.aggregator == 'rgat':
            self.local_agg = RGATLayer(self.dim, self.max_relid, self.config['alpha'], dropout=self.dropout_atten)
        elif self.aggregator == 'hete_attention': # transformer-based
            self.local_agg = HeteAttenLayer(config, self.dim, self.max_relid, alpha=0.1, dropout=self.dropout_atten)
        elif self.aggregator == 'kv_attention': # transformer-based
            self.local_agg = KVAttentionLayer(self.dim, self.max_relid, alpha=0.1, dropout=self.dropout_atten)
        elif self.aggregator == 'gcn':
            self.local_agg = GCNLayer(input_dim=self.dim, output_dim=self.dim, n_heads=1, activation=F.relu, dropout=self.dropout_local)
        elif self.aggregator == 'graphsage':
            self.local_agg = SAGELayer(input_dim=self.dim, output_dim=self.dim, n_heads=1, activation=F.relu, dropout=self.dropout_local)
        elif self.aggregator == 'gat':
            self.local_agg = GATLayer(input_dim=self.dim, output_dim=self.dim, n_heads=1, activation=F.relu, dropout=self.dropout_local)

        # Item representation & Position representation
        self.embedding = nn.Embedding(self.num_node, self.dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(150, self.dim)
        self.node_type_embedding = nn.Embedding(4, self.dim)
        # Parameters
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.w_pos_type = nn.Parameter(torch.Tensor((len(self.auxiliary_info)+1) * self.dim, self.dim))

        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)

        self.projection = nn.Sequential(nn.Linear(self.dim, self.dim), nn.ReLU(True), nn.Linear(self.dim, 1)) # for gate

        self.fusion_layer = nn.Linear(self.dim*3, self.dim)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config['lr'], weight_decay=config['l2'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config['lr_dc_step'], gamma=config['lr_dc'])

        self.reset_parameters()
        # innitialize embedding
        assert self.num_image == len(img_cluster_feature) and self.num_text == len(txt_cluster_feature)
        self.embedding.weight[self.num_item : self.num_item+self.num_image].data.copy_(torch.from_numpy(self.normalize_array(img_cluster_feature)))
        self.embedding.weight[self.num_item+self.num_image : self.num_item+self.num_image+self.num_text].data.copy_(torch.from_numpy(self.normalize_array(txt_cluster_feature)))

        item_image_dict = torch.tensor(item_image_list, requires_grad=False).cuda() # [I K]
        self.image_indices = item_image_dict.reshape(-1)
        item_text_dict = torch.tensor(item_text_list, requires_grad=False).cuda() # [I K]
        self.text_indices = item_text_dict.reshape(-1)

        self.k = item_image_dict.shape[-1]
        assert self.image_indices.shape[0] == self.num_item * self.k
        assert self.text_indices.shape[0] == self.num_item * self.k

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def normalize_array(self, array):
        # array = image_feature
        min_val = np.min(array)
        max_val = np.max(array)
        stdv = 1.0 / math.sqrt(self.dim)
        normalized_array = (array - min_val) * (stdv - (-stdv)) / (max_val - min_val) + (-stdv)
        return normalized_array

    def forward(self, adj, nodes, node_type_mask, node_pos_matrix, stage='train'):
        h_nodes = self.embedding(nodes)

        if len(self.auxiliary_info)>0:
            auxiliary_embedding = [h_nodes]
            if 'node_type' in self.auxiliary_info:
                # @_@: add node type embedding
                node_type_embedding = self.node_type_embedding(node_type_mask)
                auxiliary_embedding.append(node_type_embedding)
            if 'pos' in self.auxiliary_info:
                # @_@: add pos embedding
                L = node_pos_matrix.shape[-1]
                pos_emb = self.pos_embedding.weight[:L] # [L D]
                pos_embedding = torch.matmul(node_pos_matrix, pos_emb) # [B L D]
                pos_num = node_pos_matrix.sum(dim=-1, keepdim=True)
                pos_embedding = pos_embedding / (pos_num+1e-9) # mean
                pos_embedding = pos_embedding * torch.clamp(node_type_mask, max=1).unsqueeze(-1) # mask out pad
                auxiliary_embedding.append(pos_embedding)
            h_nodes = torch.cat(auxiliary_embedding, -1)
            h_nodes = torch.matmul(h_nodes, self.w_pos_type)

        # aggregation
        for i in range(self.n_layer):
            h_nodes = self.local_agg(h_nodes, adj, node_type_mask, stage)
            h_nodes = F.dropout(h_nodes, self.dropout_local, training=self.training) # output nodes' hidden: [B L D]
            h_nodes = h_nodes * torch.clamp(node_type_mask, max=1).unsqueeze(-1)

        return h_nodes

    def get_sequence_representation(self, seq_hiddens, mask, pooling_method='last'):
        '''
        input: seq_hiddens [B L D]
        out: hiddens [B D]
        '''
        # Attention Pooling
        if pooling_method == 'last_attention': # keep the last k items
            batch_size = seq_hiddens.shape[0]
            len = seq_hiddens.shape[1]
            pos_emb = self.pos_embedding.weight[:len]
            pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
            nh = torch.matmul(torch.cat([pos_emb, seq_hiddens], -1), self.w_1)
            nh = torch.tanh(nh)

            last_n = 3 # TODO: this is a hyper-parameter
            gather_index = torch.sum(mask, dim = -1) - 1
            last_n_hiddens = [] # [B n D]    
            for n in range(last_n):
                _gather_index = gather_index - n
                _gather_index[_gather_index<0] = 0
                _gather_index = _gather_index.view(-1, 1, 1).expand(-1, -1, nh.shape[-1])
                _hiddens = nh.gather(dim=1, index=_gather_index) # [B 1 D]
                last_n_hiddens.append(_hiddens.squeeze(1))
            last_n_hiddens = torch.stack(last_n_hiddens, dim=1) # [B n D]

            mask = mask.float().unsqueeze(-1)
            hs = torch.sum(seq_hiddens * mask, -2) / torch.sum(mask, 1) # sequence's representation [B D]
            hs = hs.unsqueeze(-2).repeat(1, last_n, 1) # [B n D]

            # @_@: add-gate
            # nh = torch.sigmoid(self.glu1(last_n_hiddens) + self.glu2(hs)) # [B n D]
            # beta = torch.matmul(last_n_hiddens, self.w_2) # [B n 1]
            # hiddens = torch.sum(beta * last_n_hiddens, 1)
            # @_@: cat-gate
            input_h = torch.cat((last_n_hiddens.unsqueeze(-2), hs.unsqueeze(-2)), -2) # [B n 2 D]
            energy = self.projection(input_h) # [B N 2 1]
            weights = torch.softmax(energy.squeeze(-1), dim=-1) # [B, N, 2]
            gate_output = (input_h * weights.unsqueeze(-1)).sum(dim=-2) # # (B, n, 2, D) * (B, n, 2, 1) -> (B, n, D)
            hiddens = torch.sum(gate_output, 1)

        if pooling_method == 'attention':
            mask = mask.float().unsqueeze(-1)

            batch_size = seq_hiddens.shape[0]
            len = seq_hiddens.shape[1]
            pos_emb = self.pos_embedding.weight[:len]
            pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

            hs = torch.sum(seq_hiddens * mask, -2) / torch.sum(mask, 1) # sequence's representation [B D]
            hs = hs.unsqueeze(-2).repeat(1, len, 1) # [B L D]

            nh = torch.matmul(torch.cat([pos_emb, seq_hiddens], -1), self.w_1)
            nh = torch.tanh(nh)
            nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
            beta = torch.matmul(nh, self.w_2)
            beta = beta * mask
            hiddens = torch.sum(beta * seq_hiddens, 1)
        elif pooling_method == 'mean':
            mask = mask.float().unsqueeze(-1)
            hiddens = torch.sum(seq_hiddens * mask, -2) / torch.sum(mask, 1) # sequence's representation [B D]
        elif pooling_method == 'last':
            gather_index = torch.sum(mask, dim = -1) - 1
            gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, seq_hiddens.shape[-1])
            hiddens = seq_hiddens.gather(dim=1, index=gather_index)
            hiddens = hiddens.squeeze(1)
        return hiddens


    def compute_full_scores(self, node_hiddens, alias_item_inputs, alias_img_inputs, alias_txt_inputs, item_seq_mask):
        '''
        Given node's hidden [B N D], predicting the scores [N] for all list of items
        alias_item_inputs: [B L]
        alias_img_inputs: [B L K]
        '''
        mask = item_seq_mask.float().unsqueeze(-1) # [B L 1]
        batch_size = node_hiddens.shape[0]
        L = alias_item_inputs.shape[1]
        batch_idx = torch.arange(batch_size).unsqueeze(1)

        # @_@: Items' hidden
        item_seq_hiddens = torch.gather(node_hiddens, 1, torch.unsqueeze(alias_item_inputs, dim=-1).expand(node_hiddens.shape[0], node_hiddens.shape[1], node_hiddens.shape[2]))
        item_seq_hiddens = mask * item_seq_hiddens
        item_hidden = self.get_sequence_representation(item_seq_hiddens, item_seq_mask, pooling_method=self.config['seq_pooling']) # [B L D] => [B D]
        item_emb = self.embedding.weight[1:self.num_item]  # (n_nodes+1) x latent_size

        if self.config['modality_prediction']:
            # @_@: Images' hidden 
            _ = alias_img_inputs.view(alias_img_inputs.shape[0], -1) # [B L*K]
            _ = node_hiddens[batch_idx, _] # [B L*K D]
            img_seq_hiddens = _.view(batch_size, L, self.k, self.dim) # [B L K D]
            img_seq_hiddens = torch.sum(img_seq_hiddens, -2) / self.k # [B L D]
            img_hiddens = torch.sum(img_seq_hiddens * mask, -2) / torch.sum(mask, 1) # [B D]
            
            selected_rows = torch.gather(self.embedding.weight, 0, self.image_indices.unsqueeze(1).repeat(1, self.dim))
            img_emb = selected_rows.reshape(self.num_item, self.k, self.dim)
            img_emb = torch.sum(img_emb, -2) # [I D]

            # @_@: Texts' hidden        
            _ = alias_txt_inputs.view(alias_txt_inputs.shape[0], -1) # [B L*K]
            _ = node_hiddens[batch_idx, _] # [B L*K D]
            txt_seq_hiddens = _.view(batch_size, L, self.k, self.dim) # [B L K D]

            txt_seq_hiddens = torch.sum(txt_seq_hiddens, -2) / self.k # [B L D]
            txt_hiddens = torch.sum(txt_seq_hiddens * mask, -2) / torch.sum(mask, 1) # [B D]

            selected_rows = torch.gather(self.embedding.weight, 0, self.text_indices.unsqueeze(1).repeat(1, self.dim))
            txt_emb = selected_rows.reshape(self.num_item, self.k, self.dim)
            txt_emb = torch.sum(txt_emb, -2) # [I D]

            # @_@: linear layer
            emb = self.fusion_layer(torch.cat([item_emb, img_emb[1:], txt_emb[1:]], -1))
            hiddens = self.fusion_layer(torch.cat([item_hidden, img_hiddens, txt_hiddens], -1))
            scores = torch.matmul(hiddens, emb.transpose(1, 0))
        else:
            scores = torch.matmul(item_hidden, item_emb.transpose(1, 0))

        return scores
    

def model_process(model, data, stage='train'):
    adj, nodes, node_type_mask, node_pos_matrix, inputs_mask, targets, u_input, alias_inputs, alias_img_inputs, alias_txt_inputs = data
    adj = trans_to_cuda(adj).float()
    node_pos_matrix = trans_to_cuda(node_pos_matrix).float()
    nodes, node_type_mask, alias_inputs, alias_img_inputs, alias_txt_inputs = trans_to_cuda(nodes).long(), trans_to_cuda(node_type_mask).long(), trans_to_cuda(alias_inputs).long(), trans_to_cuda(alias_img_inputs).long(), trans_to_cuda(alias_txt_inputs).long()
    inputs_mask, targets, u_input = trans_to_cuda(inputs_mask).long(), trans_to_cuda(targets).long(), trans_to_cuda(u_input).long()

    node_hidden = model.forward(adj, nodes, node_type_mask, node_pos_matrix, stage) # [item's hidden, image's hidden, text's hidden, 0, ...]

    scores = model.compute_full_scores(node_hidden, alias_inputs, alias_img_inputs, alias_txt_inputs, inputs_mask) # the scores for all list of items
    return targets, scores


def train_and_test(model, train_loader, test_loader, topk=[20], logger=None):
    logger.info('start training.') if logger else print('start training.')
    model.train()
    total_loss = 0.0
    for data in tqdm(train_loader):
        model.optimizer.zero_grad()
        targets, scores = model_process(model, data, stage='train')
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
    logger.info('\tLoss:\t%.3f' % total_loss) if logger else print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()

    logger.info('start predicting.') if logger else print('start predicting.')
    model.eval()
    hit, mrr, ndcg = [[] for k in topk], [[] for k in topk], [[] for k in topk]
    for data in test_loader:
        targets, scores = model_process(model, data, stage='test')
        for i, k in enumerate(topk):
            hit_scores, mrr_scores, ndcg_scores = evaluate(targets, scores, k)
            hit[i] = hit[i]+hit_scores.tolist()
            mrr[i] = mrr[i]+mrr_scores.tolist()
            ndcg[i] = ndcg[i]+ndcg_scores.tolist()
    result = [[] for k in topk]
    for i, k in enumerate(topk):
        result[i].append(np.mean(hit[i]) * 100)
        result[i].append(np.mean(mrr[i]) * 100)
        result[i].append(np.mean(ndcg[i]) * 100)

    return result # [[0.1, 0.2, 0.3], [0.2, 0.4, 0.6]]: [topk5, top20]


def evaluate(targets, scores, k):
    sorted_indices = scores.topk(k)[1]
    sorted_indices = trans_to_cpu(sorted_indices).detach()
    targets = trans_to_cpu(targets-1)

    hit_scores = hit_at_k(targets, sorted_indices, k)
    mrr_scores = mrr_at_k(targets, sorted_indices, k)
    ndcg_scores = ndcg_at_k(targets, sorted_indices, k)

    return hit_scores, mrr_scores, ndcg_scores


def ndcg_at_k(targets, sorted_indices, topk):
    k = min(topk, targets.shape[-1])
    ideal_dcg = torch.sum(1.0 / torch.log2(torch.arange(k) + 2))
    dcg = torch.sum(torch.where(sorted_indices[:, :k] == targets.unsqueeze(-1), 1.0 / torch.log2(torch.arange(k) + 2), torch.tensor(0.0)), dim=-1)
    return dcg / ideal_dcg # # [B]

# 计算 Hit
def hit_at_k(targets, sorted_indices, topk):
    k = min(topk, targets.shape[-1])
    hits = torch.sum(sorted_indices[:, :k] == targets.unsqueeze(-1), dim=-1).float()
    return hits # [B]

# 计算 MRR
def mrr_at_k(targets, sorted_indices, topk):
    k = min(topk, targets.shape[-1])
    indices = torch.arange(1, k + 1)
    rranks = torch.where(sorted_indices[:, :k] == targets.unsqueeze(-1), 1.0 / indices, torch.tensor(0.0))
    return torch.max(rranks, dim=-1)[0] # [B]
