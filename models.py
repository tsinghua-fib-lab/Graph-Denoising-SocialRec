import pdb
import pickle
import numpy as np
import torch
from torch import nn
import dgl
import dgl.nn as dglnn
import torch.nn.functional as F


class interaction_GCN(nn.Module):
    def __init__(self, g, embedding_size, feature_sizes, visited_matrix, mask_matrix, rel_names, gpu_id,
                 n_layers=2, user_profile_dict=None):
        super(interaction_GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.num_users, self.num_items = feature_sizes
        self.embedding_size = embedding_size
        self.transformer_encoder = nn.TransformerEncoderLayer(embedding_size, 1, batch_first=True)
        self.transformer_predict = nn.Sequential(*[nn.Linear(embedding_size, embedding_size), nn.ReLU(),
                                                    nn.Linear(embedding_size, 1), nn.Sigmoid()])

        self.visited_matrix = torch.tensor(visited_matrix).long()
        self.mask_matrix = torch.tensor(mask_matrix).float()
        self.separator = nn.Parameter(torch.randn(1, embedding_size))
        self.seq_stop = nn.Parameter(torch.randn(1, embedding_size))
        nn.init.normal_(self.separator, std=0.1)
        nn.init.normal_(self.seq_stop, std=0.1)
        self.gpu_id = gpu_id

        self.user_embeddings = nn.Parameter(torch.randn(self.num_users, embedding_size))
        nn.init.normal_(self.user_embeddings, std=0.1)
        self.item_embeddings = nn.Parameter(torch.randn(self.num_items, embedding_size))
        nn.init.normal_(self.item_embeddings, std=0.1)

        if user_profile_dict!=None:
            user_feat_matrix = [[] for _ in range(len(user_profile_dict))]
            for k, v in user_profile_dict.items():
                user_feat_matrix[k] = v
            self.user_feat_matrix = torch.Tensor(np.array(user_feat_matrix)).cuda()
            self.W = nn.Sequential(*[nn.Linear(embedding_size+17, embedding_size), nn.Sigmoid()])
        else:
            self.user_feat_matrix = None
            self.W = None

        self.scoring_MLP = nn.Sequential(*[nn.Linear(embedding_size*2, embedding_size), nn.ReLU(),
                                           nn.Linear(embedding_size, 1), nn.Sigmoid()])

        for i in range(n_layers):
            self.layers.append(
                dglnn.HeteroGraphConv({rel: dglnn.GraphConv(embedding_size, embedding_size, weight=True,
                                                            activation=nn.ReLU(), allow_zero_in_degree=True)
                                       for rel in rel_names}, aggregate='mean'))
        self.BCEloss = nn.BCELoss()

        self.visited_matrix = self.visited_matrix.cuda()
        self.mask_matrix = self.mask_matrix.cuda()

    def forward(self, norm_edge_weight):
        if self.W!=None:
            users_emb = self.W(torch.cat([self.user_embeddings,self.user_feat_matrix], dim=1))
        else:
            users_emb = self.user_embeddings

        features = {'user': users_emb, 'item': self.item_embeddings}
        all_emb = [features]
        for i, layer in enumerate(self.layers):
            features = layer(self.g, features, mod_kwargs={'trust': {'edge_weight': norm_edge_weight}})
            all_emb.append(features)
        user_emb = torch.cat([each['user'] for each in all_emb],axis=1)
        item_emb = torch.cat([each['item'] for each in all_emb],axis=1)

        return user_emb, item_emb


    def get_user_agg_embedding(self, input_user_list1, input_user_list2):
        interact_matrix_batch1 = self.visited_matrix[input_user_list1]  # batch_size*seq_length
        mask_matrix_batch1 = self.mask_matrix[input_user_list1]

        interact_matrix_batch2 = self.visited_matrix[input_user_list2]
        mask_matrix_batch2 = self.mask_matrix[input_user_list2]

        separator = torch.unsqueeze(self.separator.repeat(input_user_list1.shape[0], 1), 1)  # batch_size*1*embedding*size
        seq_stop = torch.unsqueeze(self.seq_stop.repeat(input_user_list1.shape[0], 1), 1)

        interact_embedding_batch1 = self.item_embeddings[torch.reshape(interact_matrix_batch1, [-1])]
        interact_embedding_batch1 = torch.reshape(interact_embedding_batch1,
                                                    [interact_matrix_batch1.shape[0], interact_matrix_batch1.shape[1], -1])

        interact_embedding_batch2 = self.item_embeddings[torch.reshape(interact_matrix_batch2, [-1])]
        interact_embedding_batch2 = torch.reshape(interact_embedding_batch2,
                                                    [interact_matrix_batch2.shape[0], interact_matrix_batch2.shape[1],
                                                    -1])  # batch_size*seq_length*embedding_size

        interact_embedding_batch = torch.cat([interact_embedding_batch1, separator,
                                                interact_embedding_batch2, seq_stop], dim=1)

        mask_matrix_batch = torch.cat([mask_matrix_batch1, torch.zeros(input_user_list1.shape[0], 1, device='cuda:%d' % self.gpu_id),
                                        mask_matrix_batch2,torch.zeros(input_user_list1.shape[0], 1, device='cuda:%d' % self.gpu_id)], dim=1)
        # batch_size*(seq_lenhth*2+2)

        encoder_output = self.transformer_encoder(interact_embedding_batch, src_key_padding_mask=mask_matrix_batch)
        encoder_output = encoder_output[:, -1, :]  # #batch_size*embedding_size

        output = torch.reshape(self.transformer_predict(encoder_output), [-1])


        return output



