import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.optim import Adam
import torch.nn.functional as F
from sklearn import cluster
from .layers import GraphAttentionLayer
from torch.nn.parameter import Parameter
from torch.optim import Adam
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.cluster import KMeans
import random


class LHGNets(nn.Module):
    def __init__(self, view_num, sample_size, view_dim, cls_num, dropout_rate, nheads, gpu, latent_dim=128):
        super(LHGNets, self).__init__()
        self.view_num = view_num
        self.sample_size = sample_size
        self.view_dim = view_dim
        self.latent_dim = latent_dim
        self.cls_num = cls_num
        self.dropout_rate = dropout_rate
        self.nheads = nheads
        self.gpu = gpu
        self.alpha = 1.0
        # latent representation
        self.latent = nn.Parameter(torch.FloatTensor(sample_size, latent_dim))
        nn.init.xavier_uniform_(self.latent.data, gain=1.414)

        # reconstruction encoder
        self.encoder = self.build_encoder()
        self.classifiar = self.build_classifiar()

        # meta path attention
        self.meta_att = self.build_meta_att()

        # semantic agg
        self.semantic_agg = self.build_semantic_agg()

    def normalize(x):
        """Normalize"""
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        return x

    def build_encoder(self):
        encoder_lst = nn.ModuleList()
        for v in range(self.view_num):
            encoder_lst.append(
                nn.Sequential(
                    nn.Linear(self.latent_dim, self.view_dim[v]),
                    # nn.Linear(self.latent_dim, 128),
                    # nn.Linear(128, self.view_dim[v]),
                    nn.Dropout(self.dropout_rate)
                )
            )
        return encoder_lst

    def build_meta_att(self):
        meta_att_lst = nn.ModuleList()
        for v in range(self.view_num):
            meta_att_lst.append(
                MetaAtt(self.latent_dim, self.latent_dim, self.dropout_rate, self.nheads)
            )
        return meta_att_lst

    def build_classifiar(self):
        classifiar = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            # nn.Linear(self.latent_dim * self.view_num * self.nheads, self.cls_num),
            nn.ELU(),
        )
        return classifiar

    def build_semantic_agg(self):
        agg = nn.Sequential(
            nn.Linear(self.latent_dim * self.view_num * self.nheads, self.latent_dim)
        )
        return agg

    def get_view_adj(self, meta_path, nn_graph):
        adj = torch.matmul(meta_path.unsqueeze(1), meta_path.unsqueeze(0))
        adj = torch.mul(adj, nn_graph)

        if self.gpu != '-1':
            adj = adj + torch.eye(adj.shape[0]).cuda()
        else:
            adj = adj + torch.eye(adj.shape[0])

        # return nn_graph
        return adj

    def get_nn_graph(self, x):
        x = x.cpu().detach().numpy()
        # todo
        # nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(x)
        nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(x)
        nn_graph = nbrs.kneighbors_graph(x).toarray()
        if self.gpu != '-1':
            nn_graph = torch.Tensor(nn_graph).cuda()
        else:
            nn_graph = torch.Tensor(nn_graph)
        return nn_graph

    def forward(self, feature_mask):
        rec_vec = []
        meta_att_vec = []
        for v in range(self.view_num):
            # reconstruction
            rec_vec.append(
                self.encoder[v](self.latent)
            )
            nn_graph = self.get_nn_graph(self.latent)

            # meta path attention
            meta_path = feature_mask[:, v]
            meta_adj = self.get_view_adj(meta_path, nn_graph)
            meta_att_output = self.meta_att[v](self.latent, meta_adj)
            meta_att_vec.append(meta_att_output)
            # semantic attention

        semantic = torch.cat(meta_att_vec, dim=1)
        semantic = self.semantic_agg(semantic)

        # classifiar
        output = self.classifiar(semantic)
        output = F.log_softmax(output, dim=1)

        # cluster layer
        p = []
        q = []

        # torch.nn.init.xavier_normal_(self.cluster_layer.data)
        def target_distribution(q_tmp):
            weight = q_tmp ** 2 / np.array(q).sum(0)
            return (weight.t() / weight.sum(1)).t()

        print('Initializing cluster centers with k-means.')

        # kmeans = KMeans(n_clusters=self.cls_num, n_init=20)
        # y_predv = kmeans.fit_predict(semantic.data.cpu().numpy())
        # # print(kmeans.cluster_centers_.shape)
        # # self.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).cuda()
        # torch.nn.init.xavier_normal_(self.cluster_layer.data)
        # # print(self.cluster_layer[v].shape)
        # # print(meta_att_vec[v])
        # # print(self.cluster_layer.double().shape)
        # q_tmp = 1.0 / (1.0 + torch.sum(
        #     torch.pow(semantic.unsqueeze(1) - self.cluster_layer.double(), 2), 2) / self.alpha).cuda()
        # q_tmp = q_tmp.pow((self.alpha + 1.0) / 2.0)
        # q_tmp = (q_tmp.t() / torch.sum(q_tmp, 1)).t()
        #
        # q_tmp = q_tmp.data
        # p_tmp = target_distribution(q_tmp).cuda()
        #
        # self.cluster = []
        for v in range(self.view_num):
            kmeans = KMeans(n_clusters=self.cls_num, n_init=20)
            y_predv = kmeans.fit_predict(meta_att_vec[v].data.cpu().numpy())
            self.cluster_layer = (Parameter(torch.Tensor(meta_att_vec[v].shape).cuda()))
            torch.nn.init.xavier_normal_(self.cluster_layer.data)
            self.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).cuda()

            q_tmp = 1.0 / (1.0 + torch.sum(
                torch.pow(meta_att_vec[v].unsqueeze(1) - self.cluster_layer.double(), 2), 2) / self.alpha).cuda()
            q_tmp = q_tmp.pow((self.alpha + 1.0) / 2.0)
            q_tmp = (q_tmp.t() / torch.sum(q_tmp, 1)).t()

            q_tmp = q_tmp.data
            p_tmp = target_distribution(q_tmp).cuda()

            p.append(p_tmp)
            q.append(q_tmp)
        return rec_vec, output, semantic, self.latent, p, q, meta_att_vec


class MetaAtt(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate, nheads=4):
        super(MetaAtt, self).__init__()
        self.dropout_rate = dropout_rate
        self.attentions = [GraphAttentionLayer(input_size, output_size, dropout_rate=dropout_rate, concat=False) for _
                           in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('meta_path_attention_{}'.format(i), attention)

    def forward(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout_rate, training=self.training)
        return x
