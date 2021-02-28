import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv
import networkx as nx

from sklearn.model_selection import train_test_split
from collections import defaultdict
import pandas as pd


class SAGE(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers):
        super(SAGE, self).__init__()
        assert (num_layers > 0)

        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()

        # TODO : currently only mean aggregation
        self.convs.append(SAGEConv(in_size, hidden_size))
        # inner layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_size, hidden_size))
        # last layer
        self.convs.append(SAGEConv(hidden_size, out_size))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def inference(self, x_all, subgraph_loader, device):
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj  # .to(device)
                x = x_all[n_id]  # .to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all


class NodeClassifier(nn.Module):
    def __init__(self, gnn_model, loss_func):
        super(NodeClassifier, self).__init__()
        self.gnn_model = gnn_model
        self.loss_fcn = loss_func

    def initialize(self):
        self.gnn_model.train()

    def evaluate(self):
        self.gnn_model.eval()

    def reset_parameters(self):
        self.gnn_model.reset_parameters()

    def forward(self, x, adj):
        return self.gnn_model(x, adj)

    def loss(self, scores, labels):
        return self.loss_fcn(scores, labels)


def gs_nc_flag(model_forward, perturb_shape, data, n_id, y, optimizer, device, cr_include=False):
    STEP_SIZE = 1e-3
    M = 1

    optimizer.zero_grad()

    total_perturb = torch.FloatTensor(*data['x'].shape).uniform_(-STEP_SIZE, STEP_SIZE).to(device)
    total_perturb.requires_grad_()

    model, forward = model_forward
    out = forward(total_perturb[n_id])

    num_classes = out.size()[1]
    loss = model.loss(out, y)
    loss /= M

    for _ in range(M - 1):
        loss.backward()

        total_perturb_data = total_perturb.detach() + STEP_SIZE * torch.sign(total_perturb.grad.detach())
        total_perturb.data = total_perturb_data.data
        #         print(total_perturb.data)
        total_perturb.grad[:] = 0

        out = forward(total_perturb[n_id])
        nc_loss = model.loss(out, y)

        loss = nc_loss
        loss /= M

    loss.backward()
    optimizer.step()

    return loss


def train_flag(model, data, train_loader, optimizer, unlabeled_adj, cr_include=False):
    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        optimizer.zero_grad()
        assert (max(n_id) <= df_tr.shape[0])

        train_emb = data['x'][n_id]
        forward = lambda perturb: model(train_emb + perturb, adjs)
        if cr_include:
            cr_forward = lambda perturb: model(data['x'] + perturb, unlabeled_adj)
            model_forward = (model, forward, cr_forward)
        else:
            model_forward = (model, forward)

        total_loss += gs_nc_flag(model_forward, train_emb.shape, data, n_id, data['y'][n_id][:batch_size], optimizer, device, cr_include=cr_include)

    loss = total_loss / len(train_loader)

    return loss


def train(model, train_loader, optimizer):
    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        optimizer.zero_grad()

        adjs = [adj.to(device) for adj in adjs]
        out = model(data['x'][n_id], adjs)
        labels = data['y'][n_id][:batch_size]
        if labels.size(0) == 1:
            loss = model.loss(out, labels)
        else:
            loss = model.loss(out, labels.squeeze())
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(labels).sum())

    loss = total_loss / len(train_loader)
    acc = total_correct / (len(train_loader) * BATCH_SIZE)

    return loss, acc


@torch.no_grad()
def evaluate(model, data, subgraph_loader, ep):
    model.evaluate()
    out = model.gnn_model.inference(data['x'], subgraph_loader, device)

    outputs = {}
    # for key in ['train', 'val', 'test']:
    mask = data['train_mask']
    pred = out[mask == 0].max(dim=1)[1]

    # encounter_ids = pd.read_csv('~/workspace/cs260_project/dataset/WiDS2021/UnlabeledWiDS2021.csv')[['encounter_id']].values
    pd.DataFrame.from_dict({'encounter_id': encounter_ids,
                            'diabetes_mellitus': pred.data}).set_index(['encounter_id']).to_csv('./{}_gnn_predictions.csv'.format(str(ep)))
    with open('./{}_gnn_predictions_de.text'.format(str(ep)), 'w') as file:
        for t in out[mask == 0]:
            file.write(str(t.numpy()) + '\n')

    return outputs


if __name__ == '__main__':
    df_tr = pd.read_parquet('~/workspace/cs260_project/dataset/dummy_scale_train.parquet')
    #df_tr = df_tr.iloc[:40]
    df_te = pd.read_parquet('~/workspace/cs260_project/dataset/dummy_scale_test.parquet')

    num_nodes = df_tr.shape[0] + df_te.shape[0]
    train_mask = torch.tensor(np.concatenate((np.ones(df_tr.shape[0]), np.zeros(df_te.shape[0])), axis=0)).type(torch.bool)

    np.random.seed(2021)
    random.seed(2021)
    torch.manual_seed(2021)

    BATCH_SIZE = 128

    num_feats = df_tr.shape[1] - 2  # exclude enounter_id, label(diabetes_mellitus)
    num_feats
    num_classes = 2

    df_t = pd.concat([df_tr, df_te])
    cols = list(df_t.columns)
    cols.remove('encounter_id')
    cols.remove('diabetes_mellitus')
    x = df_t[cols].to_numpy()

    neg_nodes = df_tr.loc[df_tr.diabetes_mellitus == 0].index.values
    pos_nodes = df_tr.loc[df_tr.diabetes_mellitus == 1].index.values

    g = nx.Graph()
    g.add_nodes_from([v for v in range(df_t.shape[0])])
    NEIGHBOR_K = 10
    for i in range(len(pos_nodes) - 1):
        rad_neig = random.choices(pos_nodes, k=NEIGHBOR_K)
        if pos_nodes[i] in rad_neig:
            rad_neig.remove(pos_nodes[i])
        for r in rad_neig:
            g.add_edge(pos_nodes[i], r)
    for i in range(len(neg_nodes) - 1):
        rad_neig = random.choices(pos_nodes, k=NEIGHBOR_K)
        if neg_nodes[i] in rad_neig:
            rad_neig.remove(pos_nodes[i])
        for r in rad_neig:
            g.add_edge(neg_nodes[i], r)
    for i in range(df_t.shape[0]):
        g.add_edge(i, i)

    train_index = torch.tensor(list(g.edges()))
    train_loader = NeighborSampler(train_index.T, node_idx=train_mask, num_nodes=num_nodes,
                                   sizes=[20, 10], batch_size=BATCH_SIZE, shuffle=True)
    # full graph for evaluation
    subgraph_loader = NeighborSampler(train_index.T, node_idx=None, sizes=[-1], batch_size=BATCH_SIZE, shuffle=False)
    data = {'x': torch.tensor(x).type(torch.FloatTensor), 'train_mask': train_mask, 'train_index': train_index,
            'y': torch.tensor(df_t.diabetes_mellitus.values).type(torch.LongTensor)}
    encounter_ids = df_te.encounter_id.values
    del df_tr, df_te

    gnn_model = SAGE(num_feats, 128, num_classes, num_layers=2)
    model = NodeClassifier(gnn_model, nn.CrossEntropyLoss())
    optimizer = torch.optim.Adam(params=model.gnn_model.parameters(), lr=0.003)
    model.reset_parameters()

    device = 'cpu'
    for epoch in range(50):
        # loss = train_flag(model, data, train_loader, optimizer, None)
        loss, acc = train(model, train_loader, optimizer)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')

        if epoch in [9, 19, 29, 39, 49]:
            evals = evaluate(model, data, subgraph_loader, epoch)
            print(evals)
