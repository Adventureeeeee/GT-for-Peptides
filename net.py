import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from Gtnet import GraphTransformer



device = torch.device('cuda')
print(f'use {device}')

class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)
        return x



class Selfassembly(nn.Module):
    def __init__(self, protein_dim=128, gt_layers=10, gt_heads=8, out_dim=1):
        super(Selfassembly, self).__init__()
        self.protein_dim = protein_dim
        self.n_layers = gt_layers
        self.n_heads = gt_heads
        self.crossAttention = SelfAttention(hid_dim=self.protein_dim, n_heads=1, dropout=0.2)
        self.protein_gt = GraphTransformer(device, n_layers=gt_layers, node_dim=44, edge_dim=10,
                                                           hidden_dim=protein_dim,
                                                           out_dim=protein_dim, n_heads=gt_heads,
                                                           in_feat_dropout=0.0, dropout=0.2)


        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.joint_attn_prot = nn.Linear(protein_dim, protein_dim)
        self.bn1 = nn.BatchNorm1d(128)
        self.ln1 = nn.LayerNorm(128)
        self.fc_out = nn.Linear(protein_dim, out_dim)
        self.fc = nn.Linear(128,128)
        self.prelu = nn.PReLU(init=0.2, num_parameters=1)


        self.protein_fc = nn.Sequential(
            nn.Linear(self.protein_dim, self.protein_dim),

        )

        self.classifier = nn.Sequential(
            nn.Linear(protein_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, out_dim)

        )


    def dgl_split(self, bg, feats):
        max_num_nodes = int(bg.batch_num_nodes().max())
        batch = torch.cat([torch.full((1, x.type(torch.int)), y) for x, y in zip(bg.batch_num_nodes(), range(bg.batch_size))],
                       dim=1).reshape(-1).type(torch.long).to(bg.device)
        cum_nodes = torch.cat([batch.new_zeros(1), bg.batch_num_nodes().cumsum(dim=0)])
        idx = torch.arange(bg.num_nodes(), dtype=torch.long, device=bg.device)
        idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)
        size = [bg.batch_size * max_num_nodes] + list(feats.size())[1:]
        out = feats.new_full(size, fill_value=0)
        out[idx] = feats
        out = out.view([bg.batch_size, max_num_nodes] + list(feats.size())[1:])
        return out



    def forward(self, protein_graph): 
        protein_feat = self.protein_gt(protein_graph)
        protein_feat_x = self.dgl_split(protein_graph, protein_feat)
    
        protein_feat_xx = self.crossAttention(protein_feat_x, protein_feat_x, protein_feat_x)
        protein_feat_xxx = self.relu(protein_feat_xx)

        p_embedding_mean = torch.mean(protein_feat_xxx, dim=1)

       
        p_embedding = p_embedding_mean
        

        x = self.classifier(p_embedding)

        return x









