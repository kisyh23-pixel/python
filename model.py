# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==== 位置编码 ====
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ==== GAT ====
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features)
        self.a = nn.Linear(2*out_features, 1)
    def forward(self, h, adj):
        Wh = self.W(h)  # (B,N,out_features)
        N = Wh.size(1)
        a_input = Wh.unsqueeze(2).repeat(1,1,N,1)
        a_input = torch.cat([a_input, a_input.permute(0,2,1,3)], dim=-1)
        e = F.leaky_relu(self.a(a_input).squeeze(-1))
        attention = F.softmax(torch.where(adj>0,e,-9e15*torch.ones_like(e)),dim=-1)
        h_prime = torch.matmul(attention, Wh)
        return h_prime

# ==== 主模型 ====
class TrajectoryGNNTransformer(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64, lstm_hidden=64, num_heads=4, ff_dim=128, num_layers=3, future_len=50):
        super().__init__()
        self.lstm = nn.LSTM(in_channels, lstm_hidden, batch_first=True)
        self.gnn = GraphAttentionLayer(lstm_hidden, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, 2)
        self.future_len = future_len
    def forward(self, history, adj, last_pos):
        B,T,N,C = history.shape
        hist_flat = history.view(B*N, T, C)
        _, (h_n, _) = self.lstm(hist_flat)
        h_lstm = h_n[-1].view(B,N,-1)
        x_gnn = self.gnn(h_lstm, adj[:, -1])
        x = self.pos_enc(x_gnn)
        x_trans = self.transformer(x)
        x_main = x_trans[:,0]
        pred = self.fc_out(x_main).unsqueeze(1).repeat(1,self.future_len,1)
        pred = pred.cumsum(dim=1) + last_pos.unsqueeze(1)
        return pred
