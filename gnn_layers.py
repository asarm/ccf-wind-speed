"""
GNN Layer implementations: GCN, GAT, GraphSAGE
Each layer supports dual embeddings (self_emb and influence_emb) for separate
self and neighbor contributions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """
    GCN Layer with separate self and neighbor contributions.
    Uses dual embeddings: self_emb for self-contribution, influence_emb for neighbor contribution.
    
    output[i] = W_self(self_emb[i]) + Σ adj[j,i] * W_neighbor(influence_emb[j])
    """
    def __init__(self, in_features, out_features, dropout=0.2):
        super().__init__()
        self.linear_self = nn.Linear(in_features, out_features)
        self.linear_neighbor = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, self_emb, influence_emb, adj):
        # Self contribution: W_self * self_emb[i]
        self_out = self.linear_self(self_emb)  # [B, N, D']
        
        # Neighbor contribution: adj @ W_neighbor(influence_emb)
        neighbor_support = self.linear_neighbor(influence_emb)  # [B, N, D']
        neighbor_out = torch.matmul(adj, neighbor_support)  # [B, N, D']
        
        output = self_out + neighbor_out
        return self.dropout(F.relu(output))