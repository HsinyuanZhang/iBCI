"""SPINT model architecture - part of "SPINT: Spatial Permutation-Invariant Neural Transformer for Consistent Intracortical Motor Decoding".
Scaffolding adapted from the Hydra template (ashleve/lightning-hydra-template).
Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Trung Le.
"""
import torch
import torch.nn as nn
import random

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, attn_mask=None, key_padding_mask=None):
        # Pre-norm before cross-attention
        query_norm = self.norm1(query)
        key_value_norm = self.norm1(key_value)
        cross_attn_output, attn_score = self.cross_attn(
            query=query_norm,
            key=key_value_norm,
            value=key_value_norm,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        x = query + self.dropout(cross_attn_output)

        # Pre-norm before feedforward
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + self.dropout(ffn_output)
        return x, attn_score


class MultiLayerCrossAttention(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, query, key_value, attn_mask=None, key_padding_mask=None):
        x = query
        attn_scores = []
        for layer in self.layers:
            x, attn_score = layer(x, key_value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
            attn_scores.append(attn_score)
        return x, attn_scores


class SpintModel(nn.Module):
    def __init__(self, model_dim, num_covariates, window_size,
                 num_heads=2, num_layers=1, num_id_layers=1,
                 use_learnable_id=True, learnable_id_type='mlp', learnable_rep=True,
                 dropout_rate=0.0, dynamic_dropout=False, dynamic_dropout_low=0.0, dynamic_dropout_high=1.0, tf_drop_rate=0.1, readin_layer_type='mlp',
                ):
        super(SpintModel, self).__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_id_layers = num_id_layers
        self.num_covariates = num_covariates
        self.window_size = window_size
        self.use_learnable_id = use_learnable_id
        self.learnable_id_type = learnable_id_type
        self.learnable_rep = learnable_rep
        self.dropout_rate = dropout_rate
        self.tf_drop_rate = tf_drop_rate
        self.dynamic_dropout = dynamic_dropout
        self.dynamic_dropout_low = dynamic_dropout_low
        self.dynamic_dropout_high = dynamic_dropout_high
        self.readin_layer_type = readin_layer_type

        self.fc_in = nn.Sequential(nn.Linear(window_size, model_dim),
                                   nn.ReLU(),
                                   nn.Linear(model_dim, model_dim))
        if self.use_learnable_id:
            if self.learnable_id_type == 'mlp':
                in_layers = [nn.LazyLinear(model_dim)]
                for _ in range(self.num_id_layers - 1):
                    in_layers.extend([nn.ReLU(), nn.Linear(model_dim, model_dim)])
                self.fc_id_in = nn.Sequential(*in_layers)
                out_layers = []
                for _ in range(self.num_id_layers - 1):
                    out_layers.extend([nn.Linear(model_dim, model_dim), nn.ReLU()])
                out_layers.append(nn.Linear(model_dim, window_size))
                self.fc_id_out = nn.Sequential(*out_layers)
            else:
                raise ValueError(f"Unsupported learnable ID type: {self.learnable_id_type}")
        else:
            raise ValueError(f"Non-learnable ID not supported. Must set use_learnable_id to True.")
                

        self.fc_out = nn.Linear(model_dim, window_size)
        if learnable_rep:       
            self.rep = nn.Parameter(torch.randn(1, num_covariates, window_size)) # 1xCxW
        else:
            self.rep = torch.arange(start=1, end=num_covariates+1).unsqueeze(0).unsqueeze(-1).repeat(1, 1, window_size)/num_covariates # 1xCxW

        self.transformer = MultiLayerCrossAttention(num_layers=num_layers, d_model=model_dim, nhead=num_heads, dropout=self.tf_drop_rate)
        
    
    def forward(self, src, calib_trialized_neural_features=None):
        src = src.permute(0, 2, 1) # BxWxN -> BxNxW (batch size, num neurons, window size)
        
        batch_size = src.size(0)
        num_neurons = src.size(1)

        if self.use_learnable_id: 
            if self.learnable_id_type == 'mlp':
                id = calib_trialized_neural_features.permute(0,1,3,2) # BxMxTxN -> BxMxNxT
                id = self.fc_id_in(id) # BxMxNxT -> BxMxNxH
                id = torch.mean(id, dim=1, keepdim=False) # BxMxNxH -> BxNxH
                id = self.fc_id_out(id) # BxNxH -> BxNxW
            else:            
                raise ValueError(f"Unsupported learnable ID type: {self.learnable_id_type}")

            src = src + id # BxNxW + BxNxW -> BxNxW
        else:            
            raise ValueError(f"Non-learnable ID not supported. Must set use_learnable_id to True.")

        dropout_mask = torch.ones(batch_size, num_neurons).to(src) # BxN
        if self.dynamic_dropout:
            p = random.uniform(self.dynamic_dropout_low, self.dynamic_dropout_high)
            dropout_mask = torch.nn.functional.dropout(dropout_mask, p=p, training=self.training) # BxN
        else:
            dropout_mask = torch.nn.functional.dropout(dropout_mask, p=self.dropout_rate, training=self.training) # BxN
        src = src * dropout_mask.unsqueeze(-1) # BxNxW * (BxN -> BxNx1) -> BxNxW

        if self.readin_layer_type == 'mlp':
            src = self.fc_in(src) # BxNxW -> BxNxH
        else:
            raise ValueError(f"Unsupported readin_layer_type: {self.readin_layer_type}")

        rep = self.fc_in(self.rep).to(src) # 1xCxW -> 1xCxH

        transformer_output, _ = self.transformer(rep.repeat(src.size(0), 1, 1), src) # BxCxH

        output = self.fc_out(transformer_output) # BxCxH -> BxCxW
        behavior_pred = output.permute(0, 2, 1) # BxCxW -> BxWxC
        return behavior_pred