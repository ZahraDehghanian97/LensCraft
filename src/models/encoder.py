import torch
import torch.nn as nn

from .positional_encoding import PositionalEncoding


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, nhead, num_encoder_layers, dim_feedforward, dropout_rate, query_token_names):
        super(Encoder, self).__init__()

        self.input_projection = nn.Linear(input_dim, latent_dim)
        self.pos_encoder = PositionalEncoding(latent_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers)

        self.query_tokens = nn.ParameterDict({
            f"{qt}_query": nn.Parameter(torch.randn(1, 1, latent_dim))
            for qt in query_token_names
        })

    def forward(self, src, subject_embedded, src_key_padding_mask=None):
        src_embedded = self.input_projection(src)
        src_embedded = torch.cat([subject_embedded, src_embedded], dim=1)
        src_embedded = self.pos_encoder(src_embedded)
        src_embedded = src_embedded.permute(1, 0, 2)

        query_tokens = torch.cat([query.repeat(
            1, src_embedded.shape[1], 1) for query in self.query_tokens.values()], dim=0)
        src_with_queries = torch.cat([query_tokens, src_embedded], dim=0)

        if src_key_padding_mask is not None:
            query_mask = torch.zeros((src_key_padding_mask.shape[0], len(
                self.query_tokens)), dtype=torch.bool, device=src.device)
            # subject_padding_mask = torch.zeros_like(src_key_padding_mask)
            src_key_padding_mask = torch.cat(
                [query_mask, src_key_padding_mask, src_key_padding_mask], dim=1)

        memory = self.transformer_encoder(
            src_with_queries, src_key_padding_mask=src_key_padding_mask)

        return memory[:len(self.query_tokens)]
