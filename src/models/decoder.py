import torch
import torch.nn as nn

from .positional_encoding import PositionalEncoding


class Decoder(nn.Module):
    def __init__(self, output_dim, latent_dim, nhead, num_decoder_layers, dim_feedforward, dropout_rate):
        super(Decoder, self).__init__()

        self.pos_encoder = PositionalEncoding(latent_dim)
        self.embedding = nn.Linear(output_dim, latent_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout_rate)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers)

        self.output_projection = nn.Linear(latent_dim, output_dim)

    def forward(self, memory, decoder_input, subject_embedded, tgt_key_padding_mask=None):
        embedded = self.embedding(decoder_input)
        embedded = torch.cat([subject_embedded, embedded], dim=1)
        embedded = self.pos_encoder(embedded)
        embedded = embedded.transpose(0, 1)

        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = torch.cat(
                [tgt_key_padding_mask, tgt_key_padding_mask], dim=1)

        output = self.transformer_decoder(
            tgt=embedded, memory=memory, tgt_key_padding_mask=tgt_key_padding_mask)
        output = output.transpose(0, 1)
        output = self.output_projection(
            output[:, subject_embedded.size(1):, :])

        return output
