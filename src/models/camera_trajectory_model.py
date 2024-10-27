import torch
import torch.nn as nn
import gc

from .encoder import Encoder
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

    def forward(self, memory, decoder_input, subject_embedded, tgt_mask=None):
        embedded = self.embedding(decoder_input)
        embedded = torch.cat([subject_embedded, embedded], dim=1)
        embedded = self.pos_encoder(embedded)
        embedded = embedded.transpose(0, 1)

        output = self.transformer_decoder(embedded, memory, tgt_mask=tgt_mask)
        output = output.transpose(0, 1)
        output = self.output_projection(
            output[:, subject_embedded.size(1):, :])

        return output


class MultiTaskAutoencoder(nn.Module):
    def __init__(self, input_dim=7, subject_dim=9, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=2048,
                 dropout_rate=0.1, seq_length=30, latent_dim=512, query_token_names=['cls']):
        super(MultiTaskAutoencoder, self).__init__()

        self.subject_projection = nn.Linear(subject_dim, latent_dim)
        self.encoder = Encoder(input_dim, latent_dim, nhead,
                               num_encoder_layers, dim_feedforward, dropout_rate, query_token_names)
        self.decoder = Decoder(input_dim, latent_dim, nhead,
                               num_decoder_layers, dim_feedforward, dropout_rate)

        self.latent_merger = nn.Linear(
            latent_dim * len(query_token_names), latent_dim)

        self.seq_length = seq_length
        self.input_dim = input_dim
        self.query_token_names = query_token_names

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def single_step_decode(self, latent, subject_embedded):
        memory = latent.unsqueeze(1).repeat(1, self.seq_length, 1)
        memory = memory.transpose(0, 1)

        decoder_input = torch.zeros(
            latent.shape[0], self.seq_length, self.input_dim, device=latent.device)

        output = self.decoder(memory, decoder_input, subject_embedded)

        return output

    def autoregressive_decode(self, latent, subject_embedded, target=None, teacher_forcing_ratio=0.5):
        memory = latent.unsqueeze(1).repeat(1, self.seq_length, 1)
        memory = memory.transpose(0, 1)

        decoder_input = torch.zeros(
            latent.shape[0], 1, self.input_dim, device=latent.device)
        outputs = []

        for t in range(self.seq_length):
            tgt_mask = self.generate_square_subsequent_mask(
                t + 2).to(latent.device)

            output = self.decoder(memory, decoder_input,
                                  subject_embedded[:, t:t+1, :], tgt_mask)

            outputs.append(output[:, -1:, :])

            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = torch.cat(
                    [decoder_input, target[:, t:t+1, :]], dim=1)
            else:
                decoder_input = torch.cat(
                    [decoder_input, output[:, -1:, :]], dim=1)

            del output

        gc.collect()
        torch.cuda.empty_cache()

        return torch.cat(outputs, dim=1)

    def merge_latents(self, embeddings):
        reshaped = embeddings.permute(1, 0, 2).reshape(embeddings.shape[1], -1)
        return self.latent_merger(reshaped)

    def forward(self, src, subject_trajectory, src_key_padding_mask=None, target=None,
                teacher_forcing_ratio=0.5, decode_mode='single_step'):
        subject_embedded = self.subject_projection(subject_trajectory)
        embeddings = self.encoder(src, subject_embedded, src_key_padding_mask)
        latent = self.merge_latents(embeddings)

        if decode_mode == 'autoregressive':
            reconstructed = self.autoregressive_decode(
                latent, subject_embedded, target, teacher_forcing_ratio)
        elif decode_mode == 'single_step':
            reconstructed = self.single_step_decode(latent, subject_embedded)
        else:
            raise ValueError(f"Unknown decode_mode: {decode_mode}")

        return {
            **{f"{name}_embedding": embedding for name, embedding in zip(self.query_token_names, embeddings)},
            'reconstructed': reconstructed,
        }
