import torch
import torch.nn as nn
import gc

from .encoder import Encoder
from .decoder import Decoder


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

    def single_step_decode(self, latent, subject_embedded, tgt_key_padding_mask):
        decoder_input = torch.zeros(
            latent.shape[0], self.seq_length, self.input_dim, device=latent.device)

        output = self.decoder(latent, decoder_input,
                              subject_embedded, tgt_key_padding_mask)

        return output

    def autoregressive_decode(self, latent, subject_embedded, target=None, teacher_forcing_ratio=0.5):
        decoder_input = torch.zeros(
            latent.shape[0], 1, self.input_dim, device=latent.device)
        outputs = []

        for t in range(self.seq_length):
            tgt_mask = self.generate_square_subsequent_mask(
                t + 2).to(latent.device)

            output = self.decoder(latent, decoder_input,
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
        if len(self.query_token_names) == 1:
            return embeddings.squeeze(0)
        else:
            reshaped = embeddings.permute(
                1, 0, 2).reshape(embeddings.shape[1], -1)
            return self.latent_merger(reshaped)

    def forward(self, src, subject_trajectory, tgt_key_padding_mask=None, src_key_mask=None, target=None,
                teacher_forcing_ratio=0.5, decode_mode='single_step'):
        subject_embedded = self.subject_projection(subject_trajectory)
        embeddings = self.encoder(src, subject_embedded, src_key_mask)
        latent = self.merge_latents(embeddings)

        if decode_mode == 'autoregressive':
            reconstructed = self.autoregressive_decode(
                latent, subject_embedded, target, teacher_forcing_ratio)
        elif decode_mode == 'single_step':
            reconstructed = self.single_step_decode(
                latent, subject_embedded, tgt_key_padding_mask)
        else:
            raise ValueError(f"Unknown decode_mode: {decode_mode}")

        return {
            **{f"{name}_embedding": embedding for name, embedding in zip(self.query_token_names, embeddings)},
            'reconstructed': reconstructed,
        }
