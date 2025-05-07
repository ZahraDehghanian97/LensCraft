import torch
import torch.nn as nn
import gc

from .positional_encoding import PositionalEncoding

class Decoder(nn.Module):
    def __init__(self, output_dim, latent_dim, nhead, num_decoder_layers, dim_feedforward, dropout_rate, seq_length=30):
        super(Decoder, self).__init__()

        self.output_dim = output_dim
        self.seq_length = seq_length
        self.pos_encoder = PositionalEncoding(latent_dim)
        self.embedding = nn.Linear(output_dim, latent_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout_rate)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers)

        self.output_projection = nn.Linear(latent_dim, output_dim)

    def prepare_decoder_inputs_with_positioning(self, decoder_input, subject_embedding, tgt_key_padding_mask=None):
        embedded = self.embedding(decoder_input)
        embedded = torch.cat([subject_embedding, embedded], dim=1)
        embedded = self.pos_encoder(embedded)
        embedded = embedded.transpose(0, 1)

        if tgt_key_padding_mask is not None:
            batch_size = tgt_key_padding_mask.shape[0]
            subject_volume_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=tgt_key_padding_mask.device)
            tgt_key_padding_mask = torch.cat(
                [tgt_key_padding_mask, subject_volume_mask, tgt_key_padding_mask], dim=1)

        return embedded, tgt_key_padding_mask

    def single_step_decode(self, memory, subject_embedding, tgt_key_padding_mask=None):
        decoder_input = torch.zeros(
            memory.shape[1], self.seq_length, self.output_dim, device=memory.device)

        embedded, tgt_key_padding_mask = self.prepare_decoder_inputs_with_positioning(
            decoder_input, subject_embedding, tgt_key_padding_mask)

        output = self.transformer_decoder(
            tgt=embedded, memory=memory, tgt_key_padding_mask=tgt_key_padding_mask)
        output = output.transpose(0, 1)
        output = self.output_projection(output[:, -self.seq_length:, :])

        return output

    def autoregressive_decode(self, memory, subject_embedding, target=None, teacher_forcing_ratio=0.5, tgt_key_padding_mask=None):
        batch_size = memory.shape[1]
        device = memory.device
        
        output_trajectory = torch.zeros(batch_size, self.seq_length, self.output_dim, device=device)
        decoder_input = torch.zeros(batch_size, self.seq_length, self.output_dim, device=device)
        
        subj_len = subject_embedding.shape[1]
        
        causal_mask = torch.triu(torch.ones(2 * self.seq_length + 1, 2 * self.seq_length + 1, device=device) * float('-inf'), diagonal=1)
        
        for t in range(self.seq_length):
            embedded, padded_mask = self.prepare_decoder_inputs_with_positioning(
                decoder_input, subject_embedding, tgt_key_padding_mask)
                        
            decoder_output = self.transformer_decoder(
                tgt=embedded,
                memory=memory,
                tgt_mask=causal_mask[:embedded.size(0), :embedded.size(0)],
                tgt_key_padding_mask=padded_mask
            )
            
            decoder_output = decoder_output.transpose(0, 1)
            current_pred = self.output_projection(decoder_output[:, subj_len + t, :])
            
            output_trajectory[:, t, :] = current_pred
            
            if t < self.seq_length - 1:
                new_decoder_input = decoder_input.clone()
                use_target = (target is not None and torch.rand(1).item() < teacher_forcing_ratio)
                new_decoder_input[:, t, :] = target[:, t, :] if use_target else current_pred
                decoder_input = new_decoder_input
        
        return output_trajectory

    def forward(self, memory, subject_embedding, decode_mode='single_step', target=None, teacher_forcing_ratio=0.0, tgt_key_padding_mask=None):
        if decode_mode == 'autoregressive':
            return self.autoregressive_decode(memory, subject_embedding, target, teacher_forcing_ratio, tgt_key_padding_mask)
        elif decode_mode == 'single_step':
            return self.single_step_decode(memory, subject_embedding, tgt_key_padding_mask)
        else:
            raise ValueError(f"Unknown decode_mode: {decode_mode}")
