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

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _process_inputs(self, decoder_input, subject_embedding, tgt_key_padding_mask=None):
        embedded = self.embedding(decoder_input)
        embedded = torch.cat([subject_embedding, embedded], dim=1)
        embedded = self.pos_encoder(embedded)
        embedded = embedded.transpose(0, 1)

        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = torch.cat(
                [tgt_key_padding_mask, tgt_key_padding_mask], dim=1)

        return embedded, tgt_key_padding_mask

    def single_step_decode(self, memory, subject_embedding, tgt_key_padding_mask=None):
        decoder_input = torch.zeros(
            memory.shape[1], self.seq_length, self.output_dim, device=memory.device)

        embedded, tgt_key_padding_mask = self._process_inputs(
            decoder_input, subject_embedding, tgt_key_padding_mask)

        output = self.transformer_decoder(
            tgt=embedded, memory=memory, tgt_key_padding_mask=tgt_key_padding_mask)
        output = output.transpose(0, 1)
        output = self.output_projection(
            output[:, subject_embedding.size(1):, :])

        return output

    def autoregressive_decode(self, memory, subject_embedding, target=None, teacher_forcing_ratio=0.5):
        decoder_input = torch.zeros(
            memory.shape[1], 1, self.output_dim, device=memory.device)
        outputs = []

        for t in range(self.seq_length):
            tgt_mask = self.generate_square_subsequent_mask(
                t + 2).to(memory.device)

            embedded, _ = self._process_inputs(
                decoder_input, subject_embedding[:, t:t+1, :])

            output = self.transformer_decoder(
                tgt=embedded, memory=memory, tgt_mask=tgt_mask)
            output = output.transpose(0, 1)
            output = self.output_projection(
                output[:, subject_embedding.size(1):, :])

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

    def forward(self, memory, subject_embedding, decode_mode='single_step', target=None, teacher_forcing_ratio=0.5, tgt_key_padding_mask=None):
        if decode_mode == 'autoregressive':
            return self.autoregressive_decode(memory, subject_embedding, target, teacher_forcing_ratio)
        elif decode_mode == 'single_step':
            return self.single_step_decode(memory, subject_embedding, tgt_key_padding_mask)
        else:
            raise ValueError(f"Unknown decode_mode: {decode_mode}")
