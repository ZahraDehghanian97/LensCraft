import torch
import torch.nn as nn
import gc

from .encoder import Encoder
from .decoder import Decoder

from data.simulation.constants import cinematography_struct_size, simulation_struct_size

class MultiTaskAutoencoder(nn.Module):
    def __init__(self, input_dim=7, subject_dim=9, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=2048,
                 dropout_rate=0.1, seq_length=30, latent_dim=512, use_merged_memory=True):
        super(MultiTaskAutoencoder, self).__init__()
        
        self.num_query_tokens = cinematography_struct_size + simulation_struct_size
        self.num_memory = simulation_struct_size # FIXME: temporal using simulation_struct_size instead cinematography_struct_size

        self.subject_projection = nn.Linear(subject_dim, latent_dim)
        self.encoder = Encoder(input_dim, latent_dim, nhead,
                               num_encoder_layers, dim_feedforward, dropout_rate, self.num_query_tokens)
        self.decoder = Decoder(input_dim, latent_dim, nhead,
                               num_decoder_layers, dim_feedforward, dropout_rate)
                               
        self.embedding_merger = nn.Sequential(
            nn.Linear(self.num_query_tokens * latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        self.seq_length = seq_length
        self.input_dim = input_dim
        self.use_merged_memory = use_merged_memory

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def single_step_decode(self, memory, subject_embedded, tgt_key_padding_mask=None):
        decoder_input = torch.zeros(
            memory.shape[1], self.seq_length, self.input_dim, device=memory.device)

        output = self.decoder(memory, decoder_input, subject_embedded, tgt_key_padding_mask)

        return output

    def autoregressive_decode(self, memory, subject_embedded, target=None, teacher_forcing_ratio=0.5):
        decoder_input = torch.zeros(
            memory.shape[1], 1, self.input_dim, device=memory.device)
        outputs = []

        for t in range(self.seq_length):
            tgt_mask = self.generate_square_subsequent_mask(
                t + 2).to(memory.device)

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

    def forward(self, src, subject_trajectory, tgt_key_padding_mask=None, src_key_mask=None, target=None, dec_embeddings=None,
                teacher_forcing_ratio=0.5, mask_memory_prob=0.0, decode_mode='single_step'):
        subject_embedded = self.subject_projection(subject_trajectory)
        memory = self.encoder(src, subject_embedded, src_key_mask)
        
        if self.use_merged_memory:
            _, B, _ = memory.shape
            merged_memory = self.embedding_merger(memory.transpose(0, 1).reshape(B, -1)).unsqueeze(0)
        else:
            merged_memory = memory[:self.num_memory]
        
        if teacher_forcing_ratio > 0:
            merged_memory = (1-teacher_forcing_ratio) * merged_memory + teacher_forcing_ratio * dec_embeddings
        
        if mask_memory_prob > 0.0:
            memory_mask = (torch.rand(merged_memory.shape[0], device=merged_memory.device) > mask_memory_prob).float().unsqueeze(1).unsqueeze(2)
            merged_memory = merged_memory * memory_mask
        
        if decode_mode == 'autoregressive':
            reconstructed = self.autoregressive_decode(merged_memory, subject_embedded, target, teacher_forcing_ratio)
        elif decode_mode == 'single_step':
            reconstructed = self.single_step_decode(merged_memory, subject_embedded, tgt_key_padding_mask)
        else:
            raise ValueError(f"Unknown decode_mode: {decode_mode}")

        output = {
            'embeddings': memory,
            'reconstructed': reconstructed,
        }
        
        if self.use_merged_memory:
            output['cls_embedding'] = merged_memory[0]

        return output