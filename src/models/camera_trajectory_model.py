import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder
from data.simulation.constants import cinematography_struct_size, simulation_struct_size

class MultiTaskAutoencoder(nn.Module):
    def __init__(self, input_dim=7, subject_dim=9, nhead=4, num_encoder_layers=3, num_decoder_layers=3, 
                 dim_feedforward=2048, dropout_rate=0.1, seq_length=30, latent_dim=512, use_merged_memory=True):
        super(MultiTaskAutoencoder, self).__init__()
        
        self.num_query_tokens = cinematography_struct_size + simulation_struct_size
        self.num_memory = cinematography_struct_size

        self.subject_projection = nn.Linear(subject_dim, latent_dim)
        self.encoder = Encoder(input_dim, latent_dim, nhead,
                             num_encoder_layers, dim_feedforward, dropout_rate, self.num_query_tokens)
        self.decoder = Decoder(input_dim, latent_dim, nhead,
                             num_decoder_layers, dim_feedforward, dropout_rate, seq_length)
        
        self.embedding_merger = nn.Sequential(
            nn.Linear(self.num_query_tokens * latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        self.use_merged_memory = use_merged_memory

    def convert_encoder_memory(self, memory, dec_embeddings=None, teacher_forcing_ratio=0.0, mask_memory_prob=0.0):
        if self.use_merged_memory:
            _, B, _ = memory.shape
            merged_memory = self.embedding_merger(memory.transpose(0, 1).reshape(B, -1)).unsqueeze(0)
        else:
            merged_memory = memory[:self.num_memory]
        
        if teacher_forcing_ratio > 0 and dec_embeddings is not None:
            merged_memory = (1-teacher_forcing_ratio) * merged_memory + teacher_forcing_ratio * dec_embeddings
        
        if mask_memory_prob > 0.0:
            memory_mask = (torch.rand(merged_memory.shape[0], 
                                    device=merged_memory.device) > mask_memory_prob).float().unsqueeze(1).unsqueeze(2)
            merged_memory = merged_memory * memory_mask
            
        return merged_memory

    def forward(self, src, subject_trajectory, tgt_key_padding_mask=None, src_key_mask=None, target=None, 
                dec_embeddings=None, teacher_forcing_ratio=0.5, mask_memory_prob=0.0, decode_mode='single_step'):
        subject_embedded = self.subject_projection(subject_trajectory)
        memory = self.encoder(src, subject_embedded, src_key_mask)
        
        merged_memory = self.convert_encoder_memory(
            memory=memory,
            dec_embeddings=dec_embeddings,
            teacher_forcing_ratio=teacher_forcing_ratio,
            mask_memory_prob=mask_memory_prob
        )

        reconstructed = self.decoder(
            memory=merged_memory,
            subject_embedded=subject_embedded,
            decode_mode=decode_mode,
            target=target,
            teacher_forcing_ratio=teacher_forcing_ratio,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        output = {
            'embeddings': memory,
            'reconstructed': reconstructed,
        }
        
        if self.use_merged_memory:
            output['cls_embedding'] = merged_memory[0]

        return output