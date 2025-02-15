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
        self.memory_tokens_count = cinematography_struct_size

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

    def prepate_decoder_memory(self, camera_embedding=None, caption_embedding=None, teacher_forcing_ratio=0.0, mask_memory_prob=0.0):
        if camera_embedding is None:
            if caption_embedding is None:
                raise ValueError("Both memory and caption_embedding cannot be None")
            merged_memory = caption_embedding
        else:
            if self.use_merged_memory:
                _, B, _ = camera_embedding.shape
                merged_memory = self.embedding_merger(camera_embedding.transpose(0, 1).reshape(B, -1)).unsqueeze(0)
            else:
                merged_memory = camera_embedding[:self.memory_tokens_count]
            
            if teacher_forcing_ratio > 0 and caption_embedding is not None:
                merged_memory = (1-teacher_forcing_ratio) * merged_memory + teacher_forcing_ratio * caption_embedding
        
        if mask_memory_prob > 0.0:
            memory_mask = (torch.rand(merged_memory.shape[0], 
                                    device=merged_memory.device) > mask_memory_prob).float().unsqueeze(1).unsqueeze(2)
            merged_memory = merged_memory * memory_mask
            
        return merged_memory

    def forward(self, src, subject_trajectory, tgt_key_padding_mask=None, src_key_mask=None, target=None, 
                caption_embedding=None, teacher_forcing_ratio=0.5, mask_memory_prob=0.0, decode_mode='single_step'):
        subject_embedding = self.subject_projection(subject_trajectory)
        camera_embedding = self.encoder(src, subject_embedding, src_key_mask)
        
        memory = self.prepate_decoder_memory(
            camera_embedding=camera_embedding,
            caption_embedding=caption_embedding,
            teacher_forcing_ratio=teacher_forcing_ratio,
            mask_memory_prob=mask_memory_prob
        )

        reconstructed = self.decoder(
            memory=memory,
            subject_embedding=subject_embedding,
            decode_mode=decode_mode,
            target=target,
            teacher_forcing_ratio=teacher_forcing_ratio,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        output = {
            'embeddings': camera_embedding,
            'reconstructed': reconstructed,
        }
        
        if self.use_merged_memory:
            output['cls_embedding'] = memory[0]

        return output