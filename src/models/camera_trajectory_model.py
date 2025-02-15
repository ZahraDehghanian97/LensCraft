from typing import Optional

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

    def inference(
            self,
            caption_embedding: Optional[torch.Tensor] = None,
            camera_trajectory: Optional[torch.Tensor] = None,
            subject_trajectory: torch.Tensor = None,
            teacher_forcing_ratio: float = 0.0,
            src_key_mask: Optional[torch.Tensor] = None,
            padding_mask: Optional[torch.Tensor] = None,
            decode_mode: str = 'single_step'
        ) -> dict:
        if subject_trajectory is None:
            raise ValueError("subject_trajectory cannot be None")

        with torch.no_grad():
            device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
            
            subject_trajectory = subject_trajectory.to(device)
            
            if caption_embedding is not None:
                caption_embedding = caption_embedding.to(device)
            elif camera_trajectory is None:
                raise ValueError("Both camera_trajectory and caption_embedding cannot be None")
            
            if padding_mask is not None:
                padding_mask = padding_mask.to(device)
            
            # If camera trajectory is provided, use the full model
            if camera_trajectory is not None:
                camera_trajectory = camera_trajectory.to(device)
                if src_key_mask is not None:
                    src_key_mask = src_key_mask.to(device)
                
                return self.forward(
                    src=camera_trajectory,
                    subject_trajectory=subject_trajectory,
                    caption_embedding=caption_embedding,
                    teacher_forcing_ratio=teacher_forcing_ratio,
                    src_key_mask=src_key_mask,
                    tgt_key_padding_mask=padding_mask,
                    decode_mode=decode_mode
                )
            
            # If no camera trajectory, use only the decoder
            else:
                subject_embedding = self.subject_projection(subject_trajectory)
                memory = self.prepate_decoder_memory(caption_embedding=caption_embedding)
                reconstructed = self.decoder(
                    memory=memory,
                    subject_embedding=subject_embedding,
                    decode_mode=decode_mode,
                    tgt_key_padding_mask=padding_mask
                )
                return {'reconstructed': reconstructed}