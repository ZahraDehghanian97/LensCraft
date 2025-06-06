from typing import List, Union, Optional
import torch
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class CLIPFeatures:
    sequence_features: torch.Tensor
    pooled_features: torch.Tensor
    attention_mask: torch.Tensor
    valid_lengths: torch.Tensor

    def to(self, device: Union[str, torch.device]) -> 'CLIPFeatures':
        return CLIPFeatures(
            sequence_features=self.sequence_features.to(device),
            pooled_features=self.pooled_features.to(device),
            attention_mask=self.attention_mask.to(device),
            valid_lengths=self.valid_lengths.to(device)
        )


class CLIPEmbedder:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        max_length: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        chunk_size: int = 100
    ):
        if not model_name.startswith('openai/'):
            model_name = 'openai/' + model_name

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_name,
            clean_up_tokenization_spaces=True
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name).to(self.device)
        self.text_encoder.eval()

        self.max_length = max_length or self.tokenizer.model_max_length
        self.chunk_size = chunk_size

        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def extract_clip_embeddings(
        self,
        texts: List[str],
        return_seq: bool = False,
        pad_seq: bool = False
    ) -> Union[torch.Tensor, CLIPFeatures]:
        num_texts = len(texts)
        chunks = [texts[i:i + self.chunk_size]
                 for i in range(0, num_texts, self.chunk_size)]
        
        all_sequence_features = []
        all_pooled_features = []
        all_attention_masks = []
        all_valid_lengths = []

        for chunk in chunks: # tqdm(chunks, desc="Processing text chunks", unit="chunk"):
            inputs = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.text_encoder(**inputs)
                sequence_features = outputs.last_hidden_state
                pooled_features = outputs.pooler_output

            if not return_seq:
                all_pooled_features.append(pooled_features)
                continue

            attention_mask = inputs.attention_mask
            valid_lengths = attention_mask.sum(dim=1)

            if pad_seq and sequence_features.size(1) < self.max_length:
                pad_size = self.max_length - sequence_features.size(1)
                sequence_features = F.pad(
                    sequence_features,
                    (0, 0, 0, pad_size),
                    mode='constant',
                    value=0
                )

            all_sequence_features.append(sequence_features)
            all_attention_masks.append(attention_mask)
            all_valid_lengths.append(valid_lengths)

        if not return_seq:
            return torch.cat(all_pooled_features, dim=0)

        return CLIPFeatures(
            sequence_features=torch.cat(all_sequence_features, dim=0),
            pooled_features=torch.cat(all_pooled_features, dim=0),
            attention_mask=torch.cat(all_attention_masks, dim=0),
            valid_lengths=torch.cat(all_valid_lengths, dim=0)
        )
        
    def get_caption_feat(
        self,
        prompts: List[str],
        seq_feat: bool = False,
        context_length: int = 77
    ) -> torch.Tensor:
        if seq_feat:
            clip_features = self.extract_clip_embeddings(
                prompts, 
                return_seq=True, 
                pad_seq=False
            )
            
            padded_seqs = []
            for i in range(len(prompts)):
                seq = clip_features.sequence_features[i]
                valid_length = clip_features.valid_lengths[i].item()
                valid_seq = seq[:valid_length]
                
                if valid_seq.shape[0] > context_length:
                    valid_seq = valid_seq[:context_length]
                
                padded_seq = F.pad(valid_seq, (0, 0, 0, context_length - valid_seq.shape[0]))
                padded_seqs.append(padded_seq)
            
            caption_feat = torch.stack(padded_seqs, dim=0)
            caption_feat = caption_feat.permute(0, 2, 1)  # [B, D, L]
        else:
            caption_feat = self.extract_clip_embeddings(prompts, return_seq=False)
        
        return caption_feat
