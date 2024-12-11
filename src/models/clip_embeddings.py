from typing import Dict, List, Tuple, Union, Optional
import torch
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel
from dataclasses import dataclass


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
        device: Optional[Union[str, torch.device]] = None
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

        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def get_embeddings(
        self,
        texts: List[str],
        return_seq: bool = False,
        pad_seq: bool = False
    ) -> Union[torch.Tensor, CLIPFeatures]:
        inputs = self.tokenizer(
            texts,
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
            return pooled_features

        attention_mask = inputs.attention_mask
        valid_lengths = attention_mask.sum(dim=1)

        if pad_seq:
            padded_seq_features = sequence_features
            if sequence_features.size(1) < self.max_length:
                pad_size = self.max_length - sequence_features.size(1)
                padded_seq_features = F.pad(
                    sequence_features,
                    (0, 0, 0, pad_size),
                    mode='constant',
                    value=0
                )
            sequence_features = padded_seq_features

        return CLIPFeatures(
            sequence_features=sequence_features,
            pooled_features=pooled_features,
            attention_mask=attention_mask,
            valid_lengths=valid_lengths
        )

    def embed_descriptions(
        self,
        descriptions: Dict[str, List[str]],
        return_seq: bool = False,
        pad_seq: bool = False
    ) -> Dict[str, Union[torch.Tensor, CLIPFeatures]]:
        return {
            key: self.get_embeddings(value, return_seq, pad_seq).squeeze(0)
            for key, value in descriptions.items()
        }

    def get_sequence_embeddings(
        self,
        texts: List[str],
        return_attention_mask: bool = False
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        features = self.get_embeddings(texts, return_seq=True)

        sequences = []
        for i in range(len(texts)):
            seq_len = features.valid_lengths[i]
            seq_features = features.sequence_features[i, :seq_len]
            sequences.append(seq_features)

        if return_attention_mask:
            return sequences, features.attention_mask
        return sequences, None

    def encode_text(
        self,
        caption_raws: List[str],
        max_token_length: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        temp_max_length = self.max_length
        if max_token_length is not None:
            self.max_length = max_token_length + 2

        features = self.get_embeddings(caption_raws, return_seq=True)
        sequences = []

        for i in range(len(caption_raws)):
            seq_len = features.valid_lengths[i]
            seq_features = features.sequence_features[i, :seq_len]
            sequences.append(seq_features)

        self.max_length = temp_max_length

        return sequences, features.pooled_features


if __name__ == "__main__":
    embedder = CLIPEmbedder("clip-vit-base-patch32")

    texts = ["A camera slowly panning right", "Quick zoom in on the subject"]

    pooled_emb = embedder.get_embeddings(texts)
    print("Pooled shape:", pooled_emb.shape)

    seq_features = embedder.get_embeddings(texts, return_seq=True)
    print("Sequence shape:", seq_features.sequence_features.shape)
    print("Valid lengths:", seq_features.valid_lengths)

    sequences, attention_mask = embedder.get_sequence_embeddings(
        texts,
        return_attention_mask=True
    )
    print("First sequence shape:", sequences[0].shape)

    descriptions = {
        "movement": ["pan right", "pan left"],
        "speed": ["slowly", "quickly"]
    }
    all_embeddings = embedder.embed_descriptions(descriptions)
    for key, emb in all_embeddings.items():
        print(f"{key} embeddings shape:", emb.shape)
