import clip
import torch

class CaptionEncoder:
    def __init__(self, clip_model=None, device="cuda", max_token_length=None):
        self.device = torch.device(device)
        self.max_token_length = max_token_length
        
        if clip_model is None:
            self.clip_model = self.load_clip_model("ViT-B/32", device)
        else:
            self.clip_model = clip_model

    @staticmethod
    def load_clip_model(version="ViT-B/32", device="cuda"):
        model, _ = clip.load(version, device=device, jit=False)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model
    
    def encode_text(self, caption_raws):
        if self.max_token_length is not None:
            default_context_length = 77
            context_length = self.max_token_length + 2  # start_token + max_tokens + end_token
            assert context_length < default_context_length
            
            # Tokenize with context length limit
            texts = clip.tokenize(
                caption_raws, context_length=context_length, truncate=True
            )
            zero_pad = torch.zeros(
                [texts.shape[0], default_context_length - context_length],
                dtype=texts.dtype,
                device=texts.device,
            )
            texts = torch.cat([texts, zero_pad], dim=1)
        else:
            # Default tokenization (max 77 tokens)
            texts = clip.tokenize(caption_raws, truncate=True)

        # Process through CLIP model
        x = self.clip_model.token_embedding(texts.to(self.device)).type(self.clip_model.dtype)
        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)
        
        # Extract token features
        x_tokens = x[torch.arange(x.shape[0]), texts.argmax(dim=-1)].float()
        x_seq = [x[k, :(m + 1)].float() for k, m in enumerate(texts.argmax(dim=-1))]

        return x_seq, x_tokens