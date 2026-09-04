import torch
import torch.nn as nn

from .positional_encoding import PositionalEncoding


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, nhead, num_encoder_layers, dim_feedforward, dropout_rate, num_query_tokens):
        super(Encoder, self).__init__()

        self.input_projection = nn.Linear(input_dim, latent_dim)
        self.pos_encoder = PositionalEncoding(latent_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers)

        self.query_tokens = nn.Parameter(torch.randn(num_query_tokens, 1, latent_dim))

    def forward(
        self,
        src,
        subject_embedded,
        src_key_padding_mask=None,
        subject_key_padding_mask=None,
    ):
        src_embedded = self.input_projection(src)
        src_embedded = torch.cat([subject_embedded, src_embedded], dim=1)
        src_embedded = self.pos_encoder(src_embedded)
        src_embedded = src_embedded.permute(1, 0, 2)

        query_tokens = self.query_tokens.repeat(1, src_embedded.shape[1], 1)
        src_with_queries = torch.cat([query_tokens, src_embedded], dim=0)

        batch_size, frame_count = src.shape[:2]
        if subject_embedded.shape[:2] != (batch_size, frame_count + 1):
            raise ValueError(
                "subject_embedded must contain one token per frame followed "
                "by exactly one volume token"
            )

        if (
            src_key_padding_mask is not None
            or subject_key_padding_mask is not None
        ):
            def normalize_mask(mask, name):
                if mask is None:
                    return torch.zeros(
                        (batch_size, frame_count),
                        dtype=torch.bool,
                        device=src.device,
                    )
                mask = mask.to(device=src.device, dtype=torch.bool)
                if mask.shape != (batch_size, frame_count):
                    raise ValueError(
                        f"{name} must have shape "
                        f"{(batch_size, frame_count)}, got {tuple(mask.shape)}"
                    )
                return mask

            camera_mask = normalize_mask(
                src_key_padding_mask, "src_key_padding_mask"
            )
            subject_mask = normalize_mask(
                subject_key_padding_mask, "subject_key_padding_mask"
            )
            query_mask = torch.zeros(
                (batch_size, self.query_tokens.shape[0]),
                dtype=torch.bool,
                device=src.device,
            )
            volume_mask = torch.zeros(
                (batch_size, 1), dtype=torch.bool, device=src.device
            )
            # Token order is queries, subject frames, subject volume, camera
            # frames. Keeping the volume mask separate avoids shifting the
            # subject mask by one and accidentally hiding the volume whenever
            # the final trajectory frame is padded/masked.
            src_key_padding_mask = torch.cat(
                [
                    query_mask,
                    subject_mask,
                    volume_mask,
                    camera_mask,
                ],
                dim=1,
            )

        memory = self.transformer_encoder(
            src_with_queries, src_key_padding_mask=src_key_padding_mask)

        return memory[:self.query_tokens.shape[0]]
