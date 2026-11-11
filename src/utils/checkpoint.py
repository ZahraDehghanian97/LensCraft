import torch


def load_checkpoint(checkpoint_path, model, device, *, checkpoint=None):
    if checkpoint is None:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    from utils.load_lens_craft import categorical_checkpoint_state, migrate_geometric_context_state

    state, _ = categorical_checkpoint_state(checkpoint)
    model.load_state_dict(migrate_geometric_context_state(model, state), strict=True)
    return model
