import torch
from torch.nn.functional import mse_loss
from .angle_loss import AngleLoss
from .contrastive_loss import ContrastiveLoss
from .clip_loss import ClipLoss

from data.simulation.utils import load_clip_means

class CameraTrajectoryLoss:
    def __init__(self,
                 contrastive_loss_margin: int=5,
                 losses_list: list=[],
                 weighted_clip_loss: bool=False,
                 weight_power: int=1,
                 clip_weights: dict=None,
                 contrastive_loss_version: int=1,
                 clip_embeddings: dict=None,
                 encoder_loss_function: str="clip"
                 ):
        self.angle_loss = AngleLoss()
        self.clip_loss = ClipLoss(clip_weights=clip_weights, weight_power=weight_power)
        self.contrastive_loss_margin = contrastive_loss_margin
        self.losses_list = losses_list
        self.weighted_clip_loss = weighted_clip_loss
        self.clip_weights = clip_weights
        self.clip_embeddings = clip_embeddings
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.embedding_means = load_clip_means()
        
        self.contrastive_loss = ContrastiveLoss(
            clip_embeddings=clip_embeddings,
            device=self.device,
            embedding_means=self.embedding_means,
            get_embedding_name_func=self.get_embedding_name
        )
        
        self.contrastive_loss_version = contrastive_loss_version
        if self.contrastive_loss_version == 1:
            self.compute_contrastive_loss = self.contrastive_loss.compute_v1
        elif self.contrastive_loss_version == 2:
            self.compute_contrastive_loss = self.contrastive_loss.compute_v2
        elif self.contrastive_loss_version == 3:
            self.compute_contrastive_loss = self.contrastive_loss.compute_v3
        else:
            raise ValueError(f"Contrastive loss version should be 1 or 2, you passed {self.contrastive_loss_version}")
        
        self.encoder_loss_function = encoder_loss_function

    def __call__(self, model_output, camera_trajectory, clip_target, batch, tgt_key_padding_mask=None):
        reconstructed = model_output['reconstructed']
        clip_pred = model_output['embeddings']
        cycle_embeddings = model_output.get('cycle_embeddings', None)
        
        return self.compute_total_loss(
            reconstructed,
            camera_trajectory,
            clip_pred,
            clip_target,
            batch,
            cycle_embeddings
        )

    def compute_total_loss(self, trajectory_pred, trajectory_target, clip_pred, clip_target, batch, cycle_embeddings=None):
        loss_dict = dict()

        if self.losses_list.get("clip", 0):
            clip_losses, total_clip_loss = self.clip_loss.compute(
                clip_target=clip_target,
                clip_pred=clip_pred,
                weighted_clip_loss=self.weighted_clip_loss,
                prompt_none_mask=batch.get("prompt_none_mask", None),
                encoder_loss_function=self.encoder_loss_function
            )

            loss_dict["clip"] = total_clip_loss
            loss_dict["clip_elements"] = {i: clip_losses[i] for i in range(len(clip_losses))}



        if self.losses_list.get("cycle", 0) and cycle_embeddings is not None:
            cycle_losses, total_cycle_loss = self.clip_loss.compute(
                clip_target=clip_pred,
                clip_pred=cycle_embeddings,
                weighted_clip_loss=self.weighted_clip_loss,
                encoder_loss_function=self.encoder_loss_function
            )
            
            loss_dict["cycle"] = total_cycle_loss 
            loss_dict["cycle_elements"] = {i: cycle_losses[i] for i in range(len(cycle_losses))} # FIXME

        elif self.losses_list.get("cycle", 0) and cycle_embeddings is None:
            loss_dict["cycle"] = torch.tensor(0)


        if self.losses_list.get("first_frame", 0) or self.losses_list.get("relative", 0) or self.losses_list.get("speed", 0):
            first_frame_loss, relative_loss, speed_loss = self.compute_trajectory_loss(trajectory_pred, trajectory_target)
            
            if self.losses_list.get("first_frame", 0):
                loss_dict["first_frame"] = first_frame_loss
            
            if self.losses_list.get("relative", 0):
                loss_dict["relative"] = relative_loss
            
            if self.losses_list.get("speed", 0):
                loss_dict["speed"] = speed_loss


        if self.losses_list.get("contrastive", 0):
            contrastive_loss = self.compute_contrastive_loss(clip_pred, clip_target, batch)
            loss_dict["contrastive"] = contrastive_loss



        total_loss = 0
        loss_factor = 1
        for loss_key in self.losses_list:
            if loss_key == "cycle" and \
                    "cycle" in loss_dict and \
                    "clip" in loss_dict and \
                    loss_dict["cycle"] < loss_dict["clip"]:
                loss_factor = 0.5
            total_loss += self.losses_list[loss_key] * loss_factor * loss_dict.get(loss_key, 0)
            
        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict
    

    def compute_component_losses(self, pred, target):
        position_loss = mse_loss(
            pred[..., :3],
            target[..., :3]
        )
        rotation_loss = self.angle_loss(
            pred[..., 3:],
            target[..., 3:]
        )
        return position_loss + rotation_loss

    def compute_trajectory_loss(self, pred, target):
        first_frame_loss = self.compute_component_losses(pred[:, 0:1], target[:, 0:1])
        relative_loss = self.compute_component_losses(pred[:, 1:] - pred[:, 0:1], target[:, 1:] - target[:, 0:1])
        speed_loss = self.compute_component_losses(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])
        return first_frame_loss, relative_loss, speed_loss
    
    @staticmethod
    def get_embedding_name(name: str) -> str:
        embedding_name = str(name).split(".")[-1].lower()
        embedding_name = embedding_name.split("_")
        embedding_name = embedding_name[0] + "".join([item.capitalize() for item in embedding_name[1:]])
        return embedding_name
