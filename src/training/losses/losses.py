import torch
from torch.nn.functional import mse_loss
from .angle_loss import AngleLoss
from .contrastive_loss import ContrastiveLoss
from .clip_loss import ClipLoss


class CameraTrajectoryLoss:
    def __init__(self, 
                 contrastive_loss_margin: int=5, 
                 n_clip_embs: int=28, 
                 losses_list: list=[], 
                 weighted_clip_loss: bool=False,
                 weight_power: int=1,
                 clip_weights: dict=None,
                 contrastive_loss_version: int=1,
                 clip_loss_scaling_factor: float=37500,
                 trajectory_loss_scaling_factor: float=10,
                 contrastive_loss_scaling_factor: float=0.1,
                 angle_loss_scaling_factor: float=180,
                 clip_embeddings: dict=None,
                 encoder_loss_function: str="clip"
                 ):
        self.angle_loss = AngleLoss(angle_loss_scaling_factor)
        self.clip_loss = ClipLoss(clip_weights=clip_weights, weight_power=weight_power)
        
        self.position_slice = slice(0, 4)
        self.rotation_slice = slice(4, None)
        
        self.contrastive_loss_margin = contrastive_loss_margin
        self.n_clip_embs = n_clip_embs
        self.losses_list = losses_list
        self.weighted_clip_loss = weighted_clip_loss
        self.clip_weights = clip_weights
        self.clip_embeddings = clip_embeddings
        
        self.clip_loss_scaling_factor = clip_loss_scaling_factor
        self.trajectory_loss_scaling_factor = trajectory_loss_scaling_factor
        self.contrastive_loss_scaling_factor = contrastive_loss_scaling_factor
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.contrastive_loss = ContrastiveLoss(
            clip_embeddings=clip_embeddings,
            device=self.device,
            get_embedding_name_func=self.get_embedding_name
        )
        
        self.contrastive_loss_version = contrastive_loss_version
        if self.contrastive_loss_version == 1:
            self.compute_contrastive_loss = self.contrastive_loss.compute_v1
        elif self.contrastive_loss_version == 2:
            self.compute_contrastive_loss = self.contrastive_loss.compute_v2
        else:
            raise ValueError(f"Contrastive loss version should be 1 or 2, you passed {self.contrastive_loss_version}")
        
        self.encoder_loss_function = encoder_loss_function

    def __call__(self, model_output, camera_trajectory, clip_target, batch, tgt_key_padding_mask=None):
        reconstructed = model_output['reconstructed']
        clip_pred = model_output['embeddings']
        
        return self.compute_total_loss(
            reconstructed,
            camera_trajectory,
            clip_pred,
            clip_target,
            batch
        )

    def compute_total_loss(self, trajectory_pred, trajectory_target, clip_pred, clip_target, batch):
        if len(self.losses_list) == 0:
            raise ValueError("The losses list cannot be empty; it must contain at least one of the following: 'trajectory', 'clip', or 'contrastive'.")
        
        total_loss = 0
        loss_dict = dict()

        if "clip" in self.losses_list:
            total_clip_loss_weighted, clip_losses, total_clip_loss = self.clip_loss.compute(
                clip_target=clip_target, 
                clip_pred=clip_pred, 
                n_clip_embs=self.n_clip_embs, 
                weighted_clip_loss=self.weighted_clip_loss,
                batch=batch,
                encoder_loss_function=self.encoder_loss_function
            )
            
            if self.weighted_clip_loss:
                total_loss += total_clip_loss_weighted / clip_pred.shape[1] * self.clip_loss_scaling_factor
            else:
                if self.encoder_loss_function == "clip":
                    total_loss += total_clip_loss / clip_pred.shape[1] * self.clip_loss_scaling_factor
                elif self.encoder_loss_function == "mse":
                    total_loss += total_clip_loss / clip_pred.shape[1] * 12800

            loss_dict["clip"] = {i: clip_losses[i] for i in range(self.n_clip_embs)}
            loss_dict["average_clip"] = total_clip_loss * 200

        if "trajectory" in self.losses_list:
            trajectory_loss = self.compute_trajectory_loss(trajectory_pred, trajectory_target)
            total_loss += trajectory_loss * self.trajectory_loss_scaling_factor
            loss_dict["trajectory"] = trajectory_loss.item()
        
        if "contrastive" in self.losses_list:
            contrastive_loss = self.compute_contrastive_loss(clip_pred, clip_target, batch)
            total_loss += contrastive_loss * self.contrastive_loss_scaling_factor
            loss_dict["contrastive"] = contrastive_loss.item()

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict
    
    def compute_trajectory_only_loss(self, model_output, camera_trajectory, tgt_key_padding_mask=None):
        reconstructed = model_output['reconstructed']
        trajectory_loss = self.compute_trajectory_loss(reconstructed, camera_trajectory)
        
        loss_dict = {
            "trajectory": trajectory_loss.item(),
            "total": trajectory_loss.item()
        }
        
        return trajectory_loss, loss_dict

    def compute_component_losses(self, pred, target):
        position_loss = mse_loss(
            pred[..., self.position_slice], 
            target[..., self.position_slice]
        )
        rotation_loss = self.angle_loss(
            pred[..., self.rotation_slice], 
            target[..., self.rotation_slice]
        )
        return position_loss + rotation_loss

    def compute_trajectory_loss(self, pred, target):
        first_frame_loss = self.compute_component_losses(pred[:, 0:1], target[:, 0:1])
        relative_loss = self.compute_component_losses(pred[:, 1:] - pred[:, 0:1], target[:, 1:] - target[:, 0:1])
        
        return relative_loss * self.trajectory_loss_scaling_factor + first_frame_loss
    
    @staticmethod
    def get_embedding_name(name: str) -> str:
        embedding_name = str(name).split(".")[-1].lower()
        embedding_name = embedding_name.split("_")
        embedding_name = embedding_name[0] + "".join([item.capitalize() for item in embedding_name[1:]])
        return embedding_name