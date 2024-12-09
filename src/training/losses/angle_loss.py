import math
import torch

class AngleLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def euler_to_normal(self, angles):
        sin_angles = torch.sin(angles)
        cos_angles = torch.cos(angles)
        
        sx, sy, sz = sin_angles[:, 0], sin_angles[:, 1], sin_angles[:, 2]
        cx, cy, cz = cos_angles[:, 0], cos_angles[:, 1], cos_angles[:, 2]
        
        normal = torch.stack([
            sx * sz - cx * cz * sy,
            cz * sx + cx * sy * sz,
            cx * cy
        ], dim=1)
                
        return normal
    
    def forward(self, pred, target):
        pred_normal = self.euler_to_normal(pred)
        target_normal = self.euler_to_normal(target)
        
        loss = torch.mean((pred_normal - target_normal) ** 2) * 180
        
        return loss


def test_angle_loss():
    criterion = AngleLoss()

    # Test case 1: Same angles with different representations
    pred1 = torch.tensor([[0.0, -1.3785714917577172, 0.0]])
    target1 = torch.tensor([[math.pi, -1.3785714917577172, math.pi]])
    target2 = torch.tensor([[-math.pi, -1.3785714917577172, -math.pi]])

    loss1 = criterion(pred1, target1)
    loss2 = criterion(pred1, target2)

    print("Test case 1:")
    print(f"Loss between pred and target1: {loss1.item():.6f}")
    print(f"Loss between pred and target2: {loss2.item():.6f}")

    # Test case 2: Different angles
    pred2 = torch.tensor([[0.0, 0.0, 0.0]])
    target3 = torch.tensor([[math.pi/2, math.pi/2, math.pi/2]])

    loss3 = criterion(pred2, target3)
    print("\nTest case 2:")
    print(f"Loss between different angles: {loss3.item():.6f}")

    # Test case 3: Batch processing
    pred_batch = torch.tensor([
        [0.0, -1.3785714917577172, 0.0],
        [math.pi/4, 0.0, -math.pi/4]
    ])
    target_batch = torch.tensor([
        [math.pi, -1.3785714917577172, math.pi],
        [math.pi/4, 0.0, -math.pi/4]
    ])

    loss_batch = criterion(pred_batch, target_batch)
    print("\nTest case 3:")
    print(f"Batch loss: {loss_batch.item():.6f}")

if __name__ == 'main':
    test_angle_loss()