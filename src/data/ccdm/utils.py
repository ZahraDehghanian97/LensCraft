import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

try:
    from data.convertor_utils import handle_single_or_batch
except:
    import sys
    sys.path.append('/home/arash/abolghasemi/LensCraft/src')
    from data.convertor_utils import handle_single_or_batch

def _axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle representation to a rotation matrix."""
    axis = F.normalize(axis, dim=-1, eps=1e-10)
    
    a_x, a_y, a_z = axis.unbind(-1)
    zeros = torch.zeros_like(a_x)
    K = torch.stack([
        zeros, -a_z, a_y,
        a_z, zeros, -a_x,
        -a_y, a_x, zeros
    ], dim=-1).reshape(axis.shape[:-1] + (3, 3))
    
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device).expand(axis.shape[:-1] + (3, 3))
    angle = angle[..., None, None]
    sin_a = torch.sin(angle)
    cos_a = torch.cos(angle)
    
    axis_outer = torch.einsum('...i,...j->...ij', axis, axis)
    
    return eye * cos_a + sin_a * K + (1.0 - cos_a) * axis_outer

@handle_single_or_batch()
def convert_ccdm_to_transform(ccdm: torch.Tensor, hfov_deg: float = 45.0, aspect: float = 16 / 9) -> torch.Tensor:
    """Convert from ccdm format to a 4x4 transformation matrix."""
    x, y, z, p_x, p_y = ccdm.unbind(-1)
    position = torch.stack([x, y, z], dim=-1)
    
    fwd = -position
    fwd = F.normalize(fwd, dim=-1, eps=1e-10)
    
    world_up = torch.tensor([0.0, 1.0, 0.0], dtype=ccdm.dtype, device=ccdm.device).expand_as(fwd)
    right = F.normalize(torch.cross(world_up, fwd, dim=-1), dim=-1, eps=1e-10)
    up = torch.cross(fwd, right, dim=-1)  # Already normalized
    
    R_cam = torch.stack([right, up, fwd], dim=-1)
    
    hfov = torch.deg2rad(torch.tensor(hfov_deg, dtype=ccdm.dtype, device=ccdm.device))
    vfov = 2.0 * torch.atan(torch.tan(hfov / 2.0) / aspect)
    
    yaw = -torch.atan(p_x * torch.tan(hfov / 2.0))
    pitch = torch.atan(p_y * torch.tan(vfov / 2.0))
    
    R_yaw = _axis_angle_to_matrix(world_up, yaw)
    rotated_right = torch.einsum('...ij,...j->...i', R_yaw, right)
    
    R_pitch = _axis_angle_to_matrix(rotated_right, pitch)
    
    rot_mat = torch.bmm(torch.bmm(R_pitch, R_yaw), R_cam) if ccdm.dim() > 1 else torch.matmul(torch.matmul(R_pitch, R_yaw), R_cam)
    
    batch_dims = position.shape[:-1]
    transform = torch.zeros((*batch_dims, 4, 4), dtype=ccdm.dtype, device=ccdm.device)
    transform[..., :3, :3] = rot_mat
    transform[..., :3, 3] = position
    transform[..., 3, 3] = 1.0
    
    return transform

@handle_single_or_batch()
def transform_to_ccdm(transform: torch.Tensor, hfov_deg: float = 45.0, aspect: float = 16 / 9) -> torch.Tensor:
    """Convert from 4x4 transformation matrix to ccdm format."""
    position = transform[..., :3, 3]
    rot_mat = transform[..., :3, :3]
    
    v_world = -position
    q_cam = torch.einsum('...ji,...j->...i', rot_mat, v_world)
    
    hfov = torch.deg2rad(torch.tensor(hfov_deg, dtype=transform.dtype, device=transform.device))
    vfov = 2.0 * torch.atan(torch.tan(hfov / 2.0) / aspect)
    
    p_x = (q_cam[..., 0] / (q_cam[..., 2] + 1e-10)) / torch.tan(hfov / 2.0)
    p_y = (q_cam[..., 1] / (q_cam[..., 2] + 1e-10)) / torch.tan(vfov / 2.0)
    
    return torch.cat([position, p_x.unsqueeze(-1), p_y.unsqueeze(-1)], dim=-1)


def visualize_camera_transforms(ccdms, transforms, titles):
    """Visualize camera positions and orientations in 3D space."""
    fig = plt.figure(figsize=(15, 5 * ((len(ccdms) + 2) // 3)))
    
    for i, (ccdm, transform, title) in enumerate(zip(ccdms, transforms, titles)):
        ax = fig.add_subplot(((len(ccdms) + 2) // 3), 3, i+1, projection='3d')
        
        position = transform[:3, 3].cpu().numpy() if torch.is_tensor(transform) else transform[:3, 3]
        rotation = transform[:3, :3].cpu().numpy() if torch.is_tensor(transform) else transform[:3, :3]
        
        ax.scatter([position[0]], [position[1]], [position[2]], color='red', s=100)
        
        axis_length = 1.0
        forward = rotation[:, 2]
        ax.quiver(position[0], position[1], position[2], 
                 forward[0], forward[1], forward[2], 
                 color='blue', length=axis_length)
        
        up = rotation[:, 1]
        ax.quiver(position[0], position[1], position[2], 
                 up[0], up[1], up[2], 
                 color='green', length=axis_length)
        
        right = rotation[:, 0]
        ax.quiver(position[0], position[1], position[2], 
                 right[0], right[1], right[2], 
                 color='red', length=axis_length)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        ax.set_box_aspect([1, 1, 1])
        
        limit = 5
        ax.set_xlim([-limit, limit])
        ax.set_ylim([-limit, limit])
        ax.set_zlim([-limit, limit])
    
    plt.tight_layout()
    plt.savefig('camera_transforms.png')
    plt.close()







    

if __name__ == "__main__":
    print("Testing CCDM utility functions...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_cases = [
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], device=device),
        torch.tensor([3.0, 2.0, 4.0, 0.0, 0.0], device=device),
        torch.tensor([3.0, 2.0, 4.0, 0.5, -0.3], device=device),
        torch.tensor([5.0, 0.0, 0.0, 0.0, 0.0], device=device),
        torch.tensor([
            [1.0, 1.0, 1.0, 0.0, 0.0],
            [2.0, 3.0, 4.0, 0.2, 0.1],
            [-3.0, 2.0, -1.0, -0.5, 0.7]
        ], device=device)
    ]
    
    hfov_values = [45.0, 60.0, 90.0]
    aspect_ratios = [16/9, 4/3, 1.0]
    
    print("\n1. Testing single CCDM conversions:")
    for i, ccdm in enumerate(test_cases[:-1]):
        print(f"\nTest case {i+1}: CCDM = {ccdm}")
        
        transform = convert_ccdm_to_transform(ccdm)
        ccdm_recovered = transform_to_ccdm(transform)
        
        print(f"  Default params (HFOV=45°, aspect=16/9):")
        print(f"  Original CCDM:  {ccdm}")
        print(f"  Recovered CCDM: {ccdm_recovered}")
        print(f"  Error:          {torch.norm(ccdm - ccdm_recovered).item():.6f}")
    
    print("\n2. Testing with different HFOV and aspect ratios:")
    test_ccdm = test_cases[2]
    
    for hfov in hfov_values:
        for aspect in aspect_ratios:
            transform = convert_ccdm_to_transform(test_ccdm, hfov_deg=hfov, aspect=aspect)
            ccdm_recovered = transform_to_ccdm(transform, hfov_deg=hfov, aspect=aspect)
            
            print(f"  HFOV={hfov}°, aspect={aspect}:")
            print(f"    Original CCDM:  {test_ccdm}")
            print(f"    Recovered CCDM: {ccdm_recovered}")
            print(f"    Error:          {torch.norm(test_ccdm - ccdm_recovered).item():.6f}")
    
    print("\n3. Testing batch conversion:")
    batch_ccdm = test_cases[-1]
    batch_transform = convert_ccdm_to_transform(batch_ccdm)
    batch_ccdm_recovered = transform_to_ccdm(batch_transform)
    
    print(f"  Batch size: {batch_ccdm.shape[0]}")
    print(f"  Original CCDM shape:  {batch_ccdm.shape}")
    print(f"  Transform shape:      {batch_transform.shape}")
    print(f"  Recovered CCDM shape: {batch_ccdm_recovered.shape}")
    
    for i in range(batch_ccdm.shape[0]):
        error = torch.norm(batch_ccdm[i] - batch_ccdm_recovered[i]).item()
        print(f"  Batch item {i}:")
        print(f"    Original:  {batch_ccdm[i]}")
        print(f"    Recovered: {batch_ccdm_recovered[i]}")
        print(f"    Error:     {error:.6f}")
    
    print("\n4. Testing NumPy arrays:")
    numpy_test_cases = [
        np.array([2.0, 1.5, 3.0, 0.25, -0.15], dtype=np.float32),
        
        np.array([
            [1.5, 2.5, 3.5, 0.1, 0.2],
            [-2.0, 1.0, 4.0, -0.3, 0.4],
            [0.5, -1.5, 2.0, 0.6, -0.5]
        ], dtype=np.float32)
    ]
    
    np_ccdm = numpy_test_cases[0]
    print(f"\n  Single NumPy CCDM array: {np_ccdm}")
    print(f"  Input type: {type(np_ccdm)}")
    
    np_transform = convert_ccdm_to_transform(np_ccdm)
    print(f"  Transform type: {type(np_transform)}")
    print(f"  Transform shape: {np_transform.shape}")
    
    np_ccdm_recovered = transform_to_ccdm(np_transform)
    print(f"  Recovered CCDM type: {type(np_ccdm_recovered)}")
    
    np_ccdm_recovered_numpy = np_ccdm_recovered.cpu().numpy()
    error = np.linalg.norm(np_ccdm - np_ccdm_recovered_numpy)
    
    print(f"  Original NumPy CCDM:  {np_ccdm}")
    print(f"  Recovered CCDM:       {np_ccdm_recovered_numpy}")
    print(f"  Error:                {error:.6f}")
    
    np_batch_ccdm = numpy_test_cases[1]
    print(f"\n  Batch NumPy CCDM array: shape={np_batch_ccdm.shape}")
    
    np_batch_transform = convert_ccdm_to_transform(np_batch_ccdm)
    np_batch_ccdm_recovered = transform_to_ccdm(np_batch_transform)
    
    print(f"  Input type: {type(np_batch_ccdm)}")
    print(f"  Transform type: {type(np_batch_transform)}")
    print(f"  Recovered CCDM type: {type(np_batch_ccdm_recovered)}")
    
    np_batch_ccdm_recovered_numpy = np_batch_ccdm_recovered.cpu().numpy()
    
    for i in range(np_batch_ccdm.shape[0]):
        error = np.linalg.norm(np_batch_ccdm[i] - np_batch_ccdm_recovered_numpy[i])
        print(f"  Batch item {i}:")
        print(f"    Original:  {np_batch_ccdm[i]}")
        print(f"    Recovered: {np_batch_ccdm_recovered_numpy[i]}")
        print(f"    Error:     {error:.6f}")
    
    print(f"\n  Testing mixed operations (NumPy → PyTorch → NumPy):")
    
    np_mixed_ccdm = np.array([1.0, 2.0, 3.0, 0.3, -0.2], dtype=np.float32)
    print(f"  Original NumPy CCDM: {np_mixed_ccdm}")
    
    torch_transform = convert_ccdm_to_transform(np_mixed_ccdm)
    print(f"  Transform type: {type(torch_transform)}")
    
    np_transform = torch_transform.cpu().numpy()
    print(f"  NumPy transform shape: {np_transform.shape}")
    
    torch_ccdm_recovered = transform_to_ccdm(np_transform)
    np_ccdm_recovered = torch_ccdm_recovered.cpu().numpy()
    
    error = np.linalg.norm(np_mixed_ccdm - np_ccdm_recovered)
    print(f"  Original NumPy CCDM:  {np_mixed_ccdm}")
    print(f"  Recovered NumPy CCDM: {np_ccdm_recovered}")
    print(f"  Error:                {error:.6f}")