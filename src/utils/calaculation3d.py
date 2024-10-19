import torch


def rotation_6d_to_matrix(rotation_6d):
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    """
    x_raw = rotation_6d[:3]
    y_raw = rotation_6d[3:]

    x = x_raw / torch.norm(x_raw, dim=-1, keepdim=True)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / torch.norm(z, dim=-1, keepdim=True)
    y = torch.cross(z, x, dim=-1)

    return torch.stack([x, y, z], dim=-1)


def euler_from_matrix(matrix):
    """
    Convert 3x3 rotation matrix to Euler angles (in radians).
    """
    sy = torch.sqrt(matrix[0, 0] * matrix[0, 0] + matrix[1, 0] * matrix[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = torch.atan2(matrix[2, 1], matrix[2, 2])
        y = torch.atan2(-matrix[2, 0], sy)
        z = torch.atan2(matrix[1, 0], matrix[0, 0])
    else:
        x = torch.atan2(-matrix[1, 2], matrix[1, 1])
        y = torch.atan2(-matrix[2, 0], sy)
        z = 0

    return torch.tensor([x, y, z])
