import torch
import math


@torch.jit.script
def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to rotation matrix using JIT compilation."""
    if d6.dim() == 1:
        d6 = d6.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    # Split input into x and y components
    x_raw = d6[:, :3]
    y_raw = d6[:, 3:]

    # Normalize x
    x = torch.nn.functional.normalize(x_raw, dim=1)

    # Compute z as cross product and normalize
    z = torch.nn.functional.normalize(
        torch.cross(x, y_raw, dim=1),
        dim=1
    )

    # Compute y as cross product (already normalized)
    y = torch.cross(z, x, dim=1)

    # Stack the basis vectors
    matrix = torch.stack([x, y, z], dim=2)

    if squeeze_output:
        matrix = matrix.squeeze(0)

    return matrix


@torch.jit.script
def euler_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to euler angles using JIT compilation."""
    if matrix.dim() == 3:
        matrix = matrix.squeeze(0)

    # Extract matrix elements
    m02 = matrix[0, 2]
    abs_m02 = torch.abs(m02)

    # Pre-compute common values
    PI_HALF = torch.tensor(
        math.pi / 2, dtype=matrix.dtype, device=matrix.device)

    # Handle regular case
    y_regular = -torch.asin(torch.clamp(m02, -1.0, 1.0))
    x_regular = torch.atan2(matrix[1, 2], matrix[2, 2])
    z_regular = torch.atan2(matrix[0, 1], matrix[0, 0])

    # Handle gimbal lock cases
    z_lock = torch.zeros_like(m02)
    y_lock = torch.where(m02 < 0, PI_HALF, -PI_HALF)
    x_lock = torch.where(
        m02 < 0,
        z_lock + torch.atan2(matrix[1, 0], matrix[2, 0]),
        -z_lock + torch.atan2(-matrix[1, 0], -matrix[2, 0])
    )

    # Select appropriate values based on m02
    is_lock = (abs_m02 == 1)
    x = torch.where(is_lock, x_lock, x_regular)
    y = torch.where(is_lock, y_lock, y_regular)
    z = torch.where(is_lock, z_lock, z_regular)

    return torch.stack([x, y, z])


def test_rotation_conversions(num_tests: int = 1000):
    """Test rotation conversions with timing."""
    import time

    # Test standard cases
    d6_batch = torch.randn(num_tests, 6, dtype=torch.float32)

    # Warm-up run
    _ = rotation_6d_to_matrix(d6_batch)

    # Timed run
    start = time.perf_counter()
    matrices = rotation_6d_to_matrix(d6_batch)
    eulers = torch.stack([euler_from_matrix(m) for m in matrices])
    end = time.perf_counter()

    print(f"Processed {num_tests} rotations in {end-start:.4f} seconds")
    print(f"Average time per rotation: {(end-start)/num_tests*1000:.4f} ms")

    # Test specific cases
    print("\nTesting specific rotations:")

    # Zero rotation
    d6_zero = torch.zeros(6, dtype=torch.float32)
    matrix_zero = rotation_6d_to_matrix(d6_zero)
    euler_zero = euler_from_matrix(matrix_zero)
    print("\nZero rotation:")
    print("Euler angles (deg):", euler_zero * 180 / math.pi)

    # 90° X rotation
    d6_90x = torch.tensor([1, 0, 0, 0, 0, 1], dtype=torch.float32)
    matrix_90x = rotation_6d_to_matrix(d6_90x)
    euler_90x = euler_from_matrix(matrix_90x)
    print("\n90° X rotation:")
    print("Euler angles (deg):", euler_90x * 180 / math.pi)


if __name__ == "__main__":
    test_rotation_conversions()
