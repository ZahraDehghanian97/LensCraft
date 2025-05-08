import torch
import numpy as np
import matplotlib.pyplot as plt

try:
    from data.ccdm.utils import convert_ccdm_to_transform, transform_to_ccdm
except:
    import sys
    sys.path.append('/home/arash/abolghasemi/LensCraft/src')
    from data.ccdm.utils import convert_ccdm_to_transform, transform_to_ccdm


def visualize_camera_transforms(ccdms, transforms, titles, subject_positions=None):
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
        
        if subject_positions is not None and i < len(subject_positions) and subject_positions[i] is not None:
            subject_pos = subject_positions[i].cpu().numpy() if torch.is_tensor(subject_positions[i]) else subject_positions[i]
            ax.scatter([subject_pos[0]], [subject_pos[1]], [subject_pos[2]], color='purple', s=100, marker='s')
            
            ax.plot([position[0], subject_pos[0]], 
                    [position[1], subject_pos[1]], 
                    [position[2], subject_pos[2]], 
                    'k--', alpha=0.5)
        
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
    
    subject_positions = [
        None,  # Default (origin)
        torch.tensor([0.0, 0.0, 0.0], device=device),  # Explicit origin
        torch.tensor([1.0, 1.0, 1.0], device=device),  # Non-origin subject
        torch.tensor([-2.0, 0.5, 1.0], device=device),  # Another position
    ]
    
    batch_subject_positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [-1.0, 0.5, 2.0]
    ], device=device)
    
    hfov_values = [45.0, 60.0, 90.0]
    aspect_ratios = [16/9, 4/3, 1.0]
    
    print("\n1. Testing single CCDM conversions:")
    for i, ccdm in enumerate(test_cases[:-1]):
        print(f"\nTest case {i+1}: CCDM = {ccdm}")
        
        transform = convert_ccdm_to_transform(ccdm)
        ccdm_recovered = transform_to_ccdm(transform)
        
        print(f"  Default params (HFOV=45°, aspect=16/9, subject=None):")
        print(f"  Original CCDM:  {ccdm}")
        print(f"  Recovered CCDM: {ccdm_recovered}")
        print(f"  Error:          {torch.norm(ccdm - ccdm_recovered).item():.6f}")
        
        if i < len(subject_positions) and subject_positions[i] is not None:
            transform_with_subject = convert_ccdm_to_transform(ccdm, subject_positions[i])
            ccdm_recovered_with_subject = transform_to_ccdm(transform_with_subject, subject_positions[i])
            
            print(f"  With subject position {subject_positions[i]}:")
            print(f"  Original CCDM:  {ccdm}")
            print(f"  Recovered CCDM: {ccdm_recovered_with_subject}")
            print(f"  Error:          {torch.norm(ccdm - ccdm_recovered_with_subject).item():.6f}")
    
    print("\n2. Testing with different HFOV and aspect ratios:")
    test_ccdm = test_cases[2]
    test_subject = subject_positions[2]  # Non-origin subject
    
    for hfov in hfov_values:
        for aspect in aspect_ratios:
            transform = convert_ccdm_to_transform(test_ccdm, hfov_deg=hfov, aspect=aspect)
            ccdm_recovered = transform_to_ccdm(transform, hfov_deg=hfov, aspect=aspect)
            
            print(f"  HFOV={hfov}°, aspect={aspect}, default subject:")
            print(f"    Original CCDM:  {test_ccdm}")
            print(f"    Recovered CCDM: {ccdm_recovered}")
            print(f"    Error:          {torch.norm(test_ccdm - ccdm_recovered).item():.6f}")
            
            transform_with_subject = convert_ccdm_to_transform(test_ccdm, test_subject, 
                                                             hfov_deg=hfov, aspect=aspect)
            ccdm_recovered_with_subject = transform_to_ccdm(transform_with_subject, test_subject,
                                                          hfov_deg=hfov, aspect=aspect)
            
            print(f"  HFOV={hfov}°, aspect={aspect}, subject={test_subject}:")
            print(f"    Original CCDM:  {test_ccdm}")
            print(f"    Recovered CCDM: {ccdm_recovered_with_subject}")
            print(f"    Error:          {torch.norm(test_ccdm - ccdm_recovered_with_subject).item():.6f}")
    
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
    
    batch_transform_with_subject = convert_ccdm_to_transform(batch_ccdm, batch_subject_positions)
    batch_ccdm_recovered_with_subject = transform_to_ccdm(batch_transform_with_subject, batch_subject_positions)
    
    print(f"\n  With batch subject positions:")
    print(f"  Batch size: {batch_ccdm.shape[0]}")
    print(f"  Subject positions shape: {batch_subject_positions.shape}")
    print(f"  Transform shape:      {batch_transform_with_subject.shape}")
    print(f"  Recovered CCDM shape: {batch_ccdm_recovered_with_subject.shape}")
    
    for i in range(batch_ccdm.shape[0]):
        error = torch.norm(batch_ccdm[i] - batch_ccdm_recovered_with_subject[i]).item()
        print(f"  Batch item {i} (subject={batch_subject_positions[i]}):")
        print(f"    Original:  {batch_ccdm[i]}")
        print(f"    Recovered: {batch_ccdm_recovered_with_subject[i]}")
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
    
    numpy_subject_positions = [
        np.array([0.5, -0.5, 1.0], dtype=np.float32),
        np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [-1.0, 0.5, 0.5]
        ], dtype=np.float32)
    ]
    
    np_ccdm = numpy_test_cases[0]
    np_subject = numpy_subject_positions[0]
    
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
    
    np_transform_with_subject = convert_ccdm_to_transform(np_ccdm, np_subject)
    np_ccdm_recovered_with_subject = transform_to_ccdm(np_transform_with_subject, np_subject)
    
    np_ccdm_recovered_with_subject_numpy = np_ccdm_recovered_with_subject.cpu().numpy()
    error_with_subject = np.linalg.norm(np_ccdm - np_ccdm_recovered_with_subject_numpy)
    
    print(f"\n  With NumPy subject position: {np_subject}")
    print(f"  Original NumPy CCDM:  {np_ccdm}")
    print(f"  Recovered CCDM:       {np_ccdm_recovered_with_subject_numpy}")
    print(f"  Error:                {error_with_subject:.6f}")
    
    np_batch_ccdm = numpy_test_cases[1]
    np_batch_subject = numpy_subject_positions[1]
    
    print(f"\n  Batch NumPy CCDM array: shape={np_batch_ccdm.shape}")
    print(f"  Batch NumPy subject positions: shape={np_batch_subject.shape}")
    
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
    
    np_batch_transform_with_subject = convert_ccdm_to_transform(np_batch_ccdm, np_batch_subject)
    np_batch_ccdm_recovered_with_subject = transform_to_ccdm(np_batch_transform_with_subject, np_batch_subject)
    
    np_batch_ccdm_recovered_with_subject_numpy = np_batch_ccdm_recovered_with_subject.cpu().numpy()
    
    print(f"\n  With batch NumPy subject positions:")
    for i in range(np_batch_ccdm.shape[0]):
        error = np.linalg.norm(np_batch_ccdm[i] - np_batch_ccdm_recovered_with_subject_numpy[i])
        print(f"  Batch item {i} (subject={np_batch_subject[i]}):")
        print(f"    Original:  {np_batch_ccdm[i]}")
        print(f"    Recovered: {np_batch_ccdm_recovered_with_subject_numpy[i]}")
        print(f"    Error:     {error:.6f}")
    
    print(f"\n  Testing mixed operations (NumPy → PyTorch → NumPy):")
    
    np_mixed_ccdm = np.array([1.0, 2.0, 3.0, 0.3, -0.2], dtype=np.float32)
    np_mixed_subject = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    
    print(f"  Original NumPy CCDM: {np_mixed_ccdm}")
    print(f"  NumPy subject position: {np_mixed_subject}")
    
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
    
    torch_transform_with_subject = convert_ccdm_to_transform(np_mixed_ccdm, np_mixed_subject)
    np_transform_with_subject = torch_transform_with_subject.cpu().numpy()
    
    torch_ccdm_recovered_with_subject = transform_to_ccdm(np_transform_with_subject, np_mixed_subject)
    np_ccdm_recovered_with_subject = torch_ccdm_recovered_with_subject.cpu().numpy()
    
    error_with_subject = np.linalg.norm(np_mixed_ccdm - np_ccdm_recovered_with_subject)
    print(f"\n  With NumPy subject position:")
    print(f"  Original NumPy CCDM:  {np_mixed_ccdm}")
    print(f"  Recovered NumPy CCDM: {np_ccdm_recovered_with_subject}")
    print(f"  Error:                {error_with_subject:.6f}")
    
    print("\n5. Visualizing camera transforms:")
    vis_ccdms = []
    vis_transforms = []
    vis_titles = []
    vis_subjects = []
    
    for i, ccdm in enumerate(test_cases[:-1]):
        subject = None if i >= len(subject_positions) else subject_positions[i]
        transform = convert_ccdm_to_transform(ccdm, subject)
        
        vis_ccdms.append(ccdm)
        vis_transforms.append(transform)
        vis_titles.append(f"PyTorch Camera {i+1}" + (f" (with subject)" if subject is not None else ""))
        vis_subjects.append(subject)
    
    for i in range(batch_ccdm.shape[0]):
        single_ccdm = batch_ccdm[i]
        single_subject = batch_subject_positions[i]
        
        transform = convert_ccdm_to_transform(single_ccdm, single_subject)
        
        vis_ccdms.append(single_ccdm)
        vis_transforms.append(transform)
        vis_titles.append(f"PyTorch Batch {i+1} (with subject)")
        vis_subjects.append(single_subject)
    
    np_ccdm = numpy_test_cases[0]
    np_subject = numpy_subject_positions[0]
    np_transform = convert_ccdm_to_transform(np_ccdm, np_subject)
    
    vis_ccdms.append(torch.tensor(np_ccdm, device=device))
    vis_transforms.append(np_transform)
    vis_titles.append(f"NumPy Camera (with subject)")
    vis_subjects.append(torch.tensor(np_subject, device=device))
    
    np_batch_ccdm = numpy_test_cases[1]
    np_batch_subject = numpy_subject_positions[1]
    
    for j in range(np_batch_ccdm.shape[0]):
        single_np_ccdm = np_batch_ccdm[j]
        single_np_subject = np_batch_subject[j]
        
        single_np_transform = convert_ccdm_to_transform(single_np_ccdm, single_np_subject)
        
        vis_ccdms.append(torch.tensor(single_np_ccdm, device=device))
        vis_transforms.append(single_np_transform)
        vis_titles.append(f"NumPy Batch {j+1} (with subject)")
        vis_subjects.append(torch.tensor(single_np_subject, device=device))
    
    visualize_camera_transforms(vis_ccdms, vis_transforms, vis_titles, vis_subjects)
    print("  Visualization saved to 'camera_transforms.png'")
    
    print("\nAll tests completed successfully!")
