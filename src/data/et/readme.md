# ET Dataset

## Detail

The dataset combines trajectory, character, and caption data for motion capture or animation scenes. Each sample contains:

1. Trajectory data: Camera motion represented by rotation (6D continuous representation) and translation (3D) features.
2. Character data: Character position (3D coordinates) and possibly pose information.
3. Caption data: Textual descriptions of the scene or camera movement, encoded using CLIP embeddings (512-dimensional vectors).

## Key observations

1. The dataset aligns trajectory, character, and caption data over `num_cams` (300) time steps.
2. It includes both processed features (e.g., `traj_feat`, `char_feat`) and raw data (e.g., `char_raw`, `caption_raw`) as PyTorch tensors.
3. CLIP embeddings are used for caption representation, with a maximum of `max_feat_length` (77) tokens.
4. Padding masks are provided for each modality as binary tensors, allowing for variable-length sequences up to `num_cams` steps.

## More Detail

1. Trajectory Data:
   - `traj_filename`: A string, the name of the trajectory file. This uniquely identifies each trajectory sequence in the dataset.
   - `traj_feat`: A tensor of shape [num_traj_features, num_cams], representing trajectory features for num_cams time steps, with num_traj_features per step.
     - Each column represents a time step, with 9 features: 6 for rotation (using 6D continuous rotation representation) and 3 for translation.
     - This format allows for efficient processing of sequential data.
   - `padding_mask`: A tensor of shape [num_cams], a binary mask indicating valid (1) and padded (0) time steps.
     - Enables handling of variable-length sequences within a fixed-size tensor.
   - `intrinsics`: A numpy array of shape (4,), camera intrinsic parameters.
     - Contains [fx, fy, cx, cy] focal length and principal point for camera calibration.

2. Character Data:
   - `char_filename`: A string, the name of the character data file. Identifies the character data associated with the trajectory.
   - `num_char_features`: int = 3 (x, y, z coordinates of character's center)
   - `char_feat`: A tensor of shape [num_char_features, num_cams], representing character features for num_cams time steps.
     - Each column represents a time step, with 3 features for the character's position (x, y, z).
     - If `velocity` is True, the first row contains the initial position, and subsequent rows contain velocity (change in position).
     - Processed character features, normalized and potentially transformed for model input.
   - `char_raw`: A dictionary containing:
     - `char_raw_feat`: A tensor of shape [num_char_features, num_cams], raw character features.
       - Contains the original, unprocessed character positions for each time step.
     - `char_centers`: A tensor of shape [num_char_features, num_cams], character center positions.
       - Identical to `char_raw_feat`.
     - If `load_vertices` is True:
       - `char_vertices`: A tensor of shape [num_cams, num_vertices, 3], representing vertex positions.
       - `char_faces`: A tensor of shape [num_cams, num_faces, 3], representing face indices.
     - Provides both processed and raw data for flexibility in downstream tasks.
   - `char_padding_mask`: A tensor of shape [num_cams], a binary mask for character data.
     - Aligns with the trajectory padding mask for consistent processing.
   - Character features are processed as follows:
     - Raw features are loaded from `.npy` files.
     - Features are padded to `num_cams` length if necessary.
     - Velocity information is computed if the `velocity` flag is True:
       - First row remains the initial position.
       - Subsequent rows contain the difference between consecutive positions.
     - Features are standardized using `norm_mean` and `norm_std` if `standardize` is True:
       - If `velocity` is True and `norm_mean`/`norm_std` have 6 elements, the first 3 are used for the initial position and the last 3 for velocities.
       - Otherwise, all features are normalized using the same mean and standard deviation.
     - The data can be reshaped for sequential or non-sequential processing based on the `sequential` flag:
       - If `sequential` is True, the final shape is [num_char_features, num_cams].
       - If `sequential` is False, the final shape is [num_char_features * num_cams].

3. Caption Data:
   - `caption_filename`: A string, the name of the caption file. Links textual data to the corresponding trajectory and character data.
   - `caption_feat`: A tensor of shape [num_caption_features, max_feat_length], CLIP embeddings for the caption (num_caption_features-dimensional embeddings for max_feat_length tokens).
     - Utilizes CLIP's text encoder for semantic representation of captions.
   - `caption_raw`: A dictionary containing:
     - `caption`: A string, the actual text caption describing the scene or camera movement.
     - `segments`: A tensor of shape [num_cams], segment labels for each time step.
       - Provides fine-grained labeling of trajectory segments, useful for action recognition or segmentation tasks.
     - `clip_seq_caption`: A tensor of shape [max_feat_length, num_caption_features], the transposed version of `caption_feat`.
     - `clip_seq_mask`: A tensor of shape [max_feat_length], a mask for valid tokens in the caption.
   - `caption_padding_mask`: A tensor of shape [num_cams], a binary mask for caption data.
     - Ensures alignment with trajectory and character data masks.

## parameters

1. Trajectory Parameters:
   - `num_feats`: int = 9 (6 for rotation, 3 for translation)
   - `num_rawfeats`: int = 12 (original representation before processing)
   - `num_cams`: int = 300 (number of time steps)
   - `standardize`: bool = True (data is normalized)

2. Character Parameters:
   - `num_feats`: int = 3 (x, y, z coordinates of character's center)
   - `sequential`: bool, determined by `diffuser.network.module.cond_sequential`
   - `num_vertices`: int, number of vertices in the character mesh (if `load_vertices` is True)
   - `num_faces`: int, number of faces in the character mesh (if `load_vertices` is True)
   - `load_vertices`: bool, determines whether to load character mesh data
   - `standardize`: bool, whether to standardize the character features
   - `velocity`: bool, whether to compute velocity information

3. Caption Parameters:
   - `num_segments`: int = 27 (sequence divided into 27 segments)
   - `num_feats`: int = 512 (dimensionality of CLIP text embeddings)
   - `max_feat_length`: int = 77 (maximum number of tokens in a caption)

4. Standardization Parameters:
   - `num_interframes`: int = 0 (no interpolated frames)
   - `num_total_frames`: int = 300 (equal to `num_cams`)
   - `norm_mean` and `norm_std`: list, normalization parameters for trajectory data
   - `shift_mean` and `shift_std`: list, additional transformation parameters
   - `norm_mean_h` and `norm_std_h`: list, normalization parameters for character data
   - `velocity`: bool = True (velocity information is included)

5. Dataset Configuration:
   - Uses rot6d (6D rotation representation) for trajectories
   - Combines trajectory, caption, and optionally character data
   - Dataset name format: f"{standardization_name}-t:{trajectory_name}-c:{caption_name}(-h:{char_name})"
   - `dataset_dir`: str = `${data_dir}` (configurable path)

6. Data Processing:
   - CLIP is used for encoding captions, resulting in 512-dimensional embeddings
   - Character data is centered (as indicated by `center_char.yaml`)
   - Trajectory data uses a 6D rotation representation (continuous representation of 3D rotations)

7. Multimodal Integration:
   - The dataset aligns trajectory, character, and caption data over 300 time steps
   - Padding masks are provided for each modality to handle variable-length sequences
   - The `MultimodalDataset` class is used to combine these different data types
