1. Trajectory Data:
   - `traj_filename`: A string, the name of the trajectory file. This uniquely identifies each trajectory sequence in the dataset.
   - `traj_feat`: A tensor of shape [num_traj_features, num_cams], representing trajectory features for num_cams time steps, with num_traj_features per step.
     - Each column represents a time step, with 9 features: 6 for rotation (using 6D continuous rotation representation) and 3 for translation.
     - This format allows for efficient processing of sequential data.
   - `padding_mask`: A tensor of shape [num_cams], a binary mask indicating valid (1) and padded (0) time steps.
     - Enables handling of variable-length sequences within a fixed-size tensor.
   - `intrinsics`: A numpy array of shape (num_intrinsics_params,), camera intrinsic parameters.
     - Contains focal length (fx, fy) and principal point (cx, cy) for camera calibration.

2. Character Data:
   - `char_filename`: A string, the name of the character data file. Identifies the character data associated with the trajectory.
   - `char_feat`: A tensor of shape [num_char_features, num_cams], representing character features for num_cams time steps, with num_char_features per step.
     - Processed character features, likely normalized or transformed for model input.
   - `char_raw`: A dictionary containing:
     - `char_raw_feat`: A tensor of shape [num_char_features, num_cams], raw character features.
     - `char_centers`: A tensor of shape [num_char_features, num_cams], character center positions.
     - Provides both processed and raw data for flexibility in downstream tasks.
   - `char_padding_mask`: A tensor of shape [num_cams], a binary mask for character data.
     - Aligns with the trajectory padding mask for consistent processing.

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

Key observations:
1. The dataset aligns trajectory, character, and caption data over num_cams time steps.
2. It includes both processed features (e.g., `traj_feat`, `char_feat`) and raw data (e.g., `char_raw`, `caption_raw`).
3. CLIP embeddings are used for caption representation, with a maximum of max_feat_length tokens.
4. Padding masks are provided for each modality, allowing for variable-length sequences up to num_cams steps.
