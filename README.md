# LenseCraft

This project implements an autoencoder model for camera trajectories using a multi-task architecture. The model learns to reconstruct camera movements based on subject information and initial trajectory data. In addition to producing accurate trajectory reconstructions, it generates embeddings for movement types, easing functions, camera angles, and shot types. These generated embeddings are designed to be aligned with CLIP (Contrastive Language-Image Pre-training) embeddings, allowing for better integration with language-based interfaces and multi-modal applications.

## Model Architecture

```mermaid
graph TD
    subgraph Input
        A[Camera Trajectory<br>input_dim] --> B[Apply Noise]
        A --> C[Apply Mask]
        B & C --> D[Noisy & Masked<br>Trajectory]
        S[Subject Info<br>subject_dim] --> SP[Subject Projection<br>Linear: subject_dim → latent_dim]
    end

    subgraph Encoder["Encoder (TransformerEncoder)"]
        E[Input Projection<br>Linear: input_dim → latent_dim]
        F[Positional Encoding]
        G[Transformer Encoder Layers<br>num_encoder_layers, nhead]
        H1[Movement Query Token]
        H2[Easing Query Token]
        H3[Camera Angle Query Token]
        H4[Shot Type Query Token]
        M1[Encoder Memory<br>latent_dim per token]
    end

    subgraph LatentSpace["Latent Space Processing"]
        I1[Movement Embedding<br>latent_dim]
        I2[Easing Embedding<br>latent_dim]
        I3[Camera Angle Embedding<br>latent_dim]
        I4[Shot Type Embedding<br>latent_dim]
        J[Latent Merger<br>Linear: latent_dim*4 → latent_dim]
        K[Merged Latent<br>latent_dim]
    end

    subgraph AutoregressiveDecoding
        subgraph Decoder["Decoder (TransformerDecoder)"]
            L[Embedding Layer<br>Linear: input_dim → latent_dim]
            M[Positional Encoding]
            N[Transformer Decoder Layers<br>num_decoder_layers, nhead]
            O[Output Projection<br>Linear: latent_dim → input_dim]
        end
        P[Initial Zero Input]
        Q[Teacher Forcing]
        R1[Output t]
        M2[Decoder Memory<br>latent_dim, seq_length]
    end

    subgraph Output
        R[Reconstructed Trajectory<br>input_dim, seq_length]
    end

    subgraph Losses
        S1[Reconstruction Loss<br>MSE]
        S2[CLIP Movement Loss<br>1 - CosineSimilarity]
        S3[CLIP Easing Loss<br>1 - CosineSimilarity]
        S4[CLIP Camera Angle Loss<br>1 - CosineSimilarity]
        S5[CLIP Shot Type Loss<br>1 - CosineSimilarity]
        T[Total Loss<br>Sum of all losses]
    end

    S --> SP
    SP --> G
    D --> E --> F --> G
    H1 & H2 & H3 & H4 --> G
    G --> M1
    M1 --> I1 & I2 & I3 & I4
    I1 & I2 & I3 & I4 --> J --> K
    K --> M2
    M2 --> N
    P --> L
    L --> M --> N
    N --> O --> R1
    R1 --> L
    Q -.-> L
    R1 --> R
    R --> S1
    I1 --> S2
    I2 --> S3
    I3 --> S4
    I4 --> S5
    S1 & S2 & S3 & S4 & S5 --> T
    SP --> L

    classDef subgraphStyle fill:#f0f0f0,stroke:#333,stroke-width:2px;
    class Input,Encoder,LatentSpace,AutoregressiveDecoding,Output,Losses subgraphStyle;

    style E fill:#f9e79f
    style G fill:#f9e79f
    style J fill:#aed6f1
    style L fill:#d7bde2
    style N fill:#d7bde2
    style O fill:#d7bde2
    style M1 fill:#fad7a0
    style M2 fill:#fad7a0
    style R1 fill:#82e0aa
    style S1 fill:#f5b7b1
    style S2 fill:#f5b7b1
    style S3 fill:#f5b7b1
    style S4 fill:#f5b7b1
    style S5 fill:#f5b7b1
    style T fill:#f5b7b1
    style SP fill:#f9e79f

```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/ZahraDehghanian97/LenseCraft.git
   cd LenseCraft
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Dataset

The model expects a JSON file containing simulation data. Each simulation should include:
- Camera frames (30 frames per simulation)
- Subject information (position, size, rotation)
- Instructions (camera movement, easing, initial camera angle, initial shot type)

The `SimulationDataset` class in `data/simulation/dataset.py` handles data loading and preprocessing.

To download the dataset, use the following link:
```
https://drive.google.com/uc?id=1VT2XfBj9LFWLUBjv65dzC4bVzH0zdNDU
```
Make sure to place the downloaded dataset file in the appropriate location within your project structure.

## Usage

To train the model, run the `main.py` script with the desired arguments:
```
python main.py --data path/to/your/dataset.json --batch_size 32 --epochs 20 --lr 0.0001
```

For a full list of available arguments, run:
```
python main.py --help
```

## Training

The training process includes:
1. Data augmentation (masking and adding noise to input trajectories)
2. Teacher forcing for the autoregressive decoder
3. Gradual increase in task difficulty (noise reduction and mask ratio increase)
4. Multi-task learning (trajectory reconstruction and CLIP embedding prediction)

## Evaluation

The model is evaluated on a validation set during training. The evaluation metrics include:
1. Trajectory reconstruction loss (MSE for positions, circular distance for angles)
2. CLIP embedding similarity loss for movement types, easing functions, camera angles, and shot types

For any questions or issues, please open an issue on the GitHub repository or contact the project maintainers.