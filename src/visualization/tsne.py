import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Union
import torch
from sklearn.manifold import TSNE
import seaborn as sns

def tSNE_visualize_embeddings(
    embeddings_dict: Dict[str, Union[np.ndarray, torch.Tensor]],
    random_state: int = 42,
    perplexity: int = 30,
    max_points: int = 100,
    figsize: tuple = (12, 8),
    title: str = 'Embedding Visualization using t-SNE',
    add_density: bool = True,
    point_size: int = 50,
    point_alpha: float = 0.6,
    verbose: bool = True,
    save_path: str = "./test_results/embeddings_tSNE.png"
) -> plt.Figure:
    """Visualize embeddings from different models in 2D using t-SNE."""
    # Process embeddings
    all_embeddings, labels, model_counts = [], [], {}
    if verbose: print("Processing embeddings...")
    
    for model_name, embeddings in embeddings_dict.items():
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        elif isinstance(embeddings, list) and all(isinstance(e, torch.Tensor) for e in embeddings):
            # Handle case where embeddings is a list of tensors
            embeddings = torch.cat(embeddings, dim=0).detach().cpu().numpy()
        else:
            embeddings = np.array(embeddings)
        
        if len(embeddings) > max_points:
            embeddings = embeddings[np.random.RandomState(random_state).choice(
                len(embeddings), max_points, replace=False)]
        
        all_embeddings.append(embeddings)
        labels.extend([model_name] * len(embeddings))
        model_counts[model_name] = len(embeddings)
    
    # Run t-SNE
    if verbose: print(f"Performing t-SNE on {len(all_embeddings := np.vstack(all_embeddings))} points...")
    embeddings_2d = TSNE(n_components=2, random_state=random_state,
                         perplexity=min(perplexity, len(all_embeddings) - 1),
                         n_jobs=-1).fit_transform(all_embeddings)
    
    # Plot results
    fig, ax = plt.subplots(figsize=figsize)
    palette = sns.color_palette("husl", len(unique_labels := list(embeddings_dict.keys())))
    
    for label, color in zip(unique_labels, palette):
        mask = np.array(labels) == label
        points = embeddings_2d[mask]
        
        # Plot points
        ax.scatter(points[:, 0], points[:, 1],
                  label=f"{label} (n={model_counts[label]})",
                  alpha=point_alpha, s=point_size, color=color)
        
        # Add density contours if possible
        if add_density and sum(mask) >= 5:
            try:
                sns.kdeplot(x=points[:, 0], y=points[:, 1], levels=3,
                           color=color, alpha=0.3, linewidths=1, ax=ax)
            except Exception as e:
                if verbose: print(f"Could not create density plot for {label}: {e}")
    
    # Finalize plot
    ax.set_title(title)
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if verbose: print(f"Figure saved to {save_path}")
    
    return fig