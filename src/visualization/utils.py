import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Union, List, Tuple
import torch
from sklearn.manifold import TSNE
import seaborn as sns

def _prepare_embeddings(
    embeddings: Union[np.ndarray, torch.Tensor, List[torch.Tensor]],
    max_points: int = None,
    random_state: int = 42
) -> np.ndarray:
    """Convert embeddings to numpy array and subsample if needed."""
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    elif isinstance(embeddings, list) and all(isinstance(e, torch.Tensor) for e in embeddings):
        embeddings = torch.cat(embeddings, dim=0).detach().cpu().numpy()
    else:
        embeddings = np.array(embeddings)
    
    if max_points and len(embeddings) > max_points:
        indices = np.random.RandomState(random_state).choice(
            len(embeddings), max_points, replace=False)
        embeddings = embeddings[indices]
        return embeddings, indices
    
    return embeddings, None

def _compute_tsne(
    embeddings: np.ndarray,
    random_state: int = 42,
    perplexity: int = 30,
    verbose: bool = True
) -> np.ndarray:
    """Perform t-SNE dimensionality reduction."""
    if verbose:
        print(f"Performing t-SNE on {len(embeddings)} points...")
    
    adjusted_perplexity = min(perplexity, len(embeddings) - 1)
    
    return TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=adjusted_perplexity,
        n_jobs=-1
    ).fit_transform(embeddings)

def _setup_plot(
    figsize: tuple = (12, 8),
    title: str = 'Embedding Visualization using t-SNE'
) -> Tuple[plt.Figure, plt.Axes]:
    """Set up the basic plot figure and axes."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    return fig, ax

def _save_plot(
    fig: plt.Figure,
    save_path: str = None,
    verbose: bool = True
) -> None:
    """Save the plot if a path is provided."""
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"Figure saved to {save_path}")

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
    all_embeddings, labels, model_counts = [], [], {}
    if verbose:
        print("Processing embeddings...")
    
    for model_name, embeddings in embeddings_dict.items():
        prepared_embeddings, _ = _prepare_embeddings(embeddings, max_points, random_state)
        
        all_embeddings.append(prepared_embeddings)
        labels.extend([model_name] * len(prepared_embeddings))
        model_counts[model_name] = len(prepared_embeddings)
    
    all_embeddings = np.vstack(all_embeddings)
    
    embeddings_2d = _compute_tsne(all_embeddings, random_state, perplexity, verbose)
    
    fig, ax = _setup_plot(figsize, title)
    
    unique_labels = list(embeddings_dict.keys())
    palette = sns.color_palette("husl", len(unique_labels))
    
    for label, color in zip(unique_labels, palette):
        mask = np.array(labels) == label
        points = embeddings_2d[mask]
        
        ax.scatter(
            points[:, 0], points[:, 1],
            label=f"{label} (n={model_counts[label]})",
            alpha=point_alpha, s=point_size, color=color
        )
        
        if add_density and sum(mask) >= 5:
            try:
                sns.kdeplot(
                    x=points[:, 0], y=points[:, 1], 
                    levels=3, color=color, alpha=0.3, 
                    linewidths=1, ax=ax
                )
            except Exception as e:
                if verbose:
                    print(f"Could not create density plot for {label}: {e}")
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    _save_plot(fig, save_path, verbose)
    
    return fig


def tSNE_visualize_embeddings_by_class_type(
    caption_embeddings: Union[np.ndarray, torch.Tensor],
    encoder_embeddings: Union[np.ndarray, torch.Tensor],
    class_types: List[str],
    random_state: int = 42,
    perplexity: int = 30,
    max_points: int = 1000,
    figsize: tuple = (14, 10),
    title: str = 'Embedding Visualization using t-SNE (Colored by Movement Type)',
    point_size: int = 60,
    point_alpha: float = 0.7,
    verbose: bool = True,
    save_path: str = "./embeddings_tSNE_by_movement_type.png"
) -> plt.Figure:
    """
    Visualize caption and encoder embeddings in 2D using t-SNE, colored by class type.
    """
    prepared_caption, indices = _prepare_embeddings(caption_embeddings, max_points, random_state)
    prepared_encoder, _ = _prepare_embeddings(encoder_embeddings, max_points=None, random_state=random_state)
    
    if indices is not None:
        class_types = [class_types[i] for i in indices]
    
    assert len(prepared_caption) == len(class_types), (
        f"Number of caption embeddings ({len(prepared_caption)}) doesn't match "
        f"number of class types ({len(class_types)})"
    )
    assert len(prepared_encoder) == len(class_types), (
        f"Number of encoder embeddings ({len(prepared_encoder)}) doesn't match "
        f"number of class types ({len(class_types)})"
    )
    
    all_embeddings = np.vstack([prepared_caption, prepared_encoder])
    
    embeddings_2d = _compute_tsne(all_embeddings, random_state, perplexity, verbose)
    
    caption_points = embeddings_2d[:len(prepared_caption)]
    encoder_points = embeddings_2d[len(prepared_caption):]
    
    fig, ax = _setup_plot(figsize, title)
    
    unique_class_types = sorted(set(class_types))
    palette = sns.color_palette("hsv", len(unique_class_types))
    class_type_to_color = {mt: palette[i] for i, mt in enumerate(unique_class_types)}
    
    for mt_idx, mt in enumerate(unique_class_types):
        indices = [i for i, type_val in enumerate(class_types) if type_val == mt]
        
        if indices:
            ax.scatter(
                caption_points[indices, 0], 
                caption_points[indices, 1],
                color=class_type_to_color[mt], 
                alpha=point_alpha, 
                s=point_size, 
                marker='o',
                label=f"Caption - {mt}" if mt_idx == 0 else "_nolegend_"
            )
            
            pastel_color = [c * 0.6 + 0.4 for c in class_type_to_color[mt]]
            ax.scatter(
                encoder_points[indices, 0], 
                encoder_points[indices, 1],
                color=pastel_color, 
                alpha=point_alpha, 
                s=point_size, 
                marker='s',  # Use squares for encoder
                label=f"Encoder - {mt}" if mt_idx == 0 else "_nolegend_"
            )
    
    handles = []
    labels = []
    
    for mt in unique_class_types:
        handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=class_type_to_color[mt], markersize=10))
        labels.append(mt)
    
    handles.append(plt.Line2D([0], [0], marker='', color='w'))
    labels.append("")
    
    handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor='black', markersize=10))
    labels.append("Caption (Vivid)")
    
    handles.append(plt.Line2D([0], [0], marker='s', color='w', 
                             markerfacecolor=[0.6, 0.6, 0.6], markersize=10))
    labels.append("Encoder (Pastel)")
    
    ax.legend(handles, labels, title="Class Types", 
             bbox_to_anchor=(1.05, 1), loc='upper left', 
             frameon=True, framealpha=0.8)
    
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    _save_plot(fig, save_path, verbose)
    
    return fig