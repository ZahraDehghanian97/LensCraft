import os
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
from typing import Dict, Any, List


def generate_metric_tags(cfg: DictConfig) -> List[str]:
    tags = []
    
    tags.extend(cfg.metrics.validation)
    tags.extend(cfg.metrics.training)
    
    if cfg.metrics.clips.enabled:
        prefixes = []
        if cfg.metrics.clips.include_validation:
            prefixes.append("val_clip_")
        if cfg.metrics.clips.include_training:
            prefixes.append("train_clip_")
            
        for prefix in prefixes:
            tags.extend([f"{prefix}{i}_epoch" for i in range(cfg.metrics.clips.num_clips)])
    
    return tags


def load_tfevents(logdir: str, cfg: DictConfig) -> event_accumulator.EventAccumulator:
    event_acc = event_accumulator.EventAccumulator(
        logdir, 
        size_guidance=cfg.event_accumulator.size_guidance
    )
    event_acc.Reload()
    return event_acc


def extract_scalar_data(
    event_acc: event_accumulator.EventAccumulator,
    metric_tags: List[str]
) -> Dict[str, Dict[str, List[float]]]:
    scalar_data = {}
    available_tags = event_acc.Tags()['scalars']

    for tag in available_tags:
        if tag in metric_tags:
            events = event_acc.Scalars(tag)
            steps = [event.step for event in events]
            values = [event.value for event in events]
            scalar_data[tag] = {'steps': steps, 'values': values}

    return scalar_data


def get_line_style(tag: str, cfg: DictConfig) -> Dict[str, Any]:
    if any(tag.startswith('val_') for tag in ['val_loss', 'val_trajectory', 'val_total']):
        return cfg.line_styles.validation
    elif any(tag.startswith('train_') for tag in ['train_loss', 'train_trajectory', 'train_total']):
        return cfg.line_styles.training
    else:
        return cfg.line_styles.clips


def get_color(tag: str, cfg: DictConfig) -> str:
    if tag.startswith('val_'):
        return cfg.colors.validation
    elif tag.startswith('train_'):
        return cfg.colors.training
    else:
        return cfg.colors.clips


def plot_scalars(
    scalar_data: Dict[str, Dict[str, List[float]]], 
    cfg: DictConfig,
    save_path: str = None
):
    plt.figure(figsize=(cfg.plot.figsize.width, cfg.plot.figsize.height))

    for tag, data in scalar_data.items():
        line_style = get_line_style(tag, cfg)
        color = get_color(tag, cfg)
        
        plt.plot(
            data['steps'], 
            data['values'],
            label=tag,
            color=color,
            **line_style
        )

    plt.xlabel(cfg.plot.style.xlabel)
    plt.ylabel(cfg.plot.style.ylabel)
    plt.title(cfg.plot.style.title)
    plt.grid(cfg.plot.style.grid)
    
    plt.legend(
        loc=cfg.legend.location,
        fontsize=cfg.legend.fontsize,
        frameon=cfg.legend.frameon,
        framealpha=cfg.legend.framealpha
    )

    if save_path and cfg.output.save_plot:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(
            save_path,
            format=cfg.output.plot_format,
            dpi=cfg.output.dpi
        )
    
    if cfg.output.show_plot:
        plt.show()
    else:
        plt.close()


@hydra.main(version_base=None, config_path="../config", config_name="visualization")
def analyze_tfevents(cfg: DictConfig):
    GlobalHydra.instance().clear()
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    print(f"Loading events from: {cfg.logdir}")

    event_acc = load_tfevents(cfg.logdir, cfg)

    if cfg.output.print_available_tags:
        available_tags = event_acc.Tags()
        print("\nAvailable tags:")
        for tag_type, tags in available_tags.items():
            if tags:
                print(f"\n{tag_type}:")
                for tag in tags:
                    print(f"  - {tag}")

    metric_tags = generate_metric_tags(cfg)

    scalar_data = extract_scalar_data(event_acc, metric_tags)
    if scalar_data:
        print("\nPlotting scalar values...")
        plot_scalars(scalar_data, cfg, cfg.save_plot)
    else:
        print("\nNo scalar data found for the specified metrics.")


if __name__ == "__main__":
    analyze_tfevents()