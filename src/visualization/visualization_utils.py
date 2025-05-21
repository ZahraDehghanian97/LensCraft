import os
import numpy as np
from omegaconf import DictConfig
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
from typing import Dict, List
from data.simulation.utils import CLIP_PARAMETERS
from visualization.style import calc_dimensions, get_line_style, get_title


CLIP_PARAMETERS_KEYS = [item[0] for item in CLIP_PARAMETERS]


def generate_metric_tags(cfg: DictConfig) -> List[str]:
    tags = dict()
    subplots = cfg.metrics.losses_list

    if len(subplots):
        if "clip" in subplots:
            clip_tags = list()
            prefixes = []
            if cfg.metrics.clips.include_validation:
                prefixes.append("val")
            if cfg.metrics.clips.include_training:
                prefixes.append("train")
            for prefix in prefixes:
                for i in range(cfg.metrics.clips.num_clips):
                    clip_tags.append(f"{prefix}_clip_elements_{i}_epoch")
            tags["clip_tags"] = clip_tags
            if cfg.metrics.clips.categorize_embeddings:
                cinematography_structure = {"cinematography_" + k: v for k, v in cfg.clip_structure.cinematography.items()}
                simulation_structure = {"simulation_" + k: v for k, v in cfg.clip_structure.simulation.items()}
                embeddings = {**cinematography_structure, **simulation_structure}
                for key in embeddings.keys():
                    category_tags = list()
                    for prefix in prefixes:
                        for clip in embeddings[key]:
                            category_tags.append(f"{prefix}_{clip}_epoch")
                    tags["clip_" + key + "_tags"] = category_tags
            if cfg.metrics.clips.average_clip_loss:
                tags["average_clip"] = ["train_average_clip_loss", "val_average_clip_loss"]
            
        if "trajectory" in subplots:
            tags["trajectory_tags"] = cfg.metrics.trajectory

        if "contrastive" in subplots:
            tags["contrastive_tags"] = cfg.metrics.contrastive
        
        if "first_frame_loss" in subplots:
            tags["first_frame_tag"] = cfg.metrics.first_frame_loss

        if "relative_loss" in subplots:
            tags["relative_tag"] = cfg.metrics.relative_loss
        
        if "cycle" in subplots:
            tags["cycle_tag"] = cfg.metrics.cycle_loss

        if "speed" in subplots:
            tags["speed_tag"] = cfg.metrics.speed_loss

        if "total" in subplots:
            tags["total"] = cfg.metrics.total_loss

            
    else:
        raise ValueError("The losses list must include at least one of the following: 'clip', 'trajectory', or 'contrastive'.")
    return tags




def load_tfevents(cfg: DictConfig) -> event_accumulator.EventAccumulator:
    event_acc = event_accumulator.EventAccumulator(
        cfg.logdir, size_guidance=cfg.event_accumulator.size_guidance
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




def initialize(cfg: DictConfig) -> None:
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'font.size': cfg.plot.style.font_size})
    plt.style.use(cfg.plot.style.plt_style)
    plt.rcParams["font.family"] = cfg.plot.style.font_family







def get_average_embedding_loss(
        clip_losses: dict[str, dict[str, list]],
        plot_list: list[str],
        n_clip_embeddings: int) -> dict[str, list]:
    steps = clip_losses["val_clip_elements_0_epoch"]["steps"]
    train_values = np.zeros(len(steps))
    valid_values = np.zeros(len(steps))
    for item, value in zip(["train", "val"], [train_values, valid_values]):
        for i in range(n_clip_embeddings):
            value += np.array(clip_losses[f"{item}_clip_elements_{i}_epoch"]["values"])
    average_embedding_loss_train = {"steps": steps, "values": list(train_values / n_clip_embeddings)}
    average_embedding_loss_valid = {"steps": steps, "values": list(valid_values / n_clip_embeddings)}
    average_embedding_loss = {
        plot_list[0]: average_embedding_loss_train,
        plot_list[1]: average_embedding_loss_valid
        }
    return average_embedding_loss




def plot_scalars(
    scalar_data: Dict[str, Dict[str, List[float]]],
    metric_tags: List[str],
    cfg: DictConfig,
    save_path: str = None
):
    initialize(cfg)
    model_version = cfg.logdir.split("/")[-1]
    save_path = save_path + model_version + "." + cfg.output.plot_format
    model_version = model_version.replace("_", " ")

    subplots = list(metric_tags.keys())
    width, height, n_cols, n_rows = calc_dimensions(cfg, subplots)

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(width, height))
    fig.suptitle(
        cfg.plot.style.title + f" ({model_version})",
        fontsize=cfg.plot.style.suptitle_font_size,
        y=cfg.plot.style.suptitle_offset,
        fontweight=cfg.plot.style.suptitle_font_weight
    )

    for i, subplot in enumerate(subplots):
        col = i % n_cols
        row = i // n_cols
        if subplot == "average_clip":
            scalar_data_item = get_average_embedding_loss(
                scalar_data["clip_tags"],
                metric_tags["average_clip"],
                n_clip_embeddings=cfg.metrics.clips.num_clips
                )
        else:
            scalar_data_item = scalar_data[subplot]
            
        for tag, data in scalar_data_item.items():
            line_style = get_line_style(tag, cfg)
            ax[row, col].plot(data['steps'], data['values'], **line_style)

        ax[row, col].legend(**cfg.legend)
        title = get_title(subplot)
        ax[row, col].set_title(label=title, **cfg.plot.style.title_params)
        ax[row, col].patch.set_edgecolor(cfg.plot.style.border_color)  
        ax[row, col].patch.set_linewidth(cfg.plot.style.border_linewidth)
        ax[row, col].set_ylabel(cfg.plot.style.ylabel, labelpad=cfg.plot.style.labelpad)
        ax[row, col].set_xlabel(cfg.plot.style.xlabel, labelpad=cfg.plot.style.labelpad)
        ax[row, col].xaxis.set_tick_params(pad=cfg.plot.style.xticks_pad)
        ax[row, col].yaxis.set_tick_params(pad=cfg.plot.style.yticks_pad)
        ax[row, col].grid(**cfg.plot.style.major_grid_style)
        ax[row, col].grid(**cfg.plot.style.minor_grid_style)
        ax[row, col].minorticks_on()
        steps = list(list(scalar_data.values())[9].values())[0]["steps"]
        ax[row, col].set_xlim(min(steps), max(steps))
        ax[row, col].set_ylim(0, None)
        if subplot.startswith("clip") and cfg.plot.style.apply_fixed_ylim_clip:
            ax[row, col].set_ylim(0, 1)

    if cfg.plot.tight_layout:
        plt.tight_layout()

    if save_path and cfg.output.save_plot:
        plt.savefig(
            os.path.join(cfg.logdir, "visualization.png"),
            format=cfg.output.plot_format,
            dpi=cfg.output.dpi
        )
    
    if cfg.output.show_plot:
        plt.show()
    else:
        plt.close()




def print_available_tags(cfg: DictConfig, event_acc:event_accumulator.EventAccumulator) -> None:
    if cfg.output.print_available_tags:
        available_tags = event_acc.Tags()
        print("\nAvailable tags:")
        for tag_type, tags in available_tags.items():
            if tags:
                print(f"\n{tag_type}:")
                for tag in tags:
                    print(f"  - {tag}")




