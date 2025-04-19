import os
import math
import hydra
import string
import numpy as np
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from data.simulation.constants import CLIP_PARAMETERS

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
                    clip_tags.append(f"{prefix}_clip_{i}_epoch")
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


def get_line_style(tag: str, cfg: DictConfig) -> Dict[str, Any]:
    color = get_color(tag, cfg)
    label = get_label(tag)
    style = {"color": color, "label": label}
    if tag.count("train"):
        style = {**style, **cfg.line_styles.training}
    elif tag.count("val"):
        style = {**style, **cfg.line_styles.validation}
    return style


def get_color(tag: str, cfg: DictConfig) -> str:
    if tag.startswith("train_clip") or tag.startswith("val_clip"):
        index = int(tag.split("_")[-2])
        return list(cfg.color_map.clip_loss[index].values())[0]
    elif tag.count("loss"):
        return list(cfg.color_map.trajectory_loss[0].values())[0]
    elif tag.count("trajectory"):
        return list(cfg.color_map.trajectory_loss[1].values())[0]
    elif tag.count("total"):
        return list(cfg.color_map.trajectory_loss[2].values())[0]
    return None


def get_label(tag: str) -> str:
    label = tag
    if tag.startswith("train_clip") or tag.startswith("val_clip"):
        index = int(tag.split("_")[-2])
        label = tag.replace(str(index), CLIP_PARAMETERS_KEYS[index])
    label = label.replace("_epoch", "")
    label = label.replace("_clip", "")
    return label


def get_title(subplot: str) -> str:
    title = subplot.replace("_tags", "")
    title = title.replace("_", " ") 
    title = title + " loss"
    title = string.capwords(title)
    title = title.replace("Clip", "CLIP")
    return title


def calc_dimensions(cfg: DictConfig, subplots: List[str]) -> tuple:
    n_cols = cfg.plot.n_cols
    n_rows = math.ceil(len(subplots) / n_cols)
    width = cfg.plot.figsize.width * n_cols
    height = cfg.plot.figsize.height * n_rows
    if "clip_tags" in subplots:
        height += 6
    return width, height, n_cols, n_rows


def get_average_embedding_loss(
        clip_losses: dict[str, dict[str, list]], 
        plot_list: list[str], 
        n_clip_embeddings: int) -> dict[str, list]:
    steps = clip_losses["val_clip_0_epoch"]["steps"]
    train_values = np.zeros(len(steps))
    valid_values = np.zeros(len(steps))
    for item, value in zip(["train", "val"], [train_values, valid_values]):
        for i in range(n_clip_embeddings): 
            value += np.array(clip_losses[f"{item}_clip_{i}_epoch"]["values"])
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
        steps = list(list(scalar_data.values())[0].values())[0]["steps"]
        ax[row, col].set_xlim(min(steps), max(steps))
        ax[row, col].set_ylim(0, None)
        if subplot.startswith("clip") and cfg.plot.style.apply_fixed_ylim_clip:
            ax[row, col].set_ylim(0, 1)

    if cfg.plot.tight_layout: 
        plt.tight_layout()

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


def print_available_tags(cfg: DictConfig, event_acc:event_accumulator.EventAccumulator) -> None:
    if cfg.output.print_available_tags:
        available_tags = event_acc.Tags()
        print("\nAvailable tags:")
        for tag_type, tags in available_tags.items():
            if tags:
                print(f"\n{tag_type}:")
                for tag in tags:
                    print(f"  - {tag}")


@hydra.main(version_base=None, config_path="../config", config_name="visualization")
def analyze_tfevents(cfg: DictConfig):
    GlobalHydra.instance().clear()
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    print(f"Loading events from: {cfg.logdir}")
    event_acc = load_tfevents(cfg)
    metric_tags = generate_metric_tags(cfg)
    print_available_tags(cfg, event_acc)

    scalar_data = dict()
    for metric_tag_key in metric_tags.keys():
        scalar_data[metric_tag_key] = extract_scalar_data(event_acc, metric_tags[metric_tag_key])
    if scalar_data:
        print("\nPlotting scalar values...")
        plot_scalars(scalar_data, metric_tags, cfg, cfg.save_plot)
    else:
        print("\nNo scalar data found for the specified metrics.")


if __name__ == "__main__":
    analyze_tfevents()