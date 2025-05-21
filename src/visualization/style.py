import math
import string
from omegaconf import DictConfig
from typing import Dict, Any, List
from data.simulation.utils import CLIP_PARAMETERS



CLIP_PARAMETERS_KEYS = [item[0] for item in CLIP_PARAMETERS]

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