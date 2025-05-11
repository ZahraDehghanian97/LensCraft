import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from visualization.visualization_utils import (
    load_tfevents,
    generate_metric_tags,
    print_available_tags,
    extract_scalar_data,
    plot_scalars
)



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