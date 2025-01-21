from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

val_losses=[
  "val_loss_epoch",
  "val_trajectory_epoch",
#   "val_clip_cinematography_init_setup_epoch",
#   "val_clip_cinematography_movement_epoch",
#   "val_clip_cinematography_end_setup_epoch",
#   "val_clip_simulation_init_setup_epoch",
#   "val_clip_simulation_movement_epoch",
#   "val_clip_simulation_end_setup_epoch",
#   "val_clip_simulation_constraints_epoch",
  "val_total_epoch",
  "train_loss_epoch",
  "train_trajectory_epoch",
#   "train_clip_cinematography_init_setup_epoch",
#   "train_clip_cinematography_movement_epoch",
#   "train_clip_cinematography_end_setup_epoch",
#   "train_clip_simulation_init_setup_epoch",
#   "train_clip_simulation_movement_epoch",
#   "train_clip_simulation_end_setup_epoch",
#   "train_clip_simulation_constraints_epoch",
]

def load_tfevents(logdir):
    size_guidance = {
        'scalars': 0,
        'histograms': 0,
        'images': 0,
        'tensors': 0,
    }
    event_acc = event_accumulator.EventAccumulator(
        logdir, size_guidance=size_guidance)
    event_acc.Reload()
    return event_acc


def extract_scalar_data(event_acc):
    scalar_data = {}
    available_tags = event_acc.Tags()['scalars']

    for tag in available_tags:
        events = event_acc.Scalars(tag)
        steps = [event.step for event in events]
        values = [event.value for event in events]
        scalar_data[tag] = {'steps': steps, 'values': values}

    return scalar_data


def plot_scalars(scalar_data, figsize=(12, 8), save_path=None):
    plt.figure(figsize=figsize)

    for tag, data in scalar_data.items():
        if tag in val_losses:
            plt.plot(data['steps'], data['values'], label=tag)

    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title('TensorBoard Scalar Values')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    plt.show()


def analyze_tfevents(logdir, save_plot=None):
    print(f"Loading events from: {logdir}")

    event_acc = load_tfevents(logdir)

    available_tags = event_acc.Tags()
    print("\nAvailable tags:")
    for tag_type, tags in available_tags.items():
        if tags:
            print(f"\n{tag_type}:")
            for tag in tags:
                print(f"  - {tag}")

    scalar_data = extract_scalar_data(event_acc)
    if scalar_data:
        print("\nPlotting scalar values...")
        plot_scalars(scalar_data, save_path=save_plot)
    else:
        print("\nNo scalar data found in the events.")


if __name__ == "__main__":
    logdir = "lightning_logs/version_17"
    analyze_tfevents(logdir, save_plot="tensorboard_plot.png")
