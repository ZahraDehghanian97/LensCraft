from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

val_losses=[
#   "val_loss_epoch",
#   "val_trajectory_epoch",
  "val_clip_0_epoch",
  "val_clip_1_epoch",
  "val_clip_2_epoch",
  "val_clip_3_epoch",
  "val_clip_4_epoch",
  "val_clip_5_epoch",
  "val_clip_6_epoch",
  "val_clip_7_epoch",
  "val_clip_8_epoch",
  "val_clip_9_epoch",
  "val_clip_10_epoch",
  "val_clip_11_epoch",
  "val_clip_12_epoch",
  "val_clip_13_epoch",
  "val_clip_14_epoch",
  "val_clip_15_epoch",
  "val_clip_16_epoch",
  "val_clip_17_epoch",
  "val_clip_18_epoch",
  "val_clip_19_epoch",
  "val_clip_20_epoch",
  "val_clip_21_epoch",
  "val_clip_22_epoch",
  "val_clip_23_epoch",
  "val_clip_24_epoch",
  "val_clip_25_epoch",
  "val_clip_26_epoch",
  "val_clip_27_epoch",
#   "val_total_epoch",
#   "train_loss_epoch",
#   "train_trajectory_epoch",
#   "train_clip_0_epoch",
#   "train_clip_1_epoch",
#   "train_clip_2_epoch",
#   "train_clip_3_epoch",
#   "train_clip_4_epoch",
#   "train_clip_5_epoch",
#   "train_clip_6_epoch",
#   "train_clip_7_epoch",
#   "train_clip_8_epoch",
#   "train_clip_9_epoch",
#   "train_clip_10_epoch",
#   "train_clip_11_epoch",
#   "train_clip_12_epoch",
#   "train_clip_13_epoch",
#   "train_clip_14_epoch",
#   "train_clip_15_epoch",
#   "train_clip_16_epoch",
#   "train_clip_17_epoch",
#   "train_clip_18_epoch",
#   "train_clip_19_epoch",
#   "train_clip_20_epoch",
#   "train_clip_21_epoch",
#   "train_clip_22_epoch",
#   "train_clip_23_epoch",
#   "train_clip_24_epoch",
#   "train_clip_25_epoch",
#   "train_clip_26_epoch",
#   "train_clip_27_epoch",
#   "train_total_epoch",
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
    logdir = "lightning_logs/version_43"
    analyze_tfevents(logdir, save_plot="tensorboard_plot.png")
