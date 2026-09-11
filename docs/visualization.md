# Qualitative comparison visualizer

Run commands from the `LenseCraft/` project directory. The single entry point is
`src/visualization.py`: use a browser to inspect camera trajectories, compare
model outputs on one sample, and select the input keyframes for a paper figure.
Viser provides the interactive 3D scene; publication exports use Matplotlib.

## Try it without datasets or checkpoints

Use Python 3.10 or newer in a virtual environment:

```bash
pip install -r requirements-visualization.txt
python src/visualization.py --demo
```

Open the address printed in the terminal. The demo contains synthetic example
trajectories so you can try the controls and exports without PyTorch or a GPU.
Its model labels are illustrative; the demo is not an experimental result.

## Browser workflow

The browser has three tabs:

1. **Input**: choose **Dataset** and **Split**, optionally enter **Dataset path**,
   then set **Sample index** or **Sample ID** and click **Load sample**.
   **Browse → Previous / Next** moves through the split. In **Generate model
   outputs**, check LensCraft or several models, choose **Conditioning**, and
   click **Generate selected models**. **Text + keyframes** and **Keyframes
   only** reveal the **Input keyframes** field; enter values such as `0,14,29`.
2. **Compare**: choose **Side by side** or **Overlay** in **Layout**. Use
   **Visible trajectories** to select outputs, **Time (%)** or **Play** to move
   through the clip, and **Camera poses** to control how many camera frustums
   are shown. Toggle **Subject and motion**, **Input keyframes**, and
   **Ground grid** as needed. **Fit view** frames the scene, and
   **Download viewport PNG** captures the current browser view.
3. **Paper**: click **Add current sample to figure** to keep a sample as a row,
   then load and add further samples. With no saved rows, the current sample
   is used. Choose a **Filename**, optional **Figure title**, **Elevation**,
   **Azimuth**, and **PNG resolution**. Click **Preview figure**, then
   **Export PNG + PDF + SVG** to save the files and download them as a ZIP.
   **Save comparison data** saves and downloads the complete portable bundle.

Use **Separate input keyframe panel** in the Paper tab to include the input
poses as their own column. Figure elevation and azimuth control the
orthographic export view; the viewport PNG uses the interactive camera view.
Exports go to `qualitative/` by default, or the directory of a CLI `--export`
path. **Clear figure rows** starts a new figure collection.

For saved data, use **Input → Saved results → Result file**, choose
**Result index**, and click **Open result**. **Opened samples** returns to any
sample already loaded in the session. Dataset and result paths are paths on
the machine running the visualizer. Playback uses percentage of clip duration
so outputs with different frame counts remain aligned in time.

## Select a dataset sample

Live inference uses the existing dataset loaders, model adapters, and
`config/inference.yaml`. Install the full project dependencies and prepare the
datasets and checkpoints as described in the [main README](../README.md).

Set paths in the project `.env` file. A typical LensCraft setup includes:

```dotenv
SIMULATION_DATA_PATH=/path/to/simulation-data
CLIP_EMBEDDINGS_CACHE_DIR=/path/to/clip-cache
TEST_CHECKPOINT_PATH=/path/to/lenscraft.ckpt
TEST_CONFIG_PATH=/path/to/train/.hydra/config.yaml
```

Use the training configuration matching your checkpoint, especially when its
model architecture differs from the current defaults.

```bash
# Inspect LensCraft on a held-out simulation sample.
python src/visualization.py --dataset simulation --split test --sample 0 \
  --models lens_craft

# Compare the four supported model adapters on an E.T. sample.
python src/visualization.py --dataset et --split test --sample 0 \
  --models lens_craft et ccdm gendop

# Reconstruct a CCDM sample with LensCraft.
python src/visualization.py --dataset ccdm --sample 0 \
  --models lens_craft --mode reconstruction

# Load one simulation file directly.
python src/visualization.py --dataset simulation --split all \
  --data-path /path/to/clip.msgpack --models lens_craft
```

`--dataset` selects the input source; `--models` selects which models generate
outputs for that source. Available datasets are `simulation`, `et`, and `ccdm`.
Available models are `lens_craft`, `et`, `ccdm`, and `gendop`.
LensCraft text conditioning requires simulation data or annotated E.T. data.
Raw CCDM data lacks the structured cinematography annotations, so use
`--mode reconstruction` or `--mode key_framing` for LensCraft on CCDM;
`prompt_generation` and `key_framing+prompt` report an unavailable output.
The browser and terminal show model errors while retaining any successful
outputs in a comparison.

Omit `--models` to load the reference and subject first, then choose models in
the browser. `--sample` is an index within the selected split. For a known
simulation filename, use `--sample-id` with its path or filename stem and
`--split all` to search the complete dataset:

```bash
python src/visualization.py --dataset simulation --split all \
  --sample-id my-simulation --models lens_craft
```

Simulation and CCDM splits use seed 42 by default, matching the data module;
E.T. uses its native splits. Use `--seed` to control the dataset split and
generation seed.

`--data-path` overrides the configured input location for the selected dataset.
For simulation, it accepts a dataset directory or one `.msgpack` file; for
CCDM, a directory or `data.npy`; for E.T., the dataset directory. This is also
available as **Dataset path** in the browser.

Configure the additional paths for the models/datasets you select:

| Component | Configuration |
| --- | --- |
| E.T. input data | `ET_DATA_DIR`, `ET_CIN_LANG_PATH`, `DIRECTOR_PROJECT_DIR` |
| E.T./DIRECTOR model | `DIRECTOR_PROJECT_DIR`, `ET_DATA_DIR`; its adapter prepares the released checkpoints on first use |
| CCDM input data | `CCDM_DATA_DIR` containing `data.npy` |
| CCDM model | `CCDM_DATA_DIR`, `CCDM_CHECKPOINT_PATH` |
| GenDoP model | `GENDOP_CHECKPOINT_PATH` and the extra dependencies/cache setup in the [GenDoP section](../README.md#gendop-baseline) |

The GenDoP adapter requires CUDA. Viewing portable comparison bundles and
exporting their figures only requires the lightweight visualization dependencies.

## Show input keyframes

Keyframe indices are zero-based and refer to the displayed reference
trajectory. Simulation references normally contain 30 frames; E.T. and CCDM
references use their valid dataset frames. The viewer maps the indices to
LensCraft's model timeline and displays the camera poses actually supplied
to the model.

```bash
python src/visualization.py --dataset simulation --sample 0 \
  --models lens_craft --mode 'key_framing+prompt' --keyframes 0,14,29
```

The keyframe modes use the selected camera poses as model inputs. The viewer
also makes those poses visible so readers can distinguish the conditioning
frames from the generated path. Baseline adapters generate from prompts using
their existing interfaces; they do not acquire keyframe conditioning simply
because keyframes are displayed.

## Reuse inference results

Portable comparison bundles use the JSON schema `lenscraft.qualitative.v1`.
They contain the actual camera and subject poses, prompts, and keyframes, so
they can be moved to a laptop without datasets or checkpoints:

```bash
python src/visualization.py --results /path/to/comparison-bundle.json
```

The older `inference_result.json` files produced by `src/inference.py` are also
accepted. These files can contain normalized coordinates, so importing them
requires the project inference dependencies and `SIMULATION_DATA_PATH` pointing
to the training dataset's normalization statistics. Convert them to a portable
bundle in that environment before moving them to a machine with only the
visualization dependencies. Loading either format does not rerun inference.

Save one selected sample, including all its generated model outputs:

```bash
# On the inference machine:
python src/visualization.py --dataset simulation --sample 0 \
  --models lens_craft et ccdm gendop \
  --save-result qualitative/scene-0.bundle.json --headless

# Or convert one sample from an older inference result:
python src/visualization.py --results /path/to/inference_result.json --sample 0 \
  --save-result qualitative/scene-0.bundle.json --headless
```

Pass several files to `--results` with `--save-result` to combine their selected
samples into one portable bundle. Use **Result index** in the browser or
`--sample` on the command line to select a sample within that bundle.

## Export a paper figure

Export directly from a terminal, including on a machine without a display:

```bash
# Test the publication layout with synthetic data.
python src/visualization.py --demo --export qualitative/demo --headless

# Export a real, saved inference result.
python src/visualization.py --results /path/to/comparison-bundle.json \
  --export qualitative/comparison --headless

# Compare several samples as separate rows in the same figure.
python src/visualization.py \
  --results qualitative/scene-0.bundle.json qualitative/scene-1.bundle.json \
  --export qualitative/multi-sample --headless
```

A suffixless output path produces four files:

| File | Purpose |
| --- | --- |
| `comparison.pdf` | Vector figure for the paper |
| `comparison.svg` | Editable vector figure |
| `comparison.png` | Raster figure at export resolution |
| `comparison.json` | Provenance: sample identifiers, prompts, selected methods, keyframes, view, shared bounds, and trajectory hashes |

Pass a filename such as `--export qualitative/comparison.pdf` to produce just
that figure format and the JSON sidecar. Reusing an output path replaces its
generated files. Raster export defaults to 300 DPI; use `--dpi 600` when needed.

Each row compares one sample, with a separate input-keyframe panel and one
panel per selected trajectory. All panels in a row share the same orthographic
view and spatial bounds. Gold camera frustums mark the actual input poses;
circles and diamonds mark trajectory starts and ends. The exporter connects
the recorded positions without smoothing or independently rescaling methods.

The JSON sidecar describes how the figure was made; it does not contain the
complete trajectory arrays and is not an input to `--results`.

## Configuration and remote use

Use `--override` for dataset/model settings from the existing Hydra config:

```bash
python src/visualization.py --dataset simulation --sample 0 \
  --override 'data.dataset.config.data_path=/path/to/another/simulation-data'
```

Overrides apply to every selected model. For comparisons of several models,
set each model's checkpoint through its environment variables. For a LensCraft
run you can also provide
`--override 'training.model.inference.checkpoint_path=/path/to/lenscraft.ckpt'`
and `--override 'training.model.inference.config=/path/to/train/.hydra/config.yaml'`.

Use `python src/visualization.py --help` for the complete command-line options.
To view a server session through SSH, forward its port:

```bash
ssh -L 8080:127.0.0.1:8080 user@server
```

Run the visualizer on that server with `--port 8080`, then open
`http://127.0.0.1:8080` on your local machine.

The former standalone trajectory viewers and their dedicated Hydra configs
have been removed. For training curves, use `src/plot_losses.py` or
`src/plot_clatr_losses.py`; evaluation's t-SNE helpers remain available.
# Quantitative evaluation

For fixed semantic evaluators, sparse keyframe sweeps, and measured pose/framing
errors, see [evaluation.md](evaluation.md). The evaluation cache records the
actual constraints; figure selection alone is not a quantitative keyframe test.
