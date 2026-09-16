# Qualitative comparison visualizer

Run commands from the `LensCraft/` project directory. The single entry point is
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

Use `--figure-mode static` for stationary-subject examples or
`--figure-mode dynamic` for a moving subject. Add `--demo-suite` instead of
`--demo` to open the complete set of examples for that figure type:

```bash
python src/visualization.py --demo-suite --figure-mode static
python src/visualization.py --demo-suite --figure-mode dynamic
```

The static suite includes orbit, dolly, truck, crane, pan, tilt, and spiral
camera motions. The dynamic suite covers straight tracking, curved following,
and orbiting a walking subject. All suite outputs are synthetic examples.
In the browser, **Input → Synthetic motion demos → Subject type** chooses the
suite; click **Load demo collection**, then select examples under
**Saved results → Opened samples**.

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
3. **Paper**: choose **Static subject** or **Dynamic subject** in **Figure type**.
   Click **Add current sample to figure** to keep a sample in the figure,
   then load and add further samples. With no saved samples, the current sample
   is used. Choose a **Filename**, optional **Figure title**, **Elevation**,
   **Azimuth**, and **PNG resolution**. Click **Preview figure**, then
   **Export PNG + PDF + SVG** to save the files and download them as a ZIP.
   **Save comparison data** saves and downloads the complete portable bundle.

Use **Separate input keyframe panel** with **Static subject** to include the
input poses as their own column. **Dynamic subject** always has five time
columns; input keyframes can be overlaid without adding a column. Figure
elevation and azimuth control the
orthographic export view; the viewport PNG uses the interactive camera view.
Exports go to `qualitative/` by default, or the directory of a CLI `--export`
path. **Clear figure samples** starts a new figure collection.

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

# Export the stationary-subject suite with varied camera movements.
python src/visualization.py --demo-suite --figure-mode static \
  --export qualitative/static-movements --headless

# Export the moving-subject suite as five-instant comparisons.
python src/visualization.py --demo-suite --figure-mode dynamic \
  --export qualitative/dynamic-moments --headless

# Export a real, saved inference result.
python src/visualization.py --results /path/to/comparison-bundle.json \
  --export qualitative/comparison --headless

# Show a saved moving-subject sample at five times for every model.
python src/visualization.py --results /path/to/moving-subject.bundle.json \
  --figure-mode dynamic --export qualitative/moving-subject --headless

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

There are two figure types, selected with `--figure-mode` or **Figure type**:

| Figure type | Rows | Columns |
| --- | --- | --- |
| `static` (default) | One sample per row | One panel per selected trajectory, plus the optional input-keyframe panel |
| `dynamic` | One row per sample and selected trajectory | Exactly five instants: 0%, 25%, 50%, 75%, and 100% of the clip |

Static figures show the complete camera movement around a stationary subject.
Circles and diamonds mark trajectory starts and ends. This preserves the
original comparison layout.

In each dynamic panel, the camera and subject at the current instant are
highlighted together. Their other sampled positions, both before and after
that instant, remain faintly visible to show the motion. Every method and all
five instants for a sample use the same orthographic view and spatial bounds,
so movement remains comparable across both time and models. Camera and subject
tracks with different frame counts use the nearest recorded pose at each
percentage of their own duration. No intermediate poses are synthesized; short
tracks can therefore repeat a pose in adjacent columns. Dynamic export requires
subject poses. A subject without dimensions is drawn as a position marker.

Gold camera frustums mark actual input keyframes when enabled. Dynamic figures
overlay these poses within the five time columns and never add a separate
input column. Both figure types connect recorded positions without smoothing
or independently rescaling methods.

The JSON sidecar records the figure type and, for dynamic figures, the sampled
times and camera/subject frame indices alongside the shared bounds. It describes
how the figure was made; it does not contain the complete trajectory arrays and
is not an input to `--results`.

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

## Build a gallery of paper candidates

Export every saved sample as a separate candidate so figures can be compared
and shortlisted before choosing the paper examples:

```bash
python scripts/export_paper_candidates.py \
  --results /path/to/comparison-bundles/ \
  --output-dir qualitative/paper-candidates
```

`--results` accepts several portable bundle files or directories. The exporter
loads all samples, combines repeated copies of the same comparison, and keeps
different model outputs or input keyframes as distinct candidates. It uses
recorded poses and does not run inference. Synthetic demo bundles are rejected.

Open `qualitative/paper-candidates/index.html` in a browser to view the gallery.
Search prompts and sample identifiers, filter the candidates, and mark a
shortlist for download. Each candidate links to its PNG preview, paper-ready
PDF, editable SVG, provenance JSON, and portable comparison bundle. Static
subjects use the comparison layout; moving subjects use one row per model and
five time columns. Subject rotation also counts as motion.

The default PNG resolution is 300 DPI; use `--dpi 600` for a higher-resolution
final export. Use a new output directory for another collection, or explicitly
pass `--overwrite` to regenerate an existing candidate gallery.

To expand the real example pool from the completed paper evaluation, run the
preparation script in the environment containing the frozen source, dataset,
and saved trajectory caches:

```bash
python scripts/prepare_paper_candidates.py \
  --project-dir /path/to/frozen/LensCraft \
  --base-run-dir /path/to/completed-paper-run \
  --output-dir /path/to/paper-candidate-bundles \
  --per-cohort 12 --seed 42
```

This selects 12 static-subject and 12 dynamic-subject scenes from the measured
held-out cohorts and reuses their saved model predictions. By default, each
scene produces a prompt comparison and a four-keyframe comparison: 48 figure
candidates from 24 distinct scenes. Use `--prompt-only` for just the 24 prompt
comparisons, or `--keyframe-count` to select another cached keyframe setting.

Selection is deterministic and cycles through recorded camera-motion types;
it does not rank examples by model quality. `--seed` changes the candidate
selection, while the measured evaluation split remains fixed. The resulting
`bundles/` directory can be copied to a machine with only the visualization
dependencies and passed to `export_paper_candidates.py` as above. The preparation
manifest records source-cache hashes and the selection protocol. Increasing
`--per-cohort` expands the selection without retraining or rerunning inference.
