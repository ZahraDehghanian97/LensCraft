"""Browse camera trajectories and export qualitative paper comparisons."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    source = cli.add_mutually_exclusive_group()
    source.add_argument("--demo", action="store_true", help="Synthetic example; no checkpoints needed")
    source.add_argument("--dataset", choices=("simulation", "et", "ccdm"))
    source.add_argument("--results", nargs="+", type=Path, help="Saved comparisons or inference_result.json files")
    cli.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    cli.add_argument("--sample", type=int, default=0, help="Zero-based index in dataset split or result batch")
    cli.add_argument("--sample-id", help="Exact dataset sample ID or filename")
    cli.add_argument("--data-path", type=Path, help="Dataset directory or a simulation file (use --split all for exact files)")
    cli.add_argument("--models", nargs="+", choices=("lens_craft", "et", "ccdm", "gendop"))
    cli.add_argument("--mode", choices=("prompt_generation", "key_framing+prompt", "key_framing", "reconstruction"), default="prompt_generation")
    cli.add_argument("--keyframes", default="", help="Input frame indices, e.g. 0,14,29")
    cli.add_argument("--override", action="append", default=[], metavar="KEY=VALUE", help="Hydra override; may be repeated")
    cli.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--host", default="127.0.0.1")
    cli.add_argument("--port", type=int, default=8080)
    cli.add_argument("--export", type=Path, help="Figure path; no suffix exports PNG/PDF/SVG plus metadata")
    cli.add_argument("--headless", action="store_true", help="Export without starting a browser server")
    cli.add_argument("--save-result", type=Path, help="Save a reusable comparison bundle")
    cli.add_argument("--dpi", type=int, default=300)
    return cli


def main(argv: list[str] | None = None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    if args.headless and not (args.export or args.save_result):
        cli.error("--headless requires --export or --save-result")
    if args.models and not args.dataset:
        cli.error("--models requires --dataset")
    if args.data_path and not args.dataset:
        cli.error("--data-path requires --dataset")
    if args.sample < 0 or args.dpi < 1:
        cli.error("--sample must be non-negative and --dpi must be positive")
    if args.export and args.save_result and args.export.with_suffix(".json").resolve() == args.save_result.resolve():
        cli.error("Figure metadata and comparison data need different paths; use --save-result NAME.bundle.json")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    from visualization.data import VisualizationRepository, load_result, make_demo, save_result
    try:
        keyframes = sorted(set(int(v.strip()) for v in args.keyframes.split(",") if v.strip())) or None
        repository = VisualizationRepository(overrides=args.override, device=args.device, seed=args.seed)
        if args.dataset:
            sample = repository.load_sample(args.dataset, args.split, index=args.sample, sample_id=args.sample_id, data_path=args.data_path)
            if args.models:
                sample = repository.generate(sample, models=args.models, mode=args.mode, keyframes=keyframes)
                if args.headless and not sample.metadata.get("last_run", {}).get("generated_methods"):
                    raise RuntimeError(f"No requested model produced an output: {sample.metadata.get('errors', {})}")
            samples = [sample]
        elif args.results:
            samples = [load_result(path, index=args.sample) for path in args.results]
        else:
            samples = [make_demo()]
        for sample in samples:
            if sample.metadata.get("errors"):
                logging.warning("Unavailable model outputs: %s", sample.metadata["errors"])
        if args.save_result:
            save_result(samples, args.save_result)
        if args.export:
            from visualization.export import export_figure
            for kind, path in export_figure(samples, args.export, dpi=args.dpi).items():
                print(f"{kind}: {path}")
        if not args.headless:
            from visualization.app import VisualizationApp
            VisualizationApp(samples, repository=repository, host=args.host, port=args.port,
                             output_dir=args.export.parent if args.export else Path("qualitative")).run()
    except (ValueError, IndexError, OSError, ImportError, RuntimeError) as exc:
        cli.exit(1, f"Visualizer: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
