"""CLI for fresh human text -> LensCraft conditioning via Vercel AI Gateway."""

from __future__ import annotations

import argparse
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys

from annotator.gateway import GatewayTranslator, TranslationError, _atomic_json


def _records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    rows = json.loads(text) if path.suffix == ".json" else [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list) or not rows:
        raise TranslationError("Batch input must be a nonempty JSON array or JSONL of {id, prompt} objects")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
            raise TranslationError("Every batch row needs a nonempty string id")
        if row["id"] in seen:
            raise TranslationError(f"Duplicate input id: {row['id']}")
        seen.add(row["id"])
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise TranslationError(f"Input {row['id']} needs a nonempty prompt")
    return rows


def translate_batch(rows, translator, *, output, metadata_output=None, cache_dir=None, workers=4):
    """Deduplicate captions and preserve all successful calls, including on failure."""
    groups = {}
    for row in rows:
        key = translator.cache_key(row["prompt"])
        groups.setdefault(key, []).append(row)
    annotations, metadata, failures = {}, {}, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {
            pool.submit(translator.translate_record, group[0]["prompt"], cache_dir=cache_dir): group
            for group in groups.values()
        }
        for future in as_completed(pending):
            group = pending[future]
            try:
                record = future.result()
            except CancelledError:
                continue
            except Exception as error:
                failures.append((group[0]["id"], error))
                if len(failures) == 1:
                    detail = str(error) if isinstance(error, TranslationError) else type(error).__name__
                    print(f"Translation failed at {group[0]['id']}: {detail}. Saving successful in-flight calls before stopping.",
                          file=sys.stderr, flush=True)
                # Running calls may still succeed. Drain them and save their results.
                for other in pending:
                    other.cancel()
                continue
            for row in group:
                annotations[row["id"]] = record["translation"]
                metadata[row["id"]] = record["metadata"]
            # Completion order must not change dataset item ordering.
            _atomic_json(output, {row["id"]: annotations[row["id"]] for row in rows if row["id"] in annotations})
            if metadata_output:
                _atomic_json(metadata_output, {row["id"]: metadata[row["id"]] for row in rows if row["id"] in metadata})
            print(f"Translated {len(annotations)}/{len(rows)}: {group[0]['id']}", file=sys.stderr, flush=True)
    if failures:
        item_id, error = failures[0]
        detail = str(error) if isinstance(error, TranslationError) else type(error).__name__
        raise TranslationError(f"Batch failed at {item_id}; saved {len(annotations)}/{len(rows)} entries. {detail}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="Human camera description for one shot")
    source.add_argument("--input", type=Path, help="JSON array or JSONL with {id, prompt} rows")
    parser.add_argument("--output", type=Path, help="Write conditioning JSON; batch outputs are keyed by input id")
    parser.add_argument("--metadata-output", type=Path, help="Write translation provenance separately")
    parser.add_argument("--model", help="Gateway provider/model ID; default AI_GATEWAY_MODEL or zai/glm-5.3-flash")
    parser.add_argument("--env-file", type=Path, default=Path(__file__).resolve().parents[1] / ".env")
    parser.add_argument("--cache-dir", type=Path, help="Optional resumable cache, keyed by prompt/model/schema/settings")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent batch requests (default 4)")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--validation-retries", type=int, default=2,
                        help="Shared retries for invalid JSON/schema or truncated output (default 2)")
    parser.add_argument("--reasoning-effort", choices=("low", "high", "max"), default="low",
                        help="Gateway GLM reasoning effort (default low for extraction)")
    parser.add_argument("--response-format", choices=("json_schema", "text"), default="json_schema")
    args = parser.parse_args(argv)
    if args.input and not args.output:
        parser.error("--input requires --output")
    if args.workers < 1:
        parser.error("--workers must be positive")
    try:
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=False)
        translator = GatewayTranslator(model=args.model, timeout=args.timeout,
                                       response_format=args.response_format, max_tokens=args.max_tokens,
                                       validation_retries=args.validation_retries,
                                       reasoning_effort=args.reasoning_effort)
        if args.input:
            rows = _records(args.input)
            translate_batch(rows, translator, output=args.output, metadata_output=args.metadata_output,
                            cache_dir=args.cache_dir, workers=args.workers)
        else:
            record = translator.translate_record(args.prompt, cache_dir=args.cache_dir)
            if args.output:
                _atomic_json(args.output, record["translation"])
            else:
                print(json.dumps(record["translation"], ensure_ascii=False, indent=2))
            if args.metadata_output:
                _atomic_json(args.metadata_output, record["metadata"])
        return 0
    except (TranslationError, OSError, ValueError, ImportError) as error:
        print(f"Translation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
