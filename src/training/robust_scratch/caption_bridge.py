#!/usr/bin/env python3
"""Deterministic CCDM template captions -> LensCraft's 46 conditioning slots.

This candidate reads caption text only. It never reads camera values or changes
historical annotations. Missing/ambiguous facts remain None, with diagnostics.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any


BRIDGE_VERSION = 1
PAPER_URL = "https://arxiv.org/html/2402.16143v1"
UPSTREAM_URL = "https://github.com/jianghd1996/Camera-control/tree/main/%5B2024%5D%5BEG%5DText%2Bkeyframe"

PAPER_AUDIT = {
    "source": PAPER_URL,
    "table_1_text_only": {"FID": 48.25, "R_Prec_percent": 97.78,
                           "diversity": 61.93, "multimodality": 47.75},
    "native_evaluation": "60 frames; six-motion classifier features; 12,000 generated sequences",
    "paper_reproduction_limit": "Exact classifier split/checkpoint and 60-frame generator are not supplied by the local release. Current simulation-trained CLaTr/FCD values are not paper FID.",
    "released_baseline_limit": "Released main.py trains on the complete cam/info corpus. No original generator train/test IDs are supplied; baseline self-evaluation is not a verified held-out test.",
    "representation": "Subject-relative XYZ and normalized head-target screen XY; no roll. The synthetic dataset uses a default 45-degree FOV.",
    "parser_semantics": "Native pan can encode subject-relative translation from a globally fixed camera. Pan motion aliases and world-space locking constraints are masked. Orbit handedness is masked; steady/normal/breathy do not establish velocity easing.",
}

VIEWS = {
    "front": "front", "back": "back", "left": "left", "right": "right",
    "left front": "threeQuarterFrontLeft", "right front": "threeQuarterFrontRight",
    "left back": "threeQuarterBackLeft", "right back": "threeQuarterBackRight",
}
SHOTS = {
    "extreme close shot": "extremeCloseUp", "extreme close-up shot": "extremeCloseUp",
    "close shot": "closeUp", "close-up shot": "closeUp",
    "medium close-up shot": "mediumCloseUp", "medium close shot": "mediumCloseUp",
    "medium shot": "mediumShot", "full shot": "fullShot", "long shot": "longShot",
    "very long shot": "veryLongShot", "extreme long shot": "extremeLongShot",
}
ANGLES = {"low angle": "low", "eye level": "eye", "high angle": "high",
          "overhead": "overhead", "bird's eye": "birdsEye", "birds eye": "birdsEye"}
FRAMING = {
    "up left": "topLeft", "up center": "top", "up right": "topRight",
    "top left": "topLeft", "top center": "top", "top right": "topRight",
    "middle left": "left", "middle center": "center", "middle right": "right",
    "bottom left": "bottomLeft", "bottom center": "bottom", "bottom right": "bottomRight",
}
MOVEMENTS = {
    "the camera is static": ("static", "interpolation", None, None, "static"),
    "the camera pushes in to the character": ("dollyIn", "simple", "transition", "forward", "push_in"),
    "the camera pushes in": ("dollyIn", "simple", "transition", "forward", "push_in"),
    "the camera pulls in to the character": ("dollyIn", "simple", "transition", "forward", "push_in"),
    "the camera zooms in to the character": ("dollyIn", "simple", "transition", "forward", "push_in"),
    "the camera zooms in": ("dollyIn", "simple", "transition", "forward", "push_in"),
    "the camera pulls out to the character": ("dollyOut", "simple", "transition", "backward", "pull_out"),
    "the camera pulls out": ("dollyOut", "simple", "transition", "backward", "pull_out"),
    "the camera zooms out to the character": ("dollyOut", "simple", "transition", "backward", "pull_out"),
    "the camera zooms out": ("dollyOut", "simple", "transition", "backward", "pull_out"),
    "the camera booms up": ("pedestalUp", "simple", "transition", "up", "boom_up"),
    "the camera booms down": ("pedestalDown", "simple", "transition", "down", "boom_down"),
    # A world-static tracking pan need not be a fixed-position local pan.
    "the camera pans to the character": (None, None, None, None, "pan"),
    # The family is unambiguous; left/right handedness is deliberately absent.
    "the camera rotates around the character": (None, "simple", "arc", None, "orbit"),
}


def _endpoint() -> dict:
    return dict(cameraAngle=None, shotSize=None, subjectView=None, subjectFraming=None)


def empty_conditioning() -> dict:
    config = dict(cameraAngle=None, shotSize=None, subjectView=None,
                  subjectFraming=dict(position=None, dutchAngleScale=None))
    return {
        "cinematographyPrompts": [{"initial": _endpoint(), "movement": dict(type=None, speed=None), "final": _endpoint()}],
        "simulationInstructions": [{
            "setup": {"config": deepcopy(config), "kind": None},
            "dynamic": dict(type=None, easing=None, randomness=None,
                            complementSetup=deepcopy(config), subjectAwareInterpolation=None,
                            scale=None, direction=None, movementMode=None),
            "constraints": dict(allFramesVisibility=None, staticDistance=None, staticCameraSubjectRotation=None,
                                lockedMovement={key: None for key in ("left", "right", "up", "down", "forward", "backward")},
                                lockedRotation={key: None for key in ("left", "right", "up", "down", "rollClockwise", "rollNonClockwise")},
                                maxAccelerate=None, maxSpeed=None, importance=None),
        }],
    }


def _flatten(value: dict, prefix: str = "") -> dict[str, Any]:
    result = {}
    for key, item in value.items():
        path = f"{prefix}_{key}" if prefix else key
        if isinstance(item, dict):
            result.update(_flatten(item, path))
        else:
            result[path] = item
    return result


def slot_values(conditioning: dict) -> list:
    """Return 10 prompt leaves then 36 instruction leaves in encoder order."""
    return [*(_flatten(conditioning["cinematographyPrompts"][0]).values()),
            *(_flatten(conditioning["simulationInstructions"][0]).values())]


def _property(text: str) -> tuple[str, str] | None:
    text = text.removesuffix(" view").strip()
    for field, values in (("subjectView", VIEWS), ("shotSize", SHOTS), ("cameraAngle", ANGLES)):
        if text in values:
            return field, values[text]
    return None


def parse_with_provenance(text: str) -> tuple[dict, dict]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("A nonempty native caption string is required")
    conditioning = empty_conditioning()
    prompt = conditioning["cinematographyPrompts"][0]
    instruction = conditioning["simulationInstructions"][0]
    evidence: dict[tuple[str, str], list[tuple[int, str]]] = {}
    motions, families, known, unknown, masked = [], [], [], [], []
    randomness = []
    orbit_directions = []

    def assign(field: str, value: str, endpoints=("initial", "final"), priority=1):
        for endpoint in endpoints:
            evidence.setdefault((endpoint, field), []).append((priority, value))

    sentences = [sentence.strip().lower() for sentence in re.split(r"[.!?]+", text) if sentence.strip()]
    for sentence in sentences:
        handled = True
        if sentence in MOVEMENTS:
            motions.append(MOVEMENTS[sentence][:4])
            families.append(MOVEMENTS[sentence][4])
        elif sentence in ("the camera is steady", "the camera moves normal", "the camera is breathy"):
            # Steadiness is not an easing or numeric speed; breathy has no
            # matching current enum and does not prove hand-held acquisition.
            masked.append(sentence + ": no velocity/easing inference")
        elif sentence == "the camera is shaky":
            randomness.append("shaky")
        elif sentence in ("the camera moves in clockwise direction", "the camera moves in anti-clockwise direction"):
            orbit_directions.append(sentence)
            masked.append(sentence + ": local arc handedness unverified")
        elif (match := re.fullmatch(r"the character is at the (.+) of the screen", sentence)):
            value = FRAMING.get(match[1])
            if value is None:
                handled = False
            else:
                assign("subjectFraming", value)
        elif (match := re.fullmatch(r"the camera (?:moves|switches) from (.+) to (.+)", sentence)):
            start, end = _property(match[1]), _property(match[2])
            if start is None or end is None or start[0] != end[0]:
                handled = False
            else:
                assign(*start, endpoints=("initial",), priority=3)
                assign(*end, endpoints=("final",), priority=3)
        elif (match := re.fullmatch(r"the camera switches to (.+)", sentence)):
            prop = _property(match[1])
            if prop is None:
                handled = False
            else:
                assign(*prop, endpoints=("final",), priority=2)
        elif (match := re.fullmatch(r"the camera (?:shoots in|shoots at|moves at) (.+)", sentence)):
            prop = _property(match[1])
            if prop is None:
                handled = False
            else:
                assign(*prop)
        else:
            handled = False
        (known if handled else unknown).append(sentence)

    conflicts = []
    for (endpoint, field), candidates in evidence.items():
        priority = max(item[0] for item in candidates)
        values = {value for rank, value in candidates if rank == priority}
        if len(values) == 1:
            prompt[endpoint][field] = next(iter(values))
        else:
            conflicts.append(f"{endpoint}.{field}")

    # Unknown sentences may describe another motion. Keep recognized endpoints,
    # but fail closed on the principal movement instead of silently ignoring it.
    if motions and len(set(motions)) == 1 and not unknown:
        move_type, dynamic_type, mode, direction = motions[0]
        prompt["movement"]["type"] = move_type
        instruction["dynamic"].update(type=dynamic_type, movementMode=mode, direction=direction)
    elif motions:
        masked.append("principal movement: conflicting or unsupported caption sentences")
    if "pan" in families:
        masked.append("native pan: world-fixed and subject-relative movement differ")
    if randomness:
        instruction["dynamic"]["randomness"] = "shaky"

    initial_known = any(value is not None for value in prompt["initial"].values())
    final_known = any(value is not None for value in prompt["final"].values())
    setup_endpoint = "initial" if initial_known else "final" if final_known else None
    if setup_endpoint:
        instruction["setup"]["kind"] = "init" if setup_endpoint == "initial" else "end"
        setup = prompt[setup_endpoint]
        complement = prompt["final"] if setup_endpoint == "initial" else prompt["initial"]
        for field in ("cameraAngle", "shotSize", "subjectView"):
            instruction["setup"]["config"][field] = setup[field]
            instruction["dynamic"]["complementSetup"][field] = complement[field]
        instruction["setup"]["config"]["subjectFraming"]["position"] = setup["subjectFraming"]
        instruction["dynamic"]["complementSetup"]["subjectFraming"]["position"] = complement["subjectFraming"]

    values = slot_values(conditioning)
    assert len(values) == 46
    provenance = {
        "bridge_version": BRIDGE_VERSION,
        "source_prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "source": "native caption text only; no camera-derived labels or API",
        "sentence_count": len(sentences), "recognized_sentences": len(known),
        "unknown_sentences": unknown, "conflicting_endpoint_fields": conflicts,
        "motion_families": sorted(set(families)), "masked_semantics": masked,
        "native_orbit_direction_text": orbit_directions,
        "slot_count": len(values), "present_slots": sum(value is not None for value in values),
        "all_sentences_recognized": not unknown,
    }
    return conditioning, provenance


def parse_caption(text: str) -> dict:
    return parse_with_provenance(text)[0]


def bridge_rows(rows: list[dict], original_annotations: list[dict] | None = None) -> tuple[list, dict, dict]:
    annotations, records = [], {}
    originals = {row["custom_id"]: row for row in original_annotations or []}
    changed = Counter()
    for row in rows:
        item_id = row.get("id", row.get("custom_id"))
        caption = row.get("prompt", row.get("source_prompt"))
        if not isinstance(item_id, str) or not item_id or item_id in records:
            raise ValueError("Caption IDs must be nonempty and unique")
        conditioning, record = parse_with_provenance(caption)
        annotations.append({"custom_id": item_id, "source_prompt": caption, **conditioning})
        records[item_id] = record
        original = originals.get(item_id)
        if original:
            if original.get("source_prompt") != caption:
                raise ValueError(f"Original caption identity mismatch: {item_id}")
            for name in ("cinematographyPrompts", "simulationInstructions"):
                prior = _flatten(original[name][0])
                for path, value in _flatten(conditioning[name][0]).items():
                    if prior.get(path) != value:
                        changed[f"{name}.{path}"] += 1
    families = Counter(family for record in records.values() for family in record["motion_families"])
    unknown = Counter(sentence for record in records.values() for sentence in record["unknown_sentences"])
    audit = {
        "bridge_version": BRIDGE_VERSION, "samples": len(rows),
        "fully_recognized_captions": sum(record["all_sentences_recognized"] for record in records.values()),
        "fully_recognized_fraction": sum(record["all_sentences_recognized"] for record in records.values()) / max(1, len(rows)),
        "unknown_sentence_counts": dict(unknown), "motion_family_counts": dict(families),
        "endpoint_conflict_captions": sum(bool(record["conflicting_endpoint_fields"]) for record in records.values()),
        "compared_original_annotations": sum(row["custom_id"] in originals for row in annotations),
        "changed_leaf_counts_vs_original": dict(changed),
        "semantic_accuracy_note": "Template recognition coverage is not measured generation quality; inference needs a paired validation run.",
        "paper_protocol": PAPER_AUDIT,
    }
    return annotations, records, audit


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_new(path: Path, value: Any):
    content = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    if path.exists() and path.read_text() != content:
        raise ValueError(f"Refusing to replace a different experiment artifact: {path}")
    path.write_text(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--original-annotations", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.prompts.read_text())
    original = json.loads(args.original_annotations.read_text()) if args.original_annotations else None
    annotations, records, audit = bridge_rows(rows, original)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Historical files must never be an output destination, including when the
    # caller mistakenly supplies their original parent directory.
    if args.original_annotations and (args.output_dir / "annotations.json").resolve() == args.original_annotations.resolve():
        raise ValueError("Output must not replace the original annotations")
    manifest = {
        "bridge_version": BRIDGE_VERSION, "bridge_sha256": _sha256(Path(__file__)),
        "prompts_path": str(args.prompts.resolve()), "prompts_sha256": _sha256(args.prompts),
        "original_annotations_sha256": _sha256(args.original_annotations) if args.original_annotations else None,
        "input_policy": "Caption text only. No camera arrays, APIs, test score fitting or historical mutation.",
        "semantic_policy": PAPER_AUDIT["parser_semantics"],
        "paper_source": PAPER_URL, "upstream_source": UPSTREAM_URL,
        "schema_slot_count": 46,
    }
    _save_new(args.output_dir / "annotations.json", annotations)
    _save_new(args.output_dir / "bridge_provenance.json", records)
    _save_new(args.output_dir / "bridge_audit.json", audit)
    manifest["annotations_sha256"] = _sha256(args.output_dir / "annotations.json")
    _save_new(args.output_dir / "bridge_manifest.json", manifest)
    print(json.dumps({key: audit[key] for key in ("samples", "fully_recognized_captions", "fully_recognized_fraction", "endpoint_conflict_captions")}))


if __name__ == "__main__":
    main()
