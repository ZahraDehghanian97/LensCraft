"""Separate simulated physical truth from the information exposed to a model.

The ET profile is a declared information contract, not an empirical fit to an
ET evaluation cohort. Its construction never reads a target camera pose.
"""

from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Real
from typing import Mapping, Optional

import torch


@dataclass(frozen=True)
class ModelObservationPolicy:
    profile: str
    default_subject_dimensions: tuple
    movement_speed_available: bool = True

    @classmethod
    def from_manifest(cls, manifest: Optional[Mapping]):
        if manifest is None or "modelObservation" not in manifest:
            return None
        declaration = manifest["modelObservation"]
        if not isinstance(declaration, Mapping):
            raise ValueError("manifest.modelObservation must be an object")
        expected = {
            "profile": "et_compatible_v1",
            "subjectOrientation": "unavailable",
            "subjectDimensions": "default",
            "simulationInstructions": "unavailable",
            "captionPolicy": "motion_only",
        }
        for key, value in expected.items():
            if declaration.get(key) != value:
                raise ValueError(f"Unsupported modelObservation {key}: {declaration.get(key)!r}; "
                                 f"expected {value!r}")
        dimensions = declaration.get("defaultSubjectDimensions")
        if (not isinstance(dimensions, (list, tuple)) or len(dimensions) != 3
                or any(isinstance(value, bool) or not isinstance(value, Real)
                       or not math.isfinite(value) or value <= 0 for value in dimensions)):
            raise ValueError("modelObservation defaultSubjectDimensions must contain three "
                             "finite positive numbers")
        dimensions = tuple(float(value) for value in dimensions)
        if dimensions != (0.5, 1.7, 0.3):
            raise ValueError("et_compatible_v1 defaultSubjectDimensions must match the "
                             "ET adapter: [0.5, 1.7, 0.3]")
        # Older declarations expose an actually supplied speed annotation.
        # New ET-like generators can instead declare their speed enum to be a
        # diagnostic placeholder, which must never become a model condition.
        speed_available = "movementSpeed" not in declaration
        if not speed_available and declaration["movementSpeed"] != "unavailable":
            raise ValueError("Unsupported modelObservation movementSpeed; expected 'unavailable'")
        return cls(profile=expected["profile"], default_subject_dimensions=dimensions,
                   movement_speed_available=speed_available)

    def subject_condition(self, subject_trajectory: torch.Tensor, subject_volume: torch.Tensor):
        """Retain measured XYZ while marking body orientation/size as unknown.

        Euler zero and the dimensions are placeholders in physical SIM units,
        not measurements. The caller supplies explicit availability metadata.
        """
        if subject_trajectory.ndim != 2 or subject_trajectory.shape[-1] != 6:
            raise ValueError("Model observation requires subject poses [frames, 6]")
        if subject_volume.shape != (1, 3):
            raise ValueError("Model observation requires subject dimensions [1, 3]")
        visible_subject = subject_trajectory.clone()
        visible_subject[..., 3:] = 0.
        visible_volume = subject_volume.new_tensor(self.default_subject_dimensions).reshape(1, 3)
        return visible_subject, visible_volume

    def caption_condition(self, prompt: Mapping, instruction: Mapping):
        """Expose supplied motion type/speed; omit simulator programs/endpoints."""
        if not isinstance(prompt, Mapping) or not isinstance(instruction, Mapping):
            raise ValueError("Model observation requires prompt and instruction objects")
        movement = prompt.get("movement", {})
        if not isinstance(movement, Mapping):
            raise ValueError("Model observation prompt movement must be an object")
        available_keys = ("type", "speed") if self.movement_speed_available else ("type",)
        visible_movement = {key: deepcopy(movement[key]) for key in available_keys if key in movement}
        return {"movement": visible_movement}, {}

    def observed_motion_text(self, prompt: Mapping):
        """Read the generator's motion caption without inspecting camera GT."""
        if "observedMotionText" not in prompt:
            return None
        text = prompt["observedMotionText"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("ET observation observedMotionText must be a non-empty string")
        return text

    def provenance(self):
        return {
            "profile": self.profile,
            "subject_orientation_semantics": "unknown; zero Euler is a placeholder",
            "subject_dimensions_semantics": "unknown; ET adapter default dimensions",
            "caption_policy": "motion_only",
            "movement_speed_available": self.movement_speed_available,
            "movement_speed_semantics": (
                "supplied annotation" if self.movement_speed_available
                else "unknown; archived speed enum is a diagnostic placeholder"
            ),
            "subject_availability_encoding": "metadata_only; core model has no learned availability cue",
        }
