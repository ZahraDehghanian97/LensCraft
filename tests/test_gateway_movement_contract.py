"""Gateway aliases must agree with the actual simulation motion vocabulary."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from annotator.gateway import (
    GatewayTranslator, TranslationError, validate_translation, validate_translation_schema,
)
from data.simulation.constants import cinematography_struct, simulation_struct


def empty_struct(struct):
    return {name: empty_struct(kind) if isinstance(kind, list) else None
            for name, kind in struct}


def translation(movement=None, dynamic_type=None, mode=None, direction=None):
    result = {
        "cinematographyPrompts": [empty_struct(cinematography_struct)],
        "simulationInstructions": [empty_struct(simulation_struct)],
    }
    result["cinematographyPrompts"][0]["movement"]["type"] = movement
    result["simulationInstructions"][0]["dynamic"].update(
        type=dynamic_type, movementMode=mode, direction=direction,
    )
    return result


class GatewayMovementContractTests(unittest.TestCase):
    def test_stored_legacy_aliases_are_schema_valid_and_preserved(self):
        for movement in ("dollyIn", "dollyOut", "arcLeft", "arcRight", "craneUp", "craneDown"):
            with self.subTest(movement=movement):
                value = translation(movement, "interpolation")
                before = copy.deepcopy(value)
                self.assertIs(validate_translation_schema(value), value)
                self.assertEqual(value, before)
                with self.assertRaisesRegex(TranslationError, r"dynamic\.type.*contradicts"):
                    validate_translation(value)
                self.assertEqual(value, before)

    def test_dolly_arc_crane_reject_old_interpolation_contract(self):
        for movement in ("dollyIn", "dollyOut", "arcLeft", "arcRight", "craneUp", "craneDown"):
            with self.subTest(movement=movement):
                with self.assertRaisesRegex(TranslationError, r"dynamic\.type.*contradicts"):
                    validate_translation(translation(movement, "interpolation"))

    def test_dolly_arc_crane_accept_training_aliases_and_reject_opposite_direction(self):
        rows = (
            ("dollyIn", "transition", "forward", "backward"),
            ("dollyOut", "transition", "backward", "forward"),
            ("arcLeft", "arc", "left", "right"),
            ("arcRight", "arc", "right", "left"),
            ("craneUp", "crane", "up", "down"),
            ("craneDown", "crane", "down", "up"),
        )
        for movement, mode, direction, opposite in rows:
            with self.subTest(movement=movement):
                value = translation(movement, "simple", mode, direction)
                self.assertIs(validate_translation(value), value)
                with self.assertRaisesRegex(TranslationError, r"dynamic\.direction.*contradicts"):
                    validate_translation(translation(movement, "simple", mode, opposite))

    def test_existing_camera_local_and_interpolated_movements_remain_valid(self):
        for args in (
            ("panLeft", "simple", "rotation", "left"),
            ("panRight", "simple", "rotation", "right"),
            ("tiltUp", "simple", "rotation", "up"),
            ("tiltDown", "simple", "rotation", "down"),
            ("truckLeft", "simple", "transition", "left"),
            ("truckRight", "simple", "transition", "right"),
            ("pedestalUp", "simple", "transition", "up"),
            ("pedestalDown", "simple", "transition", "down"),
            ("dutchLeft", "simple", "roll", "left"),
            ("dutchRight", "simple", "roll", "right"),
            ("follow", "interpolation", None, None),
            ("track", "interpolation", None, None),
            ("static", "interpolation", None, None),
        ):
            with self.subTest(movement=args[0]):
                value = translation(*args)
                self.assertIs(validate_translation(value), value)

    def test_supplied_wrong_mode_and_inapplicable_follow_direction_are_rejected(self):
        with self.assertRaisesRegex(TranslationError, r"dynamic\.movementMode.*contradicts"):
            validate_translation(translation("dollyIn", "simple", "rotation", "forward"))
        with self.assertRaisesRegex(TranslationError, r"dynamic\.direction.*contradicts"):
            validate_translation(translation("follow", "interpolation", None, "left"))

    def test_null_and_partial_facts_are_preserved_without_invented_fields(self):
        for value in (
            translation(), translation("dollyIn"), translation("follow"), translation("static"),
            translation("dollyOut", mode="transition"),
            translation("craneUp", direction="up"),
            translation(None, "simple", "transition", "right"),
        ):
            with self.subTest(value=value):
                before = copy.deepcopy(value)
                self.assertIs(validate_translation(value), value)
                self.assertEqual(value, before)

    def test_schema_errors_still_fail_before_motion_checks(self):
        value = translation("dollyIn", "simple", "transition", "forward")
        del value["simulationInstructions"][0]["dynamic"]["direction"]
        with self.assertRaisesRegex(TranslationError, "missing fields"):
            validate_translation(value)

    def test_stored_and_fresh_validators_reject_malformed_schema(self):
        bad_values = []
        value = translation("craneUp", "interpolation")
        del value["simulationInstructions"][0]["dynamic"]["direction"]
        bad_values.append(("missing", value))
        value = translation("craneUp", "interpolation")
        value["simulationInstructions"][0]["dynamic"]["unsupported"] = None
        bad_values.append(("extra", value))
        value = translation("invented", "interpolation")
        bad_values.append(("enum", value))
        value = translation("craneUp", "interpolation")
        value["simulationInstructions"][0]["constraints"]["allFramesVisibility"] = "true"
        bad_values.append(("type", value))
        for label, number in (("nan", float("nan")), ("inf", float("inf")), ("bounds", 21.0)):
            value = translation("craneUp", "interpolation")
            value["simulationInstructions"][0]["constraints"]["maxSpeed"] = number
            bad_values.append((label, value))
        value = translation("craneUp", "interpolation")
        value["cinematographyPrompts"] = []
        bad_values.append(("shot_count", value))
        for validator in (validate_translation_schema, validate_translation):
            for label, value in bad_values:
                with self.subTest(validator=validator.__name__, problem=label):
                    with self.assertRaises(TranslationError):
                        validator(value)

    def test_gateway_cache_still_rejects_legacy_alias_conflict(self):
        requests = []

        def create(**request):
            requests.append(request)
            raise AssertionError("Invalid cache must fail before calling Gateway")

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        translator = GatewayTranslator(model="test/model", client=client)
        prompt = "The camera cranes upward."
        with tempfile.TemporaryDirectory() as cache_dir:
            key = translator.cache_key(prompt)
            path = Path(cache_dir) / f"{key}.json"
            path.write_text(json.dumps({"translation": translation("craneUp", "interpolation"),
                                        "metadata": {"cache_key": key}}))
            before = path.read_bytes()
            with self.assertRaisesRegex(TranslationError, "Invalid translation cache"):
                translator.translate_record(prompt, cache_dir=cache_dir)
            self.assertEqual(path.read_bytes(), before)
        self.assertEqual(requests, [])

    def test_gateway_retries_explicit_alias_conflict_without_filling_composition(self):
        responses = [translation("dollyOut", "interpolation"),
                     translation("dollyOut", "simple", "transition", "backward")]
        requests = []

        def create(**request):
            requests.append(request)
            value = responses[len(requests) - 1]
            message = SimpleNamespace(content=json.dumps(value), refusal=None)
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=message)],
                                   usage=None, id="fake-response", model="test/model")

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        record = GatewayTranslator(model="test/model", client=client, validation_retries=1).translate_record(
            "The camera pulls out while the character remains stationary."
        )
        self.assertEqual(len(requests), 2)
        self.assertIn("contradicts", record["metadata"]["attempts"][0]["validation_error"])
        self.assertEqual(record["translation"], responses[1])
        self.assertTrue(all(value is None for value in record["translation"]["cinematographyPrompts"][0]["initial"].values()))


if __name__ == "__main__":
    unittest.main()
