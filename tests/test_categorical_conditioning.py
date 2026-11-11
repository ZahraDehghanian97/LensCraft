"""Every model input uses the same ten-field categorical coordinate system."""
import unittest

import torch

from data.categorical_conditioning import (
    FIELD_NAMES, FIELD_VALUES, FIELD_WIDTHS, cinematography_ids_from_batch,
    conditioning_from_batch, conditioning_presence_from_batch,
    film_ids_from_prompt, film_parameters_from_prompt, conditioning_from_prompt, metadata_from_prompt,
)
from models.categorical_latent import CATEGORY_VALUES, from_ids, validate_semantic


class TestCategoricalConditioning(unittest.TestCase):
    def test_dataset_parser_uses_exact_native_categories_and_ignores_simulator_fields(self):
        self.assertEqual(FIELD_VALUES, CATEGORY_VALUES)
        self.assertEqual(sum(FIELD_WIDTHS), 93)
        for raw, expected in (("nativePan", 19), ("nativeOrbit", 20)):
            prompt = {"initial": {"shotSize": "closeUp"},
                      "movement": {"type": raw, "speed": "not-a-category"},
                      "constraints": {"maxSpeed": 100}}
            parameters = film_parameters_from_prompt(prompt)
            ids = film_ids_from_prompt(prompt)
            self.assertEqual(tuple(row[0] for row in parameters), FIELD_NAMES)
            self.assertEqual(ids.shape, (10,))
            self.assertEqual(ids[1].item(), 1)
            self.assertEqual(ids[4].item(), expected)
            self.assertEqual(ids[5].item(), -1)
            self.assertTrue((ids[[0, 2, 3, 6, 7, 8, 9]] == -1).all())

    def test_explicit_ids_take_precedence_without_mutating_metric_vectors_or_masks(self):
        ids = torch.zeros((2, 10), dtype=torch.long)
        ids[:, 4] = torch.tensor([19, 20])
        ids[1, 7] = -1
        vectors = torch.randn(10, 2, 512)
        simulation = torch.randn(36, 2, 512)
        mask = torch.ones((2, 10), dtype=torch.bool)
        mask[0, 1] = False
        batch = {"cinematography_ids": ids, "cinematography_prompt": vectors,
                 "simulation_instruction": simulation, "prompt_none_mask": mask,
                 # Invalid stale metadata is intentionally ignored when IDs exist.
                 "cinematography_prompt_parameters": [[("stale", None, -1)]]}
        original_ids, original_mask = ids.clone(), mask.clone()
        values = conditioning_from_batch(batch)
        presence = conditioning_presence_from_batch(batch)
        self.assertEqual(values.shape, (10, 2, 21))
        self.assertEqual(presence.shape, (2, 10))
        self.assertFalse(presence[0, 1])
        self.assertFalse(presence[1, 7])
        self.assertEqual(values[4, 0, 19].item(), 1)
        self.assertEqual(values[4, 1, 20].item(), 1)
        validate_semantic(values, presence)
        self.assertTrue(torch.equal(ids, original_ids))
        self.assertTrue(torch.equal(mask, original_mask))
        self.assertIs(batch["cinematography_prompt"], vectors)
        self.assertIs(batch["simulation_instruction"], simulation)
        self.assertEqual(batch["prompt_none_mask"].shape, (2, 10))

    def test_metadata_fallback_uses_declared_ids_and_ignores_clip_rows(self):
        rows = [film_parameters_from_prompt({"movement": {"type": "nativeOrbit"}})]
        # Raw strings and vector values cannot resurrect an explicitly unknown ID.
        rows[0][1] = (FIELD_NAMES[1], "closeUp", -1, torch.ones(512))
        batch = {"cinematography_prompt_parameters": rows}
        ids = cinematography_ids_from_batch(batch)
        self.assertEqual(ids[0, 4].item(), 20)
        self.assertEqual(ids[0, 1].item(), -1)
        expected, presence = from_ids(ids)
        self.assertTrue(torch.equal(conditioning_from_batch(batch), expected))
        self.assertTrue(torch.equal(conditioning_presence_from_batch(batch), presence))

    def test_all_unknown_is_zero_and_has_no_present_fields(self):
        batch = {"cinematography_ids": torch.full((3, 10), -1, dtype=torch.long)}
        values = conditioning_from_batch(batch)
        presence = conditioning_presence_from_batch(batch)
        self.assertEqual(int(values.count_nonzero()), 0)
        self.assertFalse(presence.any())
        validate_semantic(values, presence)

    def test_invalid_explicit_ids_or_metadata_cannot_fall_back_to_clip_vectors(self):
        for ids, error in ((torch.zeros(2, 46, dtype=torch.long), ValueError),
                           (torch.zeros(2, 10), TypeError),
                           (torch.full((2, 10), -2, dtype=torch.long), ValueError)):
            with self.subTest(shape=ids.shape, dtype=ids.dtype), self.assertRaises(error):
                conditioning_from_batch({"cinematography_ids": ids})
        ids = torch.zeros((1, 10), dtype=torch.long)
        ids[0, 0] = 5
        with self.assertRaises(ValueError):
            cinematography_ids_from_batch({"cinematography_ids": ids})
        with self.assertRaisesRegex(ValueError, "CLIP vectors cannot supply category IDs"):
            conditioning_from_batch({"cinematography_prompt": torch.randn(10, 1, 512)})
        rows = [film_parameters_from_prompt({})]
        rows[0][0] = ("wrong_field", None, -1, None)
        with self.assertRaises(ValueError):
            cinematography_ids_from_batch({"cinematography_prompt_parameters": rows})

    def test_prompt_api_and_raw_metadata_share_exact_ids_without_simulator_tokens(self):
        prompt = {"movement": {"type": "nativeOrbit"}, "initial": {"shotSize": "closeUp"}}
        values, presence, parameters = conditioning_from_prompt(prompt)
        metadata = metadata_from_prompt(prompt, {"constraints": {"maxSpeed": 3.}})
        self.assertEqual(values.shape, (10, 1, 21))
        self.assertEqual(presence.shape, (1, 10))
        self.assertEqual(parameters, metadata["cinematography_prompt_parameters"])
        self.assertEqual(len(metadata["simulation_instruction_parameters"]), 36)
        self.assertEqual(metadata["structured_presence_mask"].shape, (46,))
        self.assertEqual(int(metadata["structured_presence_mask"].sum()), 3)
        self.assertTrue(all(entry[3] is None for entry in metadata["simulation_instruction_parameters"]))
        validate_semantic(values, presence)

    def test_raw_simulator_constraints_remain_available_without_model_semantics(self):
        from data.simulation.constants import NumericFeature, simulation_struct

        def known_values(structure):
            return {name: (known_values(kind) if isinstance(kind, list) else
                           False if kind is bool else 3.5 if isinstance(kind, NumericFeature)
                           else next(iter(kind)).value)
                    for name, kind in structure}

        metadata = metadata_from_prompt({}, known_values(simulation_struct))
        self.assertEqual(metadata["cinematography_ids"].tolist(), [-1] * 10)
        self.assertFalse(metadata["prompt_none_mask"].any())
        self.assertFalse(metadata["structured_presence_mask"][:10].any())
        self.assertTrue(metadata["structured_presence_mask"][10:].all())
        raw_fields = dict((name, (raw, index)) for name, raw, index, _ in
                          metadata["simulation_instruction_parameters"])
        self.assertEqual(raw_fields["constraints_maxSpeed"], (3.5, 0))
        false_constraints = [entry for entry in metadata["simulation_instruction_parameters"]
                             if entry[1] is False]
        self.assertEqual(len(false_constraints), 16)
        self.assertTrue(all(entry[2] == 0 for entry in false_constraints))
        batch = {"cinematography_ids": metadata["cinematography_ids"].unsqueeze(0)}
        self.assertEqual(int(conditioning_from_batch(batch).count_nonzero()), 0)

    def test_structured_batch_api_never_uses_legacy_embedding_arrays(self):
        from data.simulation.utils import structured_conditioning_from_batch

        ids = torch.zeros(2, 10, dtype=torch.long)
        batch = {"cinematography_ids": ids,
                 "cinematography_prompt": torch.randn(10, 2, 512),
                 "simulation_instruction": torch.randn(36, 2, 512)}
        self.assertTrue(torch.equal(structured_conditioning_from_batch(batch), conditioning_from_batch(batch)))
        self.assertEqual(structured_conditioning_from_batch(batch).shape, (10, 2, 21))
        with self.assertRaises(ValueError):
            structured_conditioning_from_batch({"caption_feat": torch.randn(2, 512)})


if __name__ == "__main__":
    unittest.main()
