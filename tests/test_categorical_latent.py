"""The categorical bottleneck keeps 93 bounded values and no learned table."""
import math
import unittest

import torch

from models.categorical_latent import (
    CATEGORY_VALUES, FIELD_COUNT, FIELD_NAMES, FIELD_WIDTHS, SEMANTIC_DIM,
    TOKEN_WIDTH, CategoricalLatentCodec, categorical_presence, fixed_field_codes,
    fixed_query_codes, fixed_sinusoidal_codes, from_ids, get_targets,
    ids_from_batch_parameters, masked_category_logits, masked_softmax,
    valid_category_mask, validate_semantic,
)


class TestCategoricalLatent(unittest.TestCase):
    def test_schema_has_93_valid_values_and_explicit_native_categories(self):
        valid = valid_category_mask()
        self.assertEqual(valid.shape, (10, 21))
        self.assertEqual(valid.dtype, torch.bool)
        self.assertEqual(int(valid.sum()), 93)
        self.assertEqual(int((~valid).sum()), 117)
        self.assertEqual(SEMANTIC_DIM, 93)
        self.assertEqual(tuple(map(len, CATEGORY_VALUES)), FIELD_WIDTHS)
        self.assertEqual(CATEGORY_VALUES[4][19:], ("nativePan", "nativeOrbit"))

    def test_every_valid_category_survives_one_hot_encoding(self):
        ids = torch.arange(TOKEN_WIDTH)[:, None].expand(-1, FIELD_COUNT).clone()
        widths = torch.tensor(FIELD_WIDTHS)[None, :]
        ids[ids >= widths] = -1
        semantic, presence = from_ids(ids)
        validate_semantic(semantic, presence)

        self.assertEqual(semantic.shape, (10, 21, 21))
        self.assertTrue(torch.equal(presence, ids >= 0))
        self.assertTrue(torch.equal(categorical_presence(semantic), presence))
        self.assertTrue(torch.equal(semantic.sum(-1).T, presence.float()))
        predictions = semantic.argmax(-1).T
        self.assertTrue(torch.equal(predictions[presence], ids[presence]))
        self.assertEqual(float(semantic[4, 19, 19]), 1.0)
        self.assertEqual(float(semantic[4, 20, 20]), 1.0)
        self.assertTrue(torch.equal(semantic[~valid_category_mask()[:, None, :].expand_as(semantic)],
                                    torch.zeros(117 * TOKEN_WIDTH)))

    def test_missing_fields_are_exact_zero_without_changing_other_fields(self):
        ids = torch.zeros((2, 10), dtype=torch.long)
        ids[0, 4] = -1
        ids[1] = -1
        semantic, presence = from_ids(ids, dtype=torch.float64)
        self.assertEqual(semantic.dtype, torch.float64)
        self.assertFalse(presence[0, 4])
        self.assertFalse(presence[1].any())
        self.assertEqual(int(torch.count_nonzero(semantic[:, 1])), 0)
        self.assertEqual(int(torch.count_nonzero(semantic[4, 0])), 0)
        self.assertEqual(int(torch.count_nonzero(semantic[:, 0])), 9)
        validate_semantic(semantic, presence)
        validate_semantic(semantic)

    def test_invalid_id_shapes_types_and_field_ranges_are_rejected(self):
        for shape in ((2, 9), (2, 46), (10,), (1, 10, 1)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                from_ids(torch.zeros(shape, dtype=torch.long))
        for dtype in (torch.float32, torch.bool, torch.uint8):
            with self.subTest(dtype=dtype), self.assertRaises(TypeError):
                from_ids(torch.zeros((1, 10), dtype=dtype))
        for field, value in ((0, 5), (4, 21), (5, 4), (9, 13), (0, -2)):
            ids = torch.zeros((1, 10), dtype=torch.long)
            ids[0, field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                from_ids(ids)
        with self.assertRaises(TypeError):
            from_ids(torch.zeros((1, 10), dtype=torch.long), dtype=torch.int64)

    def test_softmax_masks_padding_before_activation_and_blocks_its_gradient(self):
        valid = valid_category_mask()[:, None, :].expand(10, 2, 21)
        logits = torch.arange(10 * 2 * 21, dtype=torch.float64).reshape(10, 2, 21) / 100
        logits[~valid] = float("nan")
        logits.requires_grad_()
        masked = masked_category_logits(logits)
        semantic = masked_softmax(logits)

        self.assertTrue(torch.isneginf(masked[~valid]).all())
        self.assertTrue(torch.isnan(logits[~valid]).all())
        self.assertTrue(torch.isfinite(semantic).all())
        self.assertEqual(int(torch.count_nonzero(semantic[~valid])), 0)
        validate_semantic(semantic, torch.ones((2, 10), dtype=torch.bool))
        weights = torch.arange(21, dtype=semantic.dtype)[None, None, :]
        (semantic * weights).sum().backward()
        self.assertEqual(int(torch.count_nonzero(logits.grad[~valid])), 0)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(int(torch.count_nonzero(logits.grad[valid])), 0)

    def test_semantic_validator_rejects_nonfinite_out_of_bounds_and_bad_sums(self):
        original, presence = from_ids(torch.zeros((1, 10), dtype=torch.long))
        for name, field, category, value in (
            ("nan", 0, 0, float("nan")), ("negative", 0, 1, -1e-8),
            ("above_one", 0, 0, 1.0001), ("padding", 0, 5, 1e-8),
            ("bad_sum", 0, 0, 0.9),
        ):
            broken = original.clone()
            broken[field, 0, category] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_semantic(broken, presence)
        for bad_presence in (torch.zeros_like(presence), torch.ones((1, 9), dtype=torch.bool)):
            with self.assertRaises(ValueError):
                validate_semantic(original, bad_presence)
        with self.assertRaises(TypeError):
            validate_semantic(original, presence.float())
        with self.assertRaises(ValueError):
            validate_semantic(original.transpose(0, 1), presence)
        with self.assertRaises(TypeError):
            masked_softmax(torch.zeros((10, 1, 21), dtype=torch.long))
        with self.assertRaises(ValueError):
            masked_softmax(torch.zeros((1, 10, 21)))

    def test_metadata_uses_only_first_ten_ids_ignoring_raw_values_and_vectors(self):
        class UnusableVector:
            def __array__(self, *args, **kwargs):
                raise AssertionError("CLIP vectors must never be consulted")

        rows = []
        for movement in (19, 20):
            ids = [0] * 10
            ids[4], ids[5] = movement, -1
            rows.append([
                (field, "raw-value-is-not-reinterpreted", value, UnusableVector())
                for field, value in zip(FIELD_NAMES, ids)
            ] + [("simulation-entry-is-ignored", object(), object(), object())] * 36)
        ids = ids_from_batch_parameters(rows)
        self.assertEqual(ids[:, 4].tolist(), [19, 20])
        self.assertEqual(ids[:, 5].tolist(), [-1, -1])
        expected, expected_presence = from_ids(ids)
        actual, actual_presence = get_targets(rows)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual_presence, expected_presence))
        self.assertEqual(ids_from_batch_parameters([]).shape, (0, 10))
        empty, empty_presence = get_targets([])
        validate_semantic(empty, empty_presence)

    def test_malformed_metadata_is_rejected_instead_of_silently_remapped(self):
        row = [(field, None, 0, None) for field in FIELD_NAMES]
        with self.assertRaises(ValueError):
            ids_from_batch_parameters([row[:9]])
        broken = list(row)
        broken[0] = (FIELD_NAMES[1], None, 0, None)
        with self.assertRaises(ValueError):
            ids_from_batch_parameters([broken])
        for index in (0.0, True, "0"):
            broken = list(row)
            broken[4] = (FIELD_NAMES[4], "nativeOrbit", index, None)
            with self.subTest(index=index), self.assertRaises(TypeError):
                ids_from_batch_parameters([broken])

    def test_fixed_codes_are_deterministic_odd_width_safe_and_parameter_free(self):
        state = torch.get_rng_state().clone()
        field_codes = fixed_field_codes()
        query_codes = fixed_query_codes()
        codec = CategoricalLatentCodec()
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        self.assertEqual(field_codes.shape, (10, 21))
        self.assertEqual(query_codes.shape, (10, 256))
        self.assertEqual(sum(parameter.numel() for parameter in codec.parameters()), 0)
        self.assertTrue(torch.equal(codec.valid_mask, valid_category_mask()))
        self.assertTrue(torch.equal(field_codes, fixed_sinusoidal_codes(10, 21)))
        self.assertTrue(torch.equal(field_codes[0, 0::2], torch.zeros(11)))
        self.assertTrue(torch.equal(field_codes[0, 1::2], torch.ones(10)))
        self.assertFalse(torch.equal(field_codes[0], field_codes[1]))
        expected = torch.sin(
            torch.arange(10).float() * torch.exp(torch.tensor(20.) * (-math.log(10000.) / 21))
        )
        self.assertTrue(torch.equal(field_codes[:, -1], expected))
        semantic, presence = codec(torch.zeros((1, 10), dtype=torch.long))
        before = semantic.clone()
        prepared_attention = semantic + field_codes[:, None, :]
        self.assertTrue(torch.equal(semantic, before))
        self.assertFalse(torch.equal(prepared_attention, semantic))
        codec.validate_semantic(semantic, presence)


if __name__ == "__main__":
    unittest.main()
