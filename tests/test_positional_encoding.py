"""Positional encoding supports odd latent widths without changing even widths."""

import math
import unittest

import torch

from models.positional_encoding import PositionalEncoding


class TestPositionalEncoding(unittest.TestCase):
    def test_width_93_forward_is_finite_and_respects_max_length(self):
        encoding = PositionalEncoding(93, dropout=0.0, max_len=17)
        inputs = torch.zeros(3, 17, 93)

        outputs = encoding(inputs)

        self.assertEqual(encoding.pe.shape, (1, 17, 93))
        self.assertEqual(outputs.shape, inputs.shape)
        self.assertTrue(torch.isfinite(outputs).all())
        self.assertTrue(torch.equal(outputs[0], outputs[2]))
        self.assertTrue(torch.equal(outputs[0, 0, 0::2], torch.zeros(47)))
        self.assertTrue(torch.equal(outputs[0, 0, 1::2], torch.ones(46)))

    def test_odd_width_keeps_last_sine_channel(self):
        encoding = PositionalEncoding(93, dropout=0.0, max_len=17)
        positions = torch.arange(17, dtype=torch.float32)
        last_frequency = torch.exp(torch.tensor(92.0) * (-math.log(10000.0) / 93))
        expected = torch.sin(positions * last_frequency)

        self.assertTrue(torch.equal(encoding.pe[0, :, 92], expected))

    def test_even_width_matches_original_reference_exactly(self):
        width, max_len = 16, 29
        positions = torch.arange(max_len).unsqueeze(1).float()
        frequencies = torch.exp(
            torch.arange(0, width, 2).float() * (-math.log(10000.0) / width)
        )
        reference = torch.zeros(max_len, width)
        reference[:, 0::2] = torch.sin(positions * frequencies)
        reference[:, 1::2] = torch.cos(positions * frequencies)
        encoding = PositionalEncoding(width, dropout=0.0, max_len=max_len)

        self.assertTrue(torch.equal(encoding.pe, reference.unsqueeze(0)))

    def test_zero_dropout_batch_addition_and_gradient(self):
        encoding = PositionalEncoding(93, dropout=0.0, max_len=11)
        inputs = torch.arange(2 * 7 * 93, dtype=torch.float32).reshape(2, 7, 93)
        inputs = inputs.div(100).requires_grad_()
        expected = inputs.detach() + encoding.pe[:, :7]

        encoding.train()
        train_outputs = encoding(inputs)
        encoding.eval()
        eval_outputs = encoding(inputs)

        self.assertTrue(torch.equal(train_outputs, expected))
        self.assertTrue(torch.equal(eval_outputs, expected))
        train_outputs.sum().backward()
        self.assertTrue(torch.equal(inputs.grad, torch.ones_like(inputs)))


if __name__ == "__main__":
    unittest.main()
