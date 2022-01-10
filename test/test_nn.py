import unittest

from torch import nn
import numpy as np

from src.nn import NN


class TestNN(unittest.TestCase):
    def test_parameters_count(self):
        model = NN(1)
        model.model = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 15),
            nn.ReLU(),
            nn.Linear(15, 1),
        )

        # assuming bias is in place
        expected = (1 + 1) * 10 + (10 + 1) * 15 + (15 + 1) * 1
        self.assertEqual(model.parameters_count(), expected)

    def test__set_weights_count_mismatch(self):
        model = NN(1)
        _module = nn.Linear(1, 100)
        model.model = _module
        params_minus_1 = model.parameters_count() - 1
        w = np.ones((params_minus_1, 1))

        with self.assertRaises(AssertionError) as context:
            NN._set_weights(_module, w)
        self.assertTrue('Weights count mismatch.' in str(context.exception))

        params = model.parameters_count()
        w = np.ones((params, 1))

        try:
            NN._set_weights(_module, w)
        except AssertionError:
            self.fail('Should not raise exception.')

    def test__set_weights(self):
        module = nn.Linear(1, 80)
        self.assertTrue(not all(x == 1 for x in module.weight.data))

        w = np.ones((200, 1))

        returned_idx = NN._set_weights(module, w)

        self.assertEqual(returned_idx, 160)
        self.assertTrue(all(x == 1 for x in module.weight.data))

    def test_set_weights_for_multiple_modules(self):
        m1 = nn.Linear(1, 10)
        m2 = nn.Linear(1, 10)
        m3 = nn.Linear(1, 10)
        modules = nn.Sequential(m1, m2, m3)
        model = NN(1)
        model.model = modules
        w = np.concatenate([
            np.zeros((20, 1)),
            np.ones((20, 1)),
            np.zeros((20, 1)),
        ])

        model.set_weights(w)

        self.assertTrue(all(x == 0 for x in [*m1.weight.data, *m1.bias.data]))
        self.assertTrue(all(x == 1 for x in [*m2.weight.data, *m2.bias.data]))
        self.assertTrue(all(x == 0 for x in [*m3.weight.data, *m3.bias.data]))


if __name__ == '__main__':
    unittest.main()
