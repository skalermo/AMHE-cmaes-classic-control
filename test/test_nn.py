import unittest

from torch import nn
import numpy as np

from src.nn import NN


class TestNN(unittest.TestCase):
    def test_parameters_count_no_bias(self):
        model = NN(1)
        model.model = nn.Sequential(
            nn.Linear(1, 10, bias=False),
            nn.ReLU(),
            nn.Linear(10, 15, bias=False),
            nn.ReLU(),
            nn.Linear(15, 1, bias=False),
        )
        expected = 1 * 10 + 10 * 15 + 15 * 1
        self.assertEqual(model.parameters_count(), expected)

    def test_parameters_count_with_bias(self):
        model = NN(1)
        model.model = nn.Sequential(
            nn.Linear(1, 10, bias=True),
            nn.ReLU(),
            nn.Linear(10, 15, bias=True),
            nn.ReLU(),
            nn.Linear(15, 1, bias=True),
        )
        expected = (1 + 1) * 10 + (10 + 1) * 15 + (15 + 1) * 1
        self.assertEqual(model.parameters_count(), expected)

    def test__set_weights_count_mismatch_no_bias(self):
        model = NN(1)
        _module = nn.Linear(1, 100, bias=False)
        model.model = _module
        params_minus_1 = model.parameters_count() - 1
        w = np.ones((params_minus_1, 1))

        with self.assertRaises(AssertionError) as context:
            NN._set_weights(_module, w, bias=False)
        self.assertTrue('Weights count mismatch.' in str(context.exception))

        params = model.parameters_count()
        w = np.ones((params, 1))

        try:
            NN._set_weights(_module, w, bias=False)
        except AssertionError:
            self.fail('Should not raise exception.')

    def test__set_weights_count_mismatch_with_bias(self):
        model = NN(1)
        _module = nn.Linear(1, 100, bias=True)
        model.model = _module
        params_minus_1 = model.parameters_count() - 1
        w = np.ones((params_minus_1, 1))

        with self.assertRaises(AssertionError) as context:
            NN._set_weights(_module, w, bias=True)
        self.assertTrue('Weights count mismatch.' in str(context.exception))

        params = model.parameters_count()
        w = np.ones((params, 1))

        try:
            NN._set_weights(_module, w, bias=True)
        except AssertionError:
            self.fail('Should not raise exception.')

    def test__set_weights_no_bias(self):
        module = nn.Linear(1, 80, bias=False)
        self.assertTrue(not all(x == 1 for x in module.weight.data))

        w = np.ones((200, 1))

        returned_idx = NN._set_weights(module, w, bias=False)

        self.assertEqual(returned_idx, 80)
        self.assertTrue(all(x == 1 for x in module.weight.data))

    def test__set_weights_with_bias(self):
        module = nn.Linear(1, 80, bias=True)
        self.assertTrue(not all(x == 1 for x in module.weight.data))

        w = np.ones((200, 1))

        returned_idx = NN._set_weights(module, w, bias=True)

        self.assertEqual(returned_idx, 160)
        self.assertTrue(all(x == 1 for x in module.weight.data))

    def test_set_weights_for_multiple_modules_no_bias(self):
        m1 = nn.Linear(1, 10, bias=False)
        m2 = nn.Linear(1, 10, bias=False)
        m3 = nn.Linear(1, 10, bias=False)
        modules = nn.Sequential(m1, m2, m3)
        model = NN(1, bias=False)
        model.model = modules
        w = np.concatenate([
            np.zeros((10, 1)),
            np.ones((10, 1)),
            np.zeros((10, 1)),
        ])

        model.set_weights(w)

        self.assertTrue(all(x == 0 for x in [*m1.weight.data]))
        self.assertTrue(all(x == 1 for x in [*m2.weight.data]))
        self.assertTrue(all(x == 0 for x in [*m3.weight.data]))

    def test_set_weights_for_multiple_modules_with_bias(self):
        m1 = nn.Linear(1, 10, bias=True)
        m2 = nn.Linear(1, 10, bias=True)
        m3 = nn.Linear(1, 10, bias=True)
        modules = nn.Sequential(m1, m2, m3)
        model = NN(1, bias=True)
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

    def test_param_count_not_exceed_max_no_bias(self):
        model = NN(2, 3, max_nn_parameters=80, bias=False)
        self.assertTrue(0 < model.parameters_count() <= 80)  # 40

        model = NN(2, 3, max_nn_parameters=160, bias=False)
        self.assertTrue(80 < model.parameters_count() <= 160)  # 104

        model = NN(2, 3, max_nn_parameters=240, bias=False)
        self.assertTrue(160 < model.parameters_count() <= 240)  # 232

        model = NN(2, 3, max_nn_parameters=320, bias=False)
        self.assertTrue(240 < model.parameters_count() <= 320)  # 296

    def test_param_count_not_exceed_max_with_bias(self):
        model = NN(2, 3, max_nn_parameters=50, bias=True)
        self.assertTrue(0 < model.parameters_count() <= 50)  # 9

        model = NN(2, 3, max_nn_parameters=100, bias=True)
        self.assertTrue(50 < model.parameters_count() <= 100)  # 51

        model = NN(2, 3, max_nn_parameters=150, bias=True)
        self.assertTrue(100 < model.parameters_count() <= 150)  # 123

        model = NN(2, 3, max_nn_parameters=200, bias=True)
        self.assertTrue(150 < model.parameters_count() <= 200)  # 195


if __name__ == '__main__':
    unittest.main()
