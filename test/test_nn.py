import unittest

from torch import nn

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


if __name__ == '__main__':
    unittest.main()
