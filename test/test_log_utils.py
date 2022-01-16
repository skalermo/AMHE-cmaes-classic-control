import unittest

from src.log_utils import chunk_rollouts, extract_data, process_logs, process_cmaess_nn_logs


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        with open('test_data/example_log_a2c.txt', 'r') as f:
            self.logs = f.read()

    def test_chunk_rollouts(self):
        chunks = chunk_rollouts(self.logs)
        self.assertEqual(len(list(chunks)), 4)

    def test_extract_data(self):
        chunk = next(chunk_rollouts(self.logs))
        data = extract_data(chunk)
        expected = {'ep_rew_mean': 16, 'iterations': 4, 'total_timesteps': 20}
        self.assertDictEqual(data, expected)

    def test_process_logs(self):
        data = process_logs(self.logs)
        self.assertTrue(isinstance(data, list))
        self.assertTrue(len(data) > 0)
        for d in data:
            self.assertTrue(isinstance(d, dict))
            self.assertTrue('ep_rew_mean' in d)
            self.assertTrue('iterations' in d)
            self.assertTrue('total_timesteps' in d)

    def test_process_cmaes_nn_logs(self):
        with open('test_data/example_log_cmaesnn.txt', 'r') as f:
            logs = f.read()
        data = process_cmaess_nn_logs(logs)
        self.assertTrue(isinstance(data, list))
        self.assertTrue(len(data) > 0)
        for d in data:
            self.assertTrue(isinstance(d, dict))
            self.assertTrue('episode' in d)
            self.assertTrue('population_best_return' in d)
            self.assertTrue('population_avg_return' in d)
            self.assertTrue('population_return_std' in d)
            self.assertTrue('total_timesteps' in d)


if __name__ == '__main__':
    unittest.main()
