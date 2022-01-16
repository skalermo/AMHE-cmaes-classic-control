import unittest

import gym
from stable_baselines3 import A2C, PPO

from src.env_info import env_to_action_type
from src.log_utils import process_logs, process_cmaess_nn_logs
from src.cmaes_nn import CMAESNN
from utils import captured_output


class TestLearningLogExtraction(unittest.TestCase):
    envs = env_to_action_type.keys()

    def test_a2c(self):
        for env_id in self.envs:
            with self.subTest(f'{env_id=}'):
                with captured_output() as (out, err):
                    model = A2C(policy='MlpPolicy', env=gym.make(env_id), verbose=1)
                    model.learn(total_timesteps=1000)
                output = out.getvalue().strip()
                data = process_logs(output)
                self.assertTrue(len(data) > 0)

    def test_ppo(self):
        for env_id in self.envs:
            with self.subTest(f'{env_id=}'):
                with captured_output() as (out, err):
                    model = PPO(policy='MlpPolicy', env=gym.make(env_id), verbose=1)
                    model.learn(total_timesteps=100)
                output = out.getvalue().strip()
                data = process_logs(output)
                self.assertTrue(len(data) > 0)

    def test_cmaes_nn(self):
        for env_id in self.envs:
            with self.subTest(f'{env_id=}'):
                with captured_output() as (out, err):
                    model = CMAESNN(env_id, verbose=1)
                    model.learn(total_timesteps=100, log_interval=1)
                output = out.getvalue().strip()
                data = process_cmaess_nn_logs(output)
                self.assertTrue(len(data) > 0)


if __name__ == '__main__':
    unittest.main()
