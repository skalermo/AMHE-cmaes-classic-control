import gym
import numpy as np
from cmaes import CMA

from nn import NN


def evaluate_model(model: NN, env: gym.Env, render: bool = False) -> int:
    end_reward = 0

    state = env.reset()

    for time in range(1200):
        if render:
            env.render()

        action = model.map_to_action(state)
        if continuous:
            action = [action]
        state, reward, done, info = env.step(action)
        if not done:
            end_reward += reward
        else:
            break
    return end_reward


def main():
    EPISODES = 1000

    env_to_action_type = {
        'CartPole-v0': 'discrete',
        'CartPole-v1': 'discrete',
        'Acrobot-v1': 'discrete',
        'MountainCar-v0': 'discrete',
        'MountainCarContinuous-v0': 'continuous',
        'Pendulum-v1': 'continuous',
    }
    env_name = 'MountainCarContinuous-v0'
    env = gym.make(env_name)
    env.reset()

    global continuous
    if env_to_action_type[env_name] == 'continuous':
        actions = env.action_space.shape[0]
        continuous = True
    else:
        actions = env.action_space.n
        continuous = False

    print(actions)

    model = NN(env.observation_space.shape[0], actions, env_to_action_type[env_name])
    print(model)

    params = model.parameters_count()
    optimizer = CMA(mean=np.zeros(params), sigma=1.3, population_size=15, seed=123)

    for e in range(EPISODES):
        solutions = []
        best_reward = -1e6
        best_weights = None

        for _ in range(optimizer.population_size):
            w = optimizer.ask()
            model.set_weights(w)
            reward_obtained = evaluate_model(model, env)
            if reward_obtained > best_reward:
                best_reward = reward_obtained
                best_weights = w

            solutions.append((w, -reward_obtained))

        optimizer.tell(solutions)
        print(f'{e=} {best_reward=}')

        if e % 10 == 0:
            model.set_weights(best_weights)
            evaluate_model(model, env, render=True)


if __name__ == '__main__':
    continuous = None
    main()
