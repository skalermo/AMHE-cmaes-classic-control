import gym

from src.cmaes_agent import CMAESAgent


def main():
    env_id = 'CartPole-v0'
    model_name = '_'.join([CMAESAgent.__name__, env_id])
    cmaes_agent = CMAESAgent.load(model_name)
    if cmaes_agent is None:
        cmaes_agent = CMAESAgent(env_id, cmaes_population_size=10, max_nn_params=100, verbose=True)
        cmaes_agent.learn(total_timesteps=123_456)
        cmaes_agent.save(model_name)

    env = gym.make(env_id)
    obs = env.reset()

    while True:
        action = cmaes_agent.predict(obs)
        obs, reward, done, info = env.step(action)
        if done:
            break
        env.render()


if __name__ == '__main__':
    main()
