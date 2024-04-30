from env.TMNFEnv import TrackmaniaEnv
import time
from actor_critic import Actor
if __name__ == "__main__":
    env = TrackmaniaEnv(action_space="gamepad")
    actor = Actor(env.action_space, env.observation_space)
    i = 0
    obs = env.reset()
    while True:
        obs, _, _, _ = env.step(actor.act(obs))
        i += 1
        print(obs)
        if i % 10 == 0:
            env.reset()
            print("reset done")
        env.render()
