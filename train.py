from tmnfrl.env import env
import time

if __name__ == "__main__":
    env = env(action_space="gamepad")
    while True:
        time.sleep(0.1)
        obs = env.step([1.0, 1.0])
        print(obs)
        env.render()
