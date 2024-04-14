from env.TMNFEnv import TrackmaniaEnv
import time

if __name__ == "__main__":
    env = TrackmaniaEnv(action_space="gamepad")
    i = 0
    while True:
        time.sleep(0.1)
        obs = env.step([1, 0.2])
        i += 1
        if i % 10 == 0:
            env.reset()
            print("reset done")
        env.render()
