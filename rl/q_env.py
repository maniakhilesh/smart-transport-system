import numpy as np

# Discrete bins
SPEED_BINS = [0, 40, 80, 140]
DIST_BINS = [0, 10, 25, 100]
DENSITY_BINS = [0, 40, 100, 200]

ACTIONS = ["STAY", "LEFT", "RIGHT"]

def discretize(val, bins):
    return np.digitize(val, bins) - 1


class LaneChangeEnv:
    def __init__(self):
        self.max_steps = 50          # 🔥 IMPORTANT
        self.current_step = 0        # 🔥 IMPORTANT
        self.reset()

    def reset(self):
        self.current_step = 0        # 🔥 RESET COUNTER
        self.speed = np.random.uniform(30, 90)
        self.lane = np.random.randint(0, 3)
        self.front_dist = np.random.uniform(5, 40)
        self.density = np.random.uniform(10, 120)
        return self._get_state()

    def _get_state(self):
        return (
            discretize(self.speed, SPEED_BINS),
            self.lane,
            discretize(self.front_dist, DIST_BINS),
            discretize(self.density, DENSITY_BINS),
        )

    def step(self, action):
        reward = 0.0
        done = False

        # Lane logic
        if action == 1 and self.lane > 0:      # LEFT
            self.lane -= 1
            reward += 0.5
        elif action == 2 and self.lane < 2:    # RIGHT
            self.lane += 1
            reward += 0.5
        elif action != 0:
            reward -= 0.5  # Invalid move

        # Safety logic
        if self.front_dist < 8:
            reward -= 1.0
            done = True
        else:
            reward += 0.2

        # Update environment
        self.front_dist += np.random.uniform(-3, 3)
        self.speed += np.random.uniform(-5, 5)
        self.density += np.random.uniform(-10, 10)

        # 🔥 EPISODE TIMEOUT (CRITICAL)
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        return self._get_state(), reward, done
