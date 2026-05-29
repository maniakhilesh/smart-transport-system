import numpy as np
from rl.q_env import ACTIONS, SPEED_BINS, DIST_BINS, DENSITY_BINS, discretize


class QAgent:
    def __init__(self, q_path="rl/q_table.npy"):
        try:
            self.q_table = np.load(q_path)
            self.ready = True
        except Exception as e:
            print("⚠️ Failed to load Q-table:", e)
            self.q_table = None
            self.ready = False

    def predict(self, speed, lane, front_dist, density):
        """
        Returns:
        - action index
        - action label
        - confidence vector
        """

        if not self.ready:
            return 0, "STAY", [0.33, 0.33, 0.33]

        state = (
            discretize(speed, SPEED_BINS),
            lane,
            discretize(front_dist, DIST_BINS),
            discretize(density, DENSITY_BINS),
        )

        q_vals = self.q_table[state]
        action = int(np.argmax(q_vals))

        # Softmax confidence (for UI)
        exp_q = np.exp(q_vals - np.max(q_vals))
        probs = exp_q / exp_q.sum()

        return action, ACTIONS[action], probs.tolist()
