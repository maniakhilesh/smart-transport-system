import numpy as np
from q_env import LaneChangeEnv, ACTIONS

print("🚀 Starting Q-learning training...")

# Hyperparameters
EPISODES = 5000
ALPHA = 0.1        # learning rate
GAMMA = 0.95       # discount factor
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

env = LaneChangeEnv()

# Q-table size
q_table = np.zeros((
    len([0, 40, 80, 140]),   # speed bins
    3,                       # lanes
    len([0, 10, 25, 100]),   # distance bins
    len([0, 40, 100, 200]),  # density bins
    len(ACTIONS)             # actions
))

episode_rewards = []

# -----------------------------
# 🔁 TRAINING LOOP (IMPORTANT)
# -----------------------------
for episode in range(EPISODES):

    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # ε-greedy action
        if np.random.rand() < EPSILON:
            action = np.random.randint(len(ACTIONS))
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done = env.step(action)

        # Q-learning update
        best_next = np.max(q_table[next_state])
        q_table[state + (action,)] += ALPHA * (
            reward + GAMMA * best_next - q_table[state + (action,)]
        )

        state = next_state
        total_reward += reward

    episode_rewards.append(total_reward)

    # 🔍 LOGGING (THIS WAS YOUR ERROR POINT)
    if episode % 500 == 0:
        print(
            f"Episode {episode} | "
            f"Avg Reward (last 100): {np.mean(episode_rewards[-100:]):.2f} | "
            f"Epsilon: {EPSILON:.2f}"
        )

    # Decay epsilon
    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY

# -----------------------------
# 💾 SAVE TRAINED Q-TABLE
# -----------------------------
np.save("rl/q_table.npy", q_table)

print("✅ Q-learning training complete. Saved q_table.npy")
