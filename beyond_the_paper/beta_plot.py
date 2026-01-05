import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

GRID_SIZE = 15      
EPISODES = 400 
GOAL = (GRID_SIZE - 1, GRID_SIZE - 1)
START = (0, 0)
ACTIONS = [0, 1, 2, 3] 
GAMMA = 0.99

class GridWorld:
    def __init__(self):
        self.reset()
    def reset(self):
        self.agent_pos = START
        return self.agent_pos
    def step(self, action):
        x, y = self.agent_pos
        if action == 0: x = max(0, x - 1)
        elif action == 1: x = min(GRID_SIZE - 1, x + 1)
        elif action == 2: y = max(0, y - 1)
        elif action == 3: y = min(GRID_SIZE - 1, y + 1)
        self.agent_pos = (x, y)
        done = (self.agent_pos == GOAL)
        reward = 100 if done else -1 
        return self.agent_pos, reward, done

def get_features_low_dim(state, action):
    x, y = state
    norm_x, norm_y = x / (GRID_SIZE - 1), y / (GRID_SIZE - 1)
    feats = np.array([norm_x, norm_y, 1.0])
    phi = np.zeros(12) 
    start_idx = action * 3
    phi[start_idx : start_idx + 3] = feats
    return phi

class UCBAgent:
    def __init__(self, n_features, beta, alpha=0.01):
        self.w = np.zeros(n_features)
        self.beta = beta
        self.alpha = alpha
        self.Lambda_inv = np.eye(n_features)

    def get_optimistic_q(self, state, action):
        phi = get_features_low_dim(state, action)
        q_val = np.dot(self.w, phi)
        exploration_term = np.sqrt(np.dot(phi, np.dot(self.Lambda_inv, phi)))
        return q_val + self.beta * exploration_term # Le levier Beta

    def select_action(self, state):
        qs = [self.get_optimistic_q(state, a) for a in ACTIONS]
        max_q = np.max(qs)
        ties = [a for a, q in zip(ACTIONS, qs) if q == max_q]
        return np.random.choice(ties)

    def update(self, state, action, reward, next_state, done):
        phi = get_features_low_dim(state, action)
        top = np.dot(np.dot(self.Lambda_inv, np.outer(phi, phi)), self.Lambda_inv)
        bottom = 1 + np.dot(phi, np.dot(self.Lambda_inv, phi))
        self.Lambda_inv -= top / bottom
        if done: target = reward
        else: target = reward + GAMMA * max([self.get_optimistic_q(next_state, a) for a in ACTIONS])
        error = target - np.dot(self.w, phi)
        self.w += self.alpha * error * phi

betas_to_test = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
entropy_results = []

print("Entropy slope.")

for beta in betas_to_test:
    env = GridWorld()
    agent = UCBAgent(n_features=12, beta=beta, alpha=0.01)
    visit_counts = np.zeros((GRID_SIZE, GRID_SIZE))
    
    for e in range(EPISODES):
        state = env.reset()
        visit_counts[state] += 1
        done = False
        steps = 0
        while not done and steps < 200:
            action = agent.select_action(state)
            next_s, r, done = env.step(action)
            agent.update(state, action, r, next_s, done)
            state = next_s
            visit_counts[state] += 1
            steps += 1
            
    p_dist = visit_counts.flatten() / np.sum(visit_counts)
    ent = entropy(p_dist + 1e-9)
    entropy_results.append(ent)
    print(f"Beta={beta} -> Entropy={ent:.3f}")


plt.figure(figsize=(10, 6))
plt.plot(betas_to_test, entropy_results, marker='o', linewidth=2, label="Entropy Measure")


plt.title("Non-Linearity of Exploration: Entropy = f(Beta)")
plt.xlabel("Beta Parameter")
plt.ylabel("Shannon Entropy (Visits Diversity)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.annotate('Activation', xy=(0.3, entropy_results[2]), xytext=(1, entropy_results[2]-0.5),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate('Saturation', xy=(5.0, entropy_results[-2]), xytext=(5, entropy_results[-2]-0.5),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.show()