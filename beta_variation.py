import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# --- CONFIGURATION ---
GRID_SIZE = 15      # 225 états
EPISODES = 500      # Suffisant pour voir la saturation
GOAL = (GRID_SIZE - 1, GRID_SIZE - 1)
START = (0, 0)
ACTIONS = [0, 1, 2, 3] # Haut, Bas, Gauche, Droite
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
        # Reward classique: -1 par pas, +100 au but
        reward = 100 if done else -1 
        return self.agent_pos, reward, done

# Features (x, y, 1) normalisées
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
        # Matrice Lambda inverse pour le calcul du bonus UCB
        self.Lambda_inv = np.eye(n_features)

    def get_optimistic_q(self, state, action):
        phi = get_features_low_dim(state, action)
        q_val = np.dot(self.w, phi)
        
        # --- C'est ici que beta agit ---
        # Bonus = beta * sqrt(phi^T * Lambda^-1 * phi)
        exploration_term = np.sqrt(np.dot(phi, np.dot(self.Lambda_inv, phi)))
        return q_val + self.beta * exploration_term

    def select_action(self, state):
        # Choix glouton sur la Q-value Optimiste
        qs = [self.get_optimistic_q(state, a) for a in ACTIONS]
        max_q = np.max(qs)
        ties = [a for a, q in zip(ACTIONS, qs) if q == max_q]
        return np.random.choice(ties)

    def update(self, state, action, reward, next_state, done):
        phi = get_features_low_dim(state, action)
        
        # Mise à jour de Lambda Inverse (Sherman-Morrison)
        # O(d^2) au lieu de O(d^3) pour l'inversion
        top = np.dot(np.dot(self.Lambda_inv, np.outer(phi, phi)), self.Lambda_inv)
        bottom = 1 + np.dot(phi, np.dot(self.Lambda_inv, phi))
        self.Lambda_inv -= top / bottom
        
        # Mise à jour des poids w (LSVI style)
        if done: target = reward
        else: target = reward + GAMMA * max([self.get_optimistic_q(next_state, a) for a in ACTIONS])
            
        error = target - np.dot(self.w, phi)
        self.w += self.alpha * error * phi

def run_metric_experiment(betas):
    results = {}
    
    for beta in betas:
        print(f"Simulation pour beta={beta}...")
        env = GridWorld()
        agent = UCBAgent(n_features=12, beta=beta, alpha=0.01)
        
        global_visited = set()
        coverage_history = []
        visit_counts = np.zeros((GRID_SIZE, GRID_SIZE))
        
        for e in range(EPISODES):
            state = env.reset()
            # On compte l'état de départ
            global_visited.add(state)
            visit_counts[state] += 1
            
            done = False
            steps = 0
            while not done and steps < 200:
                action = agent.select_action(state)
                next_s, r, done = env.step(action)
                agent.update(state, action, r, next_s, done)
                state = next_s
                
                # Tracking des visites
                global_visited.add(state)
                visit_counts[state] += 1
                steps += 1
            
            # Calcul du % de couverture à la fin de l'épisode
            coverage = len(global_visited) / (GRID_SIZE * GRID_SIZE) * 100
            coverage_history.append(coverage)
            
        # Calcul de l'Entropie finale (Dispersion des visites)
        p_dist = visit_counts.flatten() / np.sum(visit_counts)
        expl_entropy = entropy(p_dist + 1e-9) # +epsilon pour éviter log(0)
            
        results[beta] = {
            "coverage": coverage_history,
            "entropy": expl_entropy
        }
        
    return results

# --- EXECUTION ---
betas = [0.0, 2.0, 10.0]
data = run_metric_experiment(betas)

# --- VISUALIZATION ---
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Couverture Temporelle (% de la carte découverte)
for beta in betas:
    label = f"Beta={beta} (Entropy={data[beta]['entropy']:.2f})"
    ax[0].plot(data[beta]['coverage'], label=label, linewidth=2)

ax[0].set_title("State Space Coverage %")
ax[0].set_xlabel("Episode")
ax[0].set_ylabel("% of Unique States Visited")
ax[0].legend()
ax[0].grid(True, alpha=0.3)

# Plot 2: Entropie (Mesure scalaire de l'uniformité)
entropies = [data[b]['entropy'] for b in betas]
colors = ['blue', 'orange', 'green']
bars = ax[1].bar([str(b) for b in betas], entropies, color=colors)
ax[1].set_title("Exploration Spread (Shannon Entropy)")
ax[1].set_xlabel("Beta Value")
ax[1].set_ylabel("Entropy (Higher = More Uniform Exploration)")

# Ajout des valeurs sur les barres
for bar in bars:
    yval = bar.get_height()
    ax[1].text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.2f}", va='bottom', ha='center')

plt.tight_layout()
plt.show()