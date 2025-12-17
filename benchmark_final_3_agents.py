import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. ENVIRONNEMENT
# ==========================================
class LinearGridWorld:
    def __init__(self, size=5, horizon=15):
        self.size = size
        self.H = horizon
        self.d = size * size 
        self.n_actions = 4
        self.target_state = self.d - 1
        self.target_x, self.target_y = divmod(self.target_state, self.size)

    def get_phi(self, state):
        phi = np.zeros(self.d)
        phi[state] = 1.0
        return phi

    def step(self, s, a):
        x, y = divmod(s, self.size)
        if a == 0: x = max(0, x - 1)
        elif a == 1: x = min(self.size - 1, x + 1)
        elif a == 2: y = max(0, y - 1)
        elif a == 3: y = min(self.size - 1, y + 1)
        ns = x * self.size + y
        
        dist = abs(x - self.target_x) + abs(y - self.target_y)
        max_dist = self.size * 2
        
        if ns == self.target_state:
            reward = 10.0
        else:
            reward = 1.0 * (1 - dist / max_dist)
            
        return ns, reward

# ==========================================
# 2. AGENTS
# ==========================================
class LSVIUCBAgent:
    def __init__(self, d, n_actions, horizon, beta=10.0):
        self.d = d
        self.H = horizon
        self.beta = beta
        # On initialise avec une petite régularisation
        self.Lambda = [np.eye(d) * 0.1 for _ in range(horizon)]
        self.w = [np.zeros(d) for _ in range(horizon)]

    def select_action(self, s, env, h):
        best_a = -1
        best_val = -np.inf
        for a in range(env.n_actions):
            ns, _ = env.step(s, a)
            phi = env.get_phi(ns) 
            
            # Inversion de matrice (Le Cerveau Mathématique)
            inv_L = np.linalg.inv(self.Lambda[h])
            
            # Bonus d'exploration (Le Cerveau Curieux)
            bonus = self.beta * np.sqrt(phi @ inv_L @ phi)
            
            # Estimation Q-value
            q_val = (phi @ self.w[h]) + bonus
            
            if q_val > best_val: best_val = q_val; best_a = a
        return best_a

    def update(self, h, phi, target):
        self.Lambda[h] += np.outer(phi, phi)
        inv_L = np.linalg.inv(self.Lambda[h])
        if not hasattr(self, 'b'): self.b = [np.zeros(self.d) for _ in range(self.H)]
        self.b[h] += phi * target
        self.w[h] = inv_L @ self.b[h]

class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=1.0, epsilon=1.0):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha  
        self.gamma = gamma  
        self.epsilon = epsilon 
        self.q_table = np.zeros((n_states, n_actions))

    def select_action(self, s):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_table[s])

    def update(self, s, a, r, ns):
        best_next_q = np.max(self.q_table[ns])
        td_target = r + self.gamma * best_next_q
        td_error = td_target - self.q_table[s, a]
        self.q_table[s, a] += self.alpha * td_error

# ==========================================
# 3. SIMULATION
# ==========================================
def run_comparison():
    SIZE = 5
    HORIZON = 15
    EPISODES = 3000 # Suffisant pour voir les différences
    
    env = LinearGridWorld(size=SIZE, horizon=HORIZON)
    
    # 1. LSVI-UCB (Le Champion)
    lsvi_ucb = LSVIUCBAgent(env.d, env.n_actions, HORIZON, beta=10.0)
    
    # 2. LSVI-Greedy (L'Ablation : sans bonus) -> NOUVEAU
    lsvi_greedy = LSVIUCBAgent(env.d, env.n_actions, HORIZON, beta=0.0)
    
    # 3. Q-Learning (Le Tâtonneur)
    ql = QLearningAgent(env.d, env.n_actions, alpha=0.1, epsilon=1.0)
    
    rewards_ucb, rewards_greedy, rewards_ql = [], [], []

    print(f"🏃‍♂️ Course à 3 : UCB vs Greedy vs QL ({EPISODES} épisodes)...")

    for k in range(EPISODES):
        # Decay Epsilon pour QL
        ql.epsilon = max(0.1, 1.0 - (k / (EPISODES * 0.6)))

        # --- JOUEUR 1 : LSVI-UCB ---
        s = 0; score = 0
        for h in range(HORIZON):
            a = lsvi_ucb.select_action(s, env, h)
            ns, r = env.step(s, a)
            phi_ns = env.get_phi(ns)
            v_next = np.max([phi_ns @ lsvi_ucb.w[h+1]]) if h < HORIZON - 1 else 0
            lsvi_ucb.update(h, env.get_phi(ns), r + v_next)
            s = ns; score += r
            if ns == env.target_state: break
        rewards_ucb.append(score)

        # --- JOUEUR 2 : LSVI-Greedy (Beta=0) ---
        s = 0; score = 0
        for h in range(HORIZON):
            a = lsvi_greedy.select_action(s, env, h)
            ns, r = env.step(s, a)
            phi_ns = env.get_phi(ns)
            v_next = np.max([phi_ns @ lsvi_greedy.w[h+1]]) if h < HORIZON - 1 else 0
            lsvi_greedy.update(h, env.get_phi(ns), r + v_next)
            s = ns; score += r
            if ns == env.target_state: break
        rewards_greedy.append(score)

        # --- JOUEUR 3 : Q-Learning ---
        s = 0; score = 0
        for h in range(HORIZON):
            a = ql.select_action(s)
            ns, r = env.step(s, a)
            ql.update(s, a, r, ns)
            s = ns; score += r
            if ns == env.target_state: break
        rewards_ql.append(score)
        
        if k % 500 == 0: print(f"Ep {k}...")

    # --- LISSAGE ---
    win = 100
    smooth_ucb = np.convolve(rewards_ucb, np.ones(win)/win, mode='valid')
    smooth_greedy = np.convolve(rewards_greedy, np.ones(win)/win, mode='valid')
    smooth_ql = np.convolve(rewards_ql, np.ones(win)/win, mode='valid')

    # --- AFFICHAGE ---
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3)

    # 1. Courbes
    ax_curve = fig.add_subplot(gs[0, :])
    ax_curve.plot(smooth_ucb, label="LSVI-UCB (Beta=10)", color='red', linewidth=2)
    ax_curve.plot(smooth_greedy, label="LSVI-Greedy (Beta=0)", color='green', linestyle='--')
    ax_curve.plot(smooth_ql, label="Q-Learning", color='blue', linestyle=':', alpha=0.8)
    ax_curve.set_title(f"Impact de l'Exploration UCB")
    ax_curve.set_ylabel("Récompense")
    ax_curve.legend()
    ax_curve.grid(True, alpha=0.3)

    # Fonction Helper pour heatmap
    def plot_heatmap(ax, agent, title, cmap):
        map_v = np.zeros((SIZE, SIZE))
        for r in range(SIZE):
            for c in range(SIZE):
                s = r * SIZE + c
                if isinstance(agent, QLearningAgent):
                    map_v[r, c] = np.max(agent.q_table[s])
                else:
                    phi = env.get_phi(s)
                    vals = [phi @ agent.w[h] for h in range(HORIZON)]
                    map_v[r, c] = np.max(vals)
        sns.heatmap(map_v, annot=True, ax=ax, cmap=cmap, fmt=".1f")
        ax.set_title(title)

    # 2. Heatmaps
    ax1 = fig.add_subplot(gs[1, 0])
    plot_heatmap(ax1, lsvi_ucb, "LSVI-UCB (Exploration Optimiste)", "Reds")

    ax2 = fig.add_subplot(gs[1, 1])
    plot_heatmap(ax2, lsvi_greedy, "LSVI-Greedy (Aucune Exploration)", "Greens")

    ax3 = fig.add_subplot(gs[1, 2])
    plot_heatmap(ax3, ql, "Q-Learning (Locale)", "Blues")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_comparison()