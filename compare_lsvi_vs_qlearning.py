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
        self.Lambda = [np.eye(d) * 0.1 for _ in range(horizon)]
        self.w = [np.zeros(d) for _ in range(horizon)]

    def select_action(self, s, env, h):
        best_a = -1
        best_val = -np.inf
        for a in range(env.n_actions):
            ns, _ = env.step(s, a)
            phi = env.get_phi(ns) 
            inv_L = np.linalg.inv(self.Lambda[h])
            bonus = self.beta * np.sqrt(phi @ inv_L @ phi)
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
# 3. SIMULATION LONGUE DURÉE
# ==========================================
def run_comparison():
    SIZE = 5
    HORIZON = 15
    EPISODES = 5000 # On augmente drastiquement !
    
    env = LinearGridWorld(size=SIZE, horizon=HORIZON)
    
    lsvi = LSVIUCBAgent(env.d, env.n_actions, HORIZON, beta=10.0)
    # Epsilon commence à 1.0 (full exploration) et descendra
    ql = QLearningAgent(env.d, env.n_actions, alpha=0.1, epsilon=1.0)
    
    rewards_lsvi = []
    rewards_ql = []

    print(f"🏃‍♂️ Marathon : LSVI vs Q-Learning sur {EPISODES} épisodes...")

    for k in range(EPISODES):
        # Gestion de Epsilon (Decay) pour Q-learning
        # On le fait descendre doucement jusqu'à 0.1
        ql.epsilon = max(0.1, 1.0 - (k / (EPISODES * 0.6)))

        # --- Tour de LSVI ---
        s = 0
        score_lsvi = 0
        for h in range(HORIZON):
            a = lsvi.select_action(s, env, h)
            ns, r = env.step(s, a)
            phi_ns = env.get_phi(ns)
            v_next = np.max([phi_ns @ lsvi.w[h+1]]) if h < HORIZON - 1 else 0
            lsvi.update(h, env.get_phi(ns), r + v_next)
            s = ns
            score_lsvi += r
            if ns == env.target_state: break
        rewards_lsvi.append(score_lsvi)

        # --- Tour de Q-Learning ---
        s = 0
        score_ql = 0
        for h in range(HORIZON):
            a = ql.select_action(s)
            ns, r = env.step(s, a)
            ql.update(s, a, r, ns)
            s = ns
            score_ql += r
            if ns == env.target_state: break
        rewards_ql.append(score_ql)
        
        if k % 500 == 0:
            print(f"Épisode {k}/{EPISODES} - Epsilon QL: {ql.epsilon:.2f}")

    # --- LISSAGE ---
    window = 100 # Fenêtre plus large pour lisser le bruit sur 5000 points
    smooth_lsvi = np.convolve(rewards_lsvi, np.ones(window)/window, mode='valid')
    smooth_ql = np.convolve(rewards_ql, np.ones(window)/window, mode='valid')

    # --- VISUALISATION ---
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2)

    # 1. Courbes
    ax_curve = fig.add_subplot(gs[0, :])
    ax_curve.plot(smooth_lsvi, label="LSVI-UCB (Immédiat)", color='red', linewidth=1.5)
    ax_curve.plot(smooth_ql, label="Q-Learning (Converge lentement)", color='blue', linestyle='--', alpha=0.8)
    ax_curve.set_title(f"Convergence sur le long terme ({EPISODES} épisodes)")
    ax_curve.set_xlabel("Épisodes")
    ax_curve.set_ylabel("Récompense Moyenne")
    ax_curve.legend()
    ax_curve.grid(True, alpha=0.3)

    # 2. Heatmap LSVI
    ax_lsvi = fig.add_subplot(gs[1, 0])
    map_lsvi = np.zeros((SIZE, SIZE))
    for r in range(SIZE):
        for c in range(SIZE):
            s = r * SIZE + c
            phi = env.get_phi(s)
            vals = [phi @ lsvi.w[h] for h in range(HORIZON)]
            map_lsvi[r, c] = np.max(vals)
    sns.heatmap(map_lsvi, annot=True, ax=ax_lsvi, cmap="Reds", fmt=".1f")
    ax_lsvi.set_title("Cerveau LSVI")

    # 3. Heatmap Q-Learning
    ax_ql = fig.add_subplot(gs[1, 1])
    map_ql = np.zeros((SIZE, SIZE))
    for r in range(SIZE):
        for c in range(SIZE):
            s = r * SIZE + c
            map_ql[r, c] = np.max(ql.q_table[s])
    sns.heatmap(map_ql, annot=True, ax=ax_ql, cmap="Blues", fmt=".1f")
    ax_ql.set_title(f"Cerveau Q-Learning (Après {EPISODES} ép.)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_comparison()