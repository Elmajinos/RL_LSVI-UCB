import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class LinearGridWorld:
    def __init__(self, size=5, horizon=20):
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

class LSVIUCBAgent:
    def __init__(self, d, n_actions, horizon, beta=1.0):
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
            
            if q_val > best_val:
                best_val = q_val
                best_a = a
        return best_a

    def update(self, h, phi, target):
        self.Lambda[h] += np.outer(phi, phi)
        inv_L = np.linalg.inv(self.Lambda[h])
        
        if not hasattr(self, 'b'): self.b = [np.zeros(self.d) for _ in range(self.H)]
        
        self.b[h] += phi * target
        self.w[h] = inv_L @ self.b[h]

def run_demo():
    SIZE = 5
    HORIZON = 15
    EPISODES = 300 
    
    env = LinearGridWorld(size=SIZE, horizon=HORIZON)
    
    agent = LSVIUCBAgent(d=env.d, n_actions=env.n_actions, horizon=HORIZON, beta=10.0)

    trajectories = [] 
    capture_indices = [0, 10, 50, 290] 

    regret = []
    
    print("Starting simulation...")
    
    for k in range(EPISODES):
        s = 0
        path = []
        episode_reward = 0
        
        for h in range(HORIZON):
            r, c = divmod(s, SIZE)
            path.append((c, r)) 
            
            a = agent.select_action(s, env, h)
            ns, rew = env.step(s, a)
            
            phi_ns = env.get_phi(ns)
            v_next = 0
            
            if h < HORIZON - 1:
                v_next = np.max([phi_ns @ agent.w[h+1]]) 
            
            agent.update(h, env.get_phi(ns), rew + v_next)
            
            s = ns
            episode_reward += rew
            if ns == env.target_state:
                path.append((SIZE-1, SIZE-1))
                break
        
        regret.append(10.0 - episode_reward if episode_reward < 10 else 0)
        
        if k in capture_indices:
            trajectories.append((k, path))

    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 2)

    ax_traj = fig.add_subplot(gs[0, 0])
    ax_traj.set_title("Trajectory Evolution (LSVI-UCB)")
    ax_traj.set_xlim(-0.5, SIZE-0.5)
    ax_traj.set_ylim(SIZE-0.5, -0.5) 
    ax_traj.grid(True)
    
    colors = ['#CCCCCC', '#FFA500', '#FF4500', '#008000'] 
    widths = [1, 2, 2, 3]
    
    for i, (ep_idx, path) in enumerate(trajectories):
        path = np.array(path)
        if len(path) > 0:
            noise = np.random.normal(0, 0.05, path.shape)
            label = f"Ep {ep_idx}"
            ax_traj.plot(path[:,0]+noise[:,0], path[:,1]+noise[:,1], 
                         color=colors[i], linewidth=widths[i], marker='.', label=label)

    ax_traj.plot(0, 0, 'bo', markersize=10, label='Start')
    ax_traj.plot(SIZE-1, SIZE-1, 'rx', markersize=12, markeredgewidth=3, label='Goal')
    ax_traj.legend()

    ax_map = fig.add_subplot(gs[0, 1])
    
    V_map = np.zeros((SIZE, SIZE))
    for r in range(SIZE):
        for c in range(SIZE):
            s = r * SIZE + c
            phi = env.get_phi(s)
            
            values_over_time = []
            for h in range(HORIZON):
                val_h = phi @ agent.w[h]
                values_over_time.append(val_h)
            
            V_map[r, c] = np.max(values_over_time)
            
    sns.heatmap(V_map, annot=True, ax=ax_map, cmap="viridis", fmt=".1f")
    ax_map.set_title("Global Value Map (Max over all h)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_demo()