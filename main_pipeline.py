import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from linear_mdp_core import LinearGridWorld
from agents import TDAgent, LSVIUCBAgent

def run_pipeline():
    # --- Configuration ---
    SIZE = 5
    HORIZON = 20
    EPISODES = 1000 # On laisse le temps d'apprendre
    
    env = LinearGridWorld(size=SIZE, horizon=HORIZON)
    v_star_matrix = env.compute_v_star()
    v_star_init = v_star_matrix[0, 0]

    # LSVI-UCB (Rouge) : Exploration intelligente
    agent_lsvi = LSVIUCBAgent(env.d, env.n_actions, HORIZON, beta=1.0)
    # TD-Learning (Bleu) : Exploration epsilon-greedy
    agent_td = TDAgent(env.d, env.n_actions, alpha=0.1, epsilon=0.2)

    regret_lsvi, regret_td = [], []
    cum_reg_lsvi, cum_reg_td = 0, 0

    # Pour stocker les chemins (x, y) à dessiner
    trajectory_snapshots = []
    snapshot_intervals = [0, 50, 200, 900] # Episodes à capturer

    print(f"Simulation sur {EPISODES} épisodes...")

    for k in range(EPISODES):
        # --- LSVI-UCB ---
        s = 0
        path_lsvi = [] # On enregistre le chemin de cet épisode
        cumulative_reward = 0
        
        for h in range(HORIZON):
            # Conversion index -> (x, y) pour le dessin
            path_lsvi.append(divmod(s, SIZE))
            
            phi_actions = np.array([env.get_phi(env._transition(s, a)) for a in range(env.n_actions)])
            a = agent_lsvi.select_action(phi_actions, h)
            ns = env._transition(s, a)
            r = env.get_reward(ns)
            
            # Mise à jour
            phi_next = np.array([env.get_phi(env._transition(ns, act)) for act in range(env.n_actions)])
            v_next = np.max(phi_next @ agent_lsvi.w[h+1]) if h < HORIZON-1 else 0
            agent_lsvi.update(h, env.get_phi(s), r, v_next)
            
            s = ns
            cumulative_reward += r
            
            if ns == env.target_state:
                path_lsvi.append(divmod(ns, SIZE)) # Ajouter la fin
                break

        # Sauvegarde de la trajectoire si on est au bon moment
        if k in snapshot_intervals:
            trajectory_snapshots.append((k, path_lsvi))

        # --- TD-Learning (Juste pour le regret, on ne trace pas ses chemins pour ne pas surcharger) ---
        s = 0
        cum_rew_td = 0
        for h in range(HORIZON):
            phi_s = env.get_phi(s)
            phi_actions = np.array([env.get_phi(env._transition(s, a)) for a in range(env.n_actions)])
            a = agent_td.select_action(phi_actions)
            ns = env._transition(s, a)
            r = env.get_reward(ns)
            phi_ns = np.array([env.get_phi(env._transition(ns, act)) for act in range(env.n_actions)])
            agent_td.update(phi_s, r, phi_ns)
            s = ns
            cum_rew_td += r
            if ns == env.target_state: break

        # Calcul Regret
        cum_reg_lsvi += (v_star_init - cumulative_reward)
        cum_reg_td += (v_star_init - cum_rew_td)
        regret_lsvi.append(cum_reg_lsvi)
        regret_td.append(cum_reg_td)

    # --- VISUALISATION ---
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 3)

    # 1. Courbes de Regret
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(regret_lsvi, label="LSVI-UCB", color='red', linewidth=2)
    ax1.plot(regret_td, label="TD-Learning", color='blue', linestyle='--')
    ax1.set_title("Regret Cumulé")
    ax1.set_xlabel("Épisodes")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Trajectoires de LSVI (La partie importante !)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("Évolution des Trajectoires (LSVI)")
    ax2.set_xlim(-0.5, SIZE-0.5)
    ax2.set_ylim(SIZE-0.5, -0.5) # Inverser Y pour avoir (0,0) en haut à gauche
    ax2.grid(True)
    
    # Couleurs pour les étapes : Début (Gris), Milieu (Orange), Fin (Rouge Vif)
    colors = ['lightgray', 'orange', 'red', 'darkred']
    
    for i, (episode_idx, path) in enumerate(trajectory_snapshots):
        path = np.array(path)
        if len(path) > 0:
            # On ajoute un petit bruit (jitter) pour ne pas superposer parfaitement les lignes
            jitter = np.random.normal(0, 0.05, path.shape) 
            ax2.plot(path[:, 1] + jitter[:, 1], path[:, 0] + jitter[:, 0], 
                     marker='.', label=f'Ep {episode_idx}', color=colors[i % len(colors)], linewidth=2)
            
    # Marquer le départ et l'arrivée
    ax2.plot(0, 0, 'go', markersize=10, label='Start') # Départ (Vert)
    target_x, target_y = divmod(env.target_state, SIZE)
    ax2.plot(target_y, target_x, 'rx', markersize=12, markeredgewidth=3, label='Goal') # Arrivée (Croix Rouge)
    ax2.legend()

    # 3. Heatmap Finale
    ax3 = fig.add_subplot(gs[0, 2])
    v_learned = (np.eye(env.d) @ agent_lsvi.w[0]).reshape(SIZE, SIZE)
    sns.heatmap(v_learned, annot=True, ax=ax3, cmap="YlOrRd", fmt=".2f")
    ax3.set_title("Carte de Valeur Finale (Ce que l'agent 'voit')")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_pipeline()