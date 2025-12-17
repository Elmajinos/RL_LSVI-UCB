import numpy as np

class LinearGridWorld:
    def __init__(self, size=5, horizon=20):
        self.size = size
        self.H = horizon
        self.d = size * size
        self.n_actions = 4
        self.target_state = self.d - 1
        # Coordonnées de la cible (ex: 4, 4)
        self.target_x, self.target_y = divmod(self.target_state, self.size)

    def get_phi(self, state):
        phi = np.zeros(self.d)
        phi[state] = 1.0
        return phi

    def _transition(self, s, a):
        x, y = divmod(s, self.size)
        if a == 0: x = max(0, x - 1)
        elif a == 1: x = min(self.size - 1, x + 1)
        elif a == 2: y = max(0, y - 1)
        elif a == 3: y = min(self.size - 1, y + 1)
        return x * self.size + y

    def get_reward(self, s):
        """Récompense 'Shaped' : Chaud/Froid en fonction de la distance."""
        if s == self.target_state:
            return 1.0
        
        # Calcul de la distance de Manhattan vers la cible
        x, y = divmod(s, self.size)
        dist = abs(x - self.target_x) + abs(y - self.target_y)
        max_dist = self.size * 2
        
        # Récompense partielle : plus on est près, plus c'est haut (entre 0 et 0.5)
        # Cela guide l'agent vers la cible de 1.0
        return 0.1 * (1 - dist / max_dist)

    def compute_v_star(self):
        V = np.zeros((self.H + 1, self.d))
        for h in range(self.H - 1, -1, -1):
            for s in range(self.d):
                q_vals = []
                for a in range(self.n_actions):
                    next_s = self._transition(s, a)
                    # On utilise la nouvelle fonction de reward
                    reward = self.get_reward(next_s)
                    q_vals.append(reward + V[h + 1, next_s])
                V[h, s] = max(q_vals)
        return V