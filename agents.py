import numpy as np

class TDAgent:
    """Agent Temporal Difference (Q-learning Linéaire) - Mise à jour incrémentale."""
    def __init__(self, d, n_actions, alpha=0.1, epsilon=0.1, gamma=0.99):
        self.d = d
        self.n_actions = n_actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.w = np.zeros(d)

    def select_action(self, phi_s_all):
        # phi_s_all est une matrice (n_actions, d)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        q_values = phi_s_all @ self.w
        return np.argmax(q_values)

    def update(self, phi, reward, phi_next_all):
        v_next = np.max(phi_next_all @ self.w)
        target = reward + self.gamma * v_next
        prediction = phi @ self.w
        # Mise à jour stochastique du gradient
        self.w += self.alpha * (target - prediction) * phi

class LSPIAgent:
    """Least-Squares Policy Iteration - Apprentissage par lots (Batch)."""
    def __init__(self, d, n_actions, gamma=0.99):
        self.d = d
        self.n_actions = n_actions
        self.gamma = gamma
        self.w = np.zeros(d)
        self.A = np.eye(d) * 0.1 # Matrice de corrélation
        self.b = np.zeros(d)

    def select_action(self, phi_s_all):
        q_values = phi_s_all @ self.w
        return np.argmax(q_values)

    def collect_data(self, phi, reward, phi_next_best):
        # phi_next_best est le vecteur de caractéristiques de l'action optimale au prochain état
        self.A += np.outer(phi, (phi - self.gamma * phi_next_best))
        self.b += phi * reward

    def fit(self):
        """Résolution analytique du point fixe de Bellman."""
        try:
            self.w = np.linalg.solve(self.A, self.b)
        except np.linalg.LinAlgError:
            self.w = np.linalg.lstsq(self.A, self.b, rcond=None)[0]

class LSVIUCBAgent:
    """Least-Squares Value Iteration avec UCB - Exploration Optimiste."""
    def __init__(self, d, n_actions, horizon, beta=0.1, lambda_reg=1.0):
        self.d = d
        self.K = n_actions
        self.H = horizon
        self.beta = beta
        self.lambda_reg = lambda_reg
        
        # Un modèle par étape de l'horizon
        self.Lambda = [np.eye(d) * lambda_reg for _ in range(horizon)]
        self.w = [np.zeros(d) for _ in range(horizon)]
        self.b = [np.zeros(d) for _ in range(horizon)]

    def select_action(self, phi_s_all, h):
        """Sélection d'action avec bonus UCB."""
        inv_L = np.linalg.inv(self.Lambda[h])
        q_estimates = phi_s_all @ self.w[h]
        
        # Calcul du bonus quadratique : beta * sqrt(phi^T * Lambda^-1 * phi)
        bonuses = self.beta * np.sqrt(np.sum((phi_s_all @ inv_L) * phi_s_all, axis=1))
        
        return np.argmax(q_estimates + bonuses)

    def update(self, h, phi, reward, v_next):
        """Mise à jour LSVI pour l'étape h."""
        self.Lambda[h] += np.outer(phi, phi)
        self.b[h] += phi * (reward + v_next)
        
        # Résolution des moindres carrés régularisés (Ridge Regression)
        # w = (Phi^T * Phi + lambda*I)^-1 * Phi^T * y
        inv_L = np.linalg.inv(self.Lambda[h])
        self.w[h] = inv_L @ self.b[h]