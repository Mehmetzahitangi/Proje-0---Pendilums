import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

import cartpole as cp
import evaluate as eval


class CartPoleContinuousEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, params=None, distribution="zor", obs_noise=0.0, F_max=10.0, dt=0.02, substeps=5, max_adim=500):
        self.params = params if params is not None else cp.CartPoleParams()
        self.f = cp.make_f(self.params)
        self.dagilim, self.obs_noise = distribution, obs_noise
        self.F_max, self.dt, self.substeps, self.max_adim = F_max, dt, substeps, max_adim
        self.Q = np.array([4.0, 1.0, 25.0, 1.0])            # Bryson (köşegen)
        self.R = 0.01
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def _observation(self):
        obs = self.s.copy()
        if self.obs_noise > 0:
            obs = obs + self.np_random.normal(0.0, self.obs_noise, size=4)
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # gymnasium kuralı
        if options is not None and "s0" in options:
            self.s = np.asarray(options["s0"], dtype=float).copy()

        else:
            alt, ust = eval.DAGILIMLAR[self.dagilim]
            self.s = self.np_random.uniform(alt, ust)
        self.adim = 0
        return self._observation(), {}

    def step(self, action):
        F = float(np.clip(action[0], -1.0, 1.0)) * self.F_max
        h = self.dt / self.substeps
        for _ in range(self.substeps):
            self.s = cp.rk4_step(self.f, self.s, F, h)       
        self.adim += 1

        c = float(self.Q @ self.s**2 + self.R * F**2)        # LQR maliyeti (Q köşegen → Σ q_i s_i²)
        reward = float(1 / (1 + c))                                           # ÖDÜL 1 / (1 + c)

        terminated = bool(abs(self.s[0]) > 2.4 or abs(self.s[2]) > np.pi / 2) # # |x| > 2.4  veya  |θ| > π/2   
        truncated = bool(self.adim >= self.max_adim)                       # adım sayısı max_adim'e ulaştı mı?
        return self._observation(), reward, terminated, truncated, {"F": F}

if __name__ == "__main__":
    check_env(CartPoleContinuousEnv())
    print("check_env uygun...")