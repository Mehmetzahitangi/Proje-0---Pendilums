"""Cartpole: dinamik, entegrasyon, doğrusallaştırma, ayrıklaştırma, animasyon.

Kural (Ek B): θ = 0 yukarı, θ > 0 sağa yatık; x > 0 sağ; F > 0 sağa.
Durum: s = [x, ẋ, θ, θ̇]. Model: noktasal sarkaç kütlesi, kütlesiz çubuk, sürtünme yok.
Türetme: notebooks/01_derive_cartpole.ipynb (sympy, Lagrange). Analiz: notebooks/02_linearize.ipynb.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm



# 1. Parametreler ve dinamik  (Aşama 1)

@dataclass(frozen=True)
class CartPoleParams:
    M: float = 1.0   # araba kütlesi [kg]
    m: float = 0.1   # sarkaç uç kütlesi [kg]
    l: float = 0.5   # çubuk uzunluğu [m]
    g: float = 9.81  # yerçekimi [m/s²]
    # Sonra eklenecek: b_x (araba sürtünmesi), b_theta (mafsal sönümü)


def accelerations(s, F, p: CartPoleParams):
    """(ẍ, θ̈): iki Euler–Lagrange denkleminin ẍ ve θ̈ için çözülmüş hâli (Ek A.1)."""
    _, _, th, thd = s
    sin, cos = np.sin(th), np.cos(th)
    den = p.M + p.m * sin**2            # kütle matrisinin determinantı / (m l²); her θ'da > 0
    xdd = (F + p.m * sin * (p.l * thd**2 - p.g * cos)) / den
    thdd = (-F * cos - p.m * p.l * thd**2 * sin * cos + (p.M + p.m) * p.g * sin) / (p.l * den)
    return xdd, thdd


def f(s, F, p: CartPoleParams):
    """ṡ = f(s, F) = [ẋ, ẍ, θ̇, θ̈]."""
    xdd, thdd = accelerations(s, F, p)
    return np.array([s[1], xdd, s[3], thdd])


def make_f(p: CartPoleParams):
    """Entegratörlere verilecek f(s, u) kapanışı (parametreler gömülü)."""
    return lambda s, F: f(s, F, p)


def energy(s, p: CartPoleParams):
    """Toplam mekanik enerji E = T + V (V'nin sıfırı: mafsal yüksekliği)."""
    _, xd, th, thd = s
    T = 0.5 * (p.M + p.m) * xd**2 + p.m * p.l * xd * thd * np.cos(th) + 0.5 * p.m * p.l**2 * thd**2
    V = p.m * p.g * p.l * np.cos(th)
    return T + V



# 2. Sayısal entegrasyon  (Aşama 1)
#    Kontrol girdisi u adım boyunca sabit tutulur (sıfırıncı derece tutma).


def euler_step(f, s, u, dt):
    """Açık Euler: s + dt·f(s, u). Global hata O(dt)."""
    return s + dt * f(s, u)


def rk4_step(f, s, u, dt):
    """Klasik 4. derece Runge–Kutta. Global hata O(dt⁴)."""
    k1 = f(s, u)
    k2 = f(s + 0.5 * dt * k1, u)
    k3 = f(s + 0.5 * dt * k2, u)
    k4 = f(s + dt * k3, u)
    return s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate(f, s0, u_fn, dt, t_final, step=rk4_step, substeps=1):
    """s0'dan t_final'a kadar simülasyon.

    u_fn(t, s) -> u: her kontrol adımının başında bir kez çağrılır, dt boyunca sabit kalır (ZOH).
    substeps: her kontrol adımı içinde kaç entegrasyon adımı atılacağı (dt/substeps ile).
    Dönüş: t (N+1,), S (N+1, n), U (N,)
    """
    n_steps = int(round(t_final / dt))
    h = dt / substeps
    s = np.asarray(s0, dtype=float)
    t = np.arange(n_steps + 1) * dt
    S = np.empty((n_steps + 1, s.size))
    U = np.empty(n_steps)
    S[0] = s
    for k in range(n_steps):
        u = u_fn(t[k], s)
        U[k] = u
        for _ in range(substeps):
            s = step(f, s, u, h)
        S[k + 1] = s
    return t, S, U



# 3. Doğrusallaştırma ve analiz  (Aşama 2)
#    δṡ ≈ A δs + B δF,  A = ∂f/∂s, B = ∂f/∂F  (s*, F* = 0)


UP = np.zeros(4)                         # yukarı denge
DOWN = np.array([0.0, 0.0, np.pi, 0.0])  # aşağı denge
C_X = np.array([[1.0, 0.0, 0.0, 0.0]])       # çıkış: x
C_THETA = np.array([[0.0, 0.0, 1.0, 0.0]])   # çıkış: θ


def analytic_AB(p: CartPoleParams, theta_star=0.0):
    """sympy Jacobian'ının (notebooks/02) kapalı biçimi; θ* = 0 (yukarı) veya π (aşağı).

    c = cos θ* = ±1 ile:  A[1,2] = −m g / M,  A[3,2] = c (M+m) g / (M l),  B = [0, 1/M, 0, −c/(M l)]ᵀ
    """
    if not np.isclose(np.sin(theta_star), 0.0):
        raise ValueError("F* = 0 ile denge yalnız θ* = 0 veya π")
    c = np.cos(theta_star)
    A = np.array([[0, 1, 0, 0],
                  [0, 0, -p.m * p.g / p.M, 0],
                  [0, 0, 0, 1],
                  [0, 0, c * (p.M + p.m) * p.g / (p.M * p.l), 0]], dtype=float)
    B = np.array([[0], [1 / p.M], [0], [-c / (p.M * p.l)]], dtype=float)
    return A, B


def numeric_AB(f, s_star, u_star, eps=1e-6):
    """Merkezi sonlu farklarla Jacobian: [f(s*+εeᵢ) − f(s*−εeᵢ)] / 2ε. Hata O(ε²)."""
    s_star = np.asarray(s_star, dtype=float)
    n = s_star.size
    A = np.zeros((n, n))
    for i in range(n):
        d = np.zeros(n)
        d[i] = eps
        A[:, i] = (f(s_star + d, u_star) - f(s_star - d, u_star)) / (2 * eps)
    B = ((f(s_star, u_star + eps) - f(s_star, u_star - eps)) / (2 * eps)).reshape(n, 1)
    return A, B


def discretize(A, B, dt):
    """ZOH ile tam ayrıklaştırma (Van Loan): exp([[A, B], [0, 0]]·dt) = [[A_d, B_d], [0, I]].

    A_d = e^{A dt},  B_d = (∫₀^dt e^{A τ} dτ) B
    """
    n, m = B.shape
    Z = np.zeros((n + m, n + m))
    Z[:n, :n] = A
    Z[:n, n:] = B
    E = expm(Z * dt)
    return E[:n, :n], E[:n, n:]


def ctrb_matrix(A, B):
    """Kontrol edilebilirlik matrisi 𝒞 = [B, AB, …, Aⁿ⁻¹B]."""
    cols = [B]
    for _ in range(A.shape[0] - 1):
        cols.append(A @ cols[-1])
    return np.hstack(cols)


def obsv_matrix(A, C):
    """Gözlemlenebilirlik matrisi 𝒪 = [C; CA; …; CAⁿ⁻¹]."""
    rows = [C]
    for _ in range(A.shape[0] - 1):
        rows.append(rows[-1] @ A)
    return np.vstack(rows)


def transfer_zeros(A, B, C):
    """SISO G(s) = C (sI − A)⁻¹ B → (sıfırlar, kutuplar, pay, payda).

    control.ss2tf payın başında ~1e-16'lık yuvarlama artığı bırakır → sahte, çok büyük bir "sıfır".
    Göreli olarak ihmal edilebilir baş katsayıları atıyoruz.
    """
    import control
    G = control.ss2tf(control.ss(A, B, C, np.zeros((C.shape[0], B.shape[1]))))
    num = np.array(G.num[0][0], dtype=float)
    den = np.array(G.den[0][0], dtype=float)
    num = np.trim_zeros(np.where(np.abs(num) < 1e-9 * np.max(np.abs(num)), 0.0, num), "f")
    return np.sort_complex(np.roots(num)), np.sort_complex(np.roots(den)), num, den


def analyze(A, B, C=C_X):
    """Özdeğerler, kontrol edilebilirlik rankı, C çıkışına göre sıfırlar."""
    zeros, poles, _, _ = transfer_zeros(A, B, C)
    return dict(eig=np.sort_complex(np.linalg.eigvals(A)),
                ctrb_rank=np.linalg.matrix_rank(ctrb_matrix(A, B)),
                zeros=zeros, poles=poles)


def simulate_linear(A_d, B_d, ds0, u, n_steps, s_star=UP):
    """Ayrık doğrusal model δs_{k+1} = A_d δs_k + B_d u_k. Dönüş: s* + δs, (n_steps+1, n)."""
    u = np.broadcast_to(np.atleast_1d(u), (n_steps,))
    S = np.empty((n_steps + 1, A_d.shape[0]))
    S[0] = ds0
    for k in range(n_steps):
        S[k + 1] = A_d @ S[k] + B_d[:, 0] * u[k]
    return S + s_star



# 4. Animasyon


def animate(t, S, p: CartPoleParams, U=None, fps=25, x_lim=2.5, save=None, show=True):
    """Simülasyon çıktısını gerçek zamanlı oynatır.

    Defterde:  HTML(animate(t, S, P, show=False).to_jshtml())
    Pencerede: animate(t, S, P)            GIF: animate(t, S, P, save="x.gif")
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    cart_w, cart_h = 0.3, 0.15
    stride = max(1, int(round(1.0 / (fps * (t[1] - t[0])))))
    idx = np.arange(0, len(t), stride)
    E = np.array([energy(s, p) for s in S])

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.set(xlim=(-x_lim, x_lim), ylim=(-p.l - 0.3, p.l + 0.3), aspect="equal", xlabel="x [m]")
    ax.axhline(-cart_h / 2, color="gray", lw=1)
    cart = plt.Rectangle((0, 0), cart_w, cart_h, fc="tab:blue", ec="k", zorder=3)
    ax.add_patch(cart)
    rod, = ax.plot([], [], "k-", lw=2, zorder=4)
    bob, = ax.plot([], [], "o", color="tab:red", ms=11, zorder=5)
    trail, = ax.plot([], [], "-", color="tab:red", lw=0.8, alpha=0.4)
    arrow, = ax.plot([], [], "-", color="tab:green", lw=3)
    txt = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top", family="monospace", fontsize=8)
    hx, hy = [], []

    def update(i):
        k = idx[i]
        x, _, th, _ = S[k]
        px, py = x + p.l * np.sin(th), p.l * np.cos(th)
        cart.set_xy((x - cart_w / 2, -cart_h / 2))
        rod.set_data([x, px], [0, py])
        bob.set_data([px], [py])
        if i == 0:
            hx.clear(), hy.clear()
        hx.append(px), hy.append(py)
        trail.set_data(hx, hy)
        F = U[min(k, len(U) - 1)] if U is not None and len(U) else 0.0
        arrow.set_data([x, x + 0.05 * F], [0, 0])   # 1 N → 5 cm
        th_w = np.degrees(np.arctan2(np.sin(th), np.cos(th)))
        txt.set_text(f"t={t[k]:5.2f}s  θ={th_w:+6.1f}°  x={x:+.3f}m  F={F:+.1f}N  E={E[k]:.4f}J")
        return cart, rod, bob, trail, arrow, txt

    anim = FuncAnimation(fig, update, frames=len(idx), interval=1000 / fps, blit=True)
    if save is not None:
        anim.save(save, writer=PillowWriter(fps=fps))
    if show and save is None:
        plt.show()
    else:
        plt.close(fig)
    return anim


H_XTHETA = np.array([[1.0, 0.0, 0.0, 0.0],      # 1. ölçüm: x
                     [0.0, 0.0, 1.0, 0.0]])     # 2. ölçüm: θ

def make_sensor(H, sigma, seed=0):
    """Gürültülü sensör: z = H·s + v,  v ~ N(0, σ²). seed: aynı gürültü dizisi yani tekrarlanabilir deney."""
    rng = np.random.default_rng(seed)
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (H.shape[0],))
    return lambda s: H @ s + rng.normal(0.0, sigma)


if __name__ == "__main__":
    # Hızlı deneme: python cartpole.py [θ0] [F]
    import sys
    th0 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
    F0 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    P = CartPoleParams()
    t, S, U = simulate(make_f(P), [0, 0, th0, 0], lambda t, s: F0, 0.02, 6.0)
    animate(t, S, P, U=U)


