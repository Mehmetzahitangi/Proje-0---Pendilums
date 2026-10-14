"""Bütün doğrulama testleri.  Çalıştır (proje kökünden):  python -m pytest -q test_all.py"""
from functools import lru_cache

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from controllers import PID, AnglePD, CascadePID, run_closed_loop, lqr_continuous, lqr_discrete, LQRController

from cartpole import (C_X, DOWN, UP, CartPoleParams, analytic_AB, analyze, ctrb_matrix, discretize,
                      energy, euler_step, f, make_f, numeric_AB, obsv_matrix, rk4_step, simulate,
                      simulate_linear)

P = CartPoleParams()
FN = make_f(P)
ZERO = lambda t, s: 0.0  # noqa: E731


# =============================================================================
# Aşama 1 — dinamik ve entegrasyon
# =============================================================================

def _rel_drift(S):
    E = np.array([energy(s, P) for s in S])
    return np.max(np.abs(E - E[0])) / abs(E[0])


def test_energy_conserved_rk4():
    """F = 0, sürtünme yok → enerji korunmalı. dt = 0.02'de 7e-6, alt adımlarla (dt/20) 1e-11."""
    _, S, _ = simulate(FN, [0, 0, 0.3, 0], ZERO, 0.02, 10.0, substeps=20)
    assert _rel_drift(S) < 1e-6


def test_energy_grows_with_euler():
    """Açık Euler sönümsüz salınıcıda enerji pompalar (|1 + iωdt| > 1)."""
    _, S, _ = simulate(FN, [0, 0, 0.3, 0], ZERO, 0.001, 10.0, step=euler_step)
    E = np.array([energy(s, P) for s in S])
    assert E[-1] > E[0] and _rel_drift(S) > 1e-3


@pytest.mark.parametrize("s_star", [UP, DOWN])
def test_equilibria(s_star):
    np.testing.assert_allclose(f(s_star, 0.0, P), 0.0, atol=1e-12)


def test_small_positive_theta_falls_right():
    _, S, _ = simulate(FN, [0, 0, 0.05, 0], ZERO, 0.01, 0.5)
    assert S[-1, 2] > 0.05 and S[-1, 3] > 0


def test_positive_force_cart_right_pole_left():
    ds = f(np.zeros(4), 1.0, P)
    assert ds[1] > 0 and ds[3] < 0                 # ẍ > 0, θ̈ < 0
    _, S, _ = simulate(FN, np.zeros(4), lambda t, s: 5.0, 0.01, 0.3)
    assert S[-1, 0] > 0 and S[-1, 2] < 0


@lru_cache(maxsize=None)
def _sympy_accelerations():
    """Bağımsız referans: Lagrange'ı sympy ile baştan türet, lambdify et (defter 01'in özeti)."""
    import sympy as sp
    t = sp.symbols("t")
    M, m, l, g, F = sp.symbols("M m l g F")
    x, th = sp.Function("x")(t), sp.Function("theta")(t)
    xp, yp = x + l * sp.sin(th), l * sp.cos(th)
    T = M * x.diff(t)**2 / 2 + m * (xp.diff(t)**2 + yp.diff(t)**2) / 2
    L = T - m * g * yp
    eqs = [L.diff(x.diff(t)).diff(t) - L.diff(x) - F, L.diff(th.diff(t)).diff(t) - L.diff(th)]
    xd, xdd, thd, thdd = sp.symbols("xd xdd thd thdd")
    for a, b in [(x.diff(t, 2), xdd), (th.diff(t, 2), thdd), (x.diff(t), xd), (th.diff(t), thd)]:
        eqs = [e.subs(a, b) for e in eqs]
    ths = sp.symbols("th")
    eqs = [e.subs(th, ths) for e in eqs]
    sol = sp.solve(eqs, [xdd, thdd], dict=True)[0]
    return sp.lambdify([xd, ths, thd, F, M, m, l, g], (sol[xdd], sol[thdd]), "numpy")


def test_matches_sympy_lambdify():
    acc = _sympy_accelerations()
    rng = np.random.default_rng(0)
    for _ in range(100):
        s = rng.uniform([-2, -3, -np.pi, -6], [2, 3, np.pi, 6])
        u = rng.uniform(-10, 10)
        ref = np.array(acc(s[1], s[2], s[3], u, P.M, P.m, P.l, P.g), dtype=float)
        np.testing.assert_allclose(f(s, u, P)[[1, 3]], ref, rtol=1e-9, atol=1e-9)


def test_dt_002_close_to_dt_0001():
    _, S_c, _ = simulate(FN, [0, 0, 0.3, 0], ZERO, 0.02, 2.0)
    _, S_f, _ = simulate(FN, [0, 0, 0.3, 0], ZERO, 0.001, 2.0)
    err = np.max(np.abs(S_c - S_f[::20]), axis=0)
    assert err[0] < 1e-3 and err[2] < 1e-3


# =============================================================================
# Aşama 2 — doğrusallaştırma ve analiz
# =============================================================================

def test_analytic_matches_ek_a2():
    A, B = analytic_AB(P)
    A_ref = np.array([[0, 1, 0, 0], [0, 0, -P.m * P.g / P.M, 0], [0, 0, 0, 1],
                      [0, 0, (P.M + P.m) * P.g / (P.M * P.l), 0]])
    B_ref = np.array([[0], [1 / P.M], [0], [-1 / (P.M * P.l)]])
    np.testing.assert_allclose(A, A_ref, atol=1e-12)
    np.testing.assert_allclose(B, B_ref, atol=1e-12)


@pytest.mark.parametrize("s_star", [UP, DOWN])
def test_analytic_equals_numeric(s_star):
    A, B = analytic_AB(P, s_star[2])
    A_n, B_n = numeric_AB(FN, s_star, 0.0)
    np.testing.assert_allclose(A_n, A, atol=1e-6)
    np.testing.assert_allclose(B_n, B, atol=1e-6)


def test_upright_eigs_ctrb_zeros():
    A, B = analytic_AB(P)
    r = analyze(A, B, C_X)
    lam = np.sqrt((P.M + P.m) * P.g / (P.M * P.l))
    np.testing.assert_allclose(np.sort(r["eig"].real), [-lam, 0, 0, lam], atol=1e-9)
    assert r["ctrb_rank"] == 4
    z = np.sqrt(P.g / P.l)
    np.testing.assert_allclose(np.sort(r["zeros"].real), [-z, z], atol=1e-6)


def test_downward_eigs_oscillatory():
    A, B = analytic_AB(P, np.pi)
    eig = np.linalg.eigvals(A)
    w = np.sqrt((P.M + P.m) * P.g / (P.M * P.l))
    np.testing.assert_allclose(eig.real, 0, atol=1e-9)
    np.testing.assert_allclose(np.sort(np.abs(eig.imag)), [0, 0, w, w], atol=1e-9)
    assert np.linalg.matrix_rank(ctrb_matrix(A, B)) == 4


def test_observability():
    A, _ = analytic_AB(P)
    assert np.linalg.matrix_rank(obsv_matrix(A, C_X)) == 4
    assert np.linalg.matrix_rank(obsv_matrix(A, np.array([[0, 0, 1.0, 0]]))) == 2


@pytest.mark.parametrize("u", [0.0, 2.0])
def test_discrete_matches_sampled_continuous(u):
    A, B = analytic_AB(P)
    dt, n = 0.02, 50
    A_d, B_d = discretize(A, B, dt)
    s0 = np.array([0.1, -0.2, 0.05, 0.3])
    S_d = simulate_linear(A_d, B_d, s0, u, n)
    te = np.arange(n + 1) * dt
    sol = solve_ivp(lambda t, s: A @ s + B[:, 0] * u, (0, te[-1]), s0, t_eval=te,
                    rtol=1e-12, atol=1e-12, method="DOP853")
    np.testing.assert_allclose(S_d, sol.y.T, rtol=1e-8, atol=1e-10)


def test_discretize_matches_python_control():
    import control
    A, B = analytic_AB(P)
    A_d, B_d = discretize(A, B, 0.02)
    sysd = control.c2d(control.ss(A, B, C_X, 0), 0.02, method="zoh")
    np.testing.assert_allclose(A_d, sysd.A, atol=1e-12)
    np.testing.assert_allclose(B_d, sysd.B, atol=1e-12)


def test_linear_close_to_nonlinear_for_small_angle():
    """θ₀ = 0.01, 0.5 s: doğrusal ve doğrusal olmayan model %1 içinde."""
    A, B = analytic_AB(P)
    A_d, B_d = discretize(A, B, 0.001)
    s0 = [0, 0, 0.01, 0]
    _, S_nl, _ = simulate(FN, s0, ZERO, 0.001, 0.5)
    S_lin = simulate_linear(A_d, B_d, np.array(s0), 0.0, 500)
    assert abs(S_lin[-1, 2] - S_nl[-1, 2]) / abs(S_nl[-1, 2]) < 0.01






def test_pid_integral():
    """Sabit hata 1, Ki=1, dt=0.1, 10 adım → integral 1·1·0.1·10 = 1.0"""
    pid = PID(Kp=0, Ki=1.0)
    for _ in range(10):
        u = pid(1.0, 0.1)
    assert abs(u - 1.0) < 1e-12          


def test_pid_saturation():
    """Kp=100, hata 5 → istenen 500, ama u_max=10 → iki yönde de 10'da kesilmeli"""
    pid = PID(Kp=100, u_max=10)
    assert pid(5.0, 0.02) == 10.0
    assert pid(-5.0, 0.02) == -10.0


def test_pid_no_derivative_kick():
    """İlk çağrıda önceki hata yok → türev 0 sayılmalı. Yoksa 1/0.02 = 50'lik bir sıçrama olurdu."""
    pid = PID(Kp=0, Kd=1.0)
    assert pid(1.0, 0.02) == 0.0


def _stuck_steps(anti_windup):
    """Yukarıdaki senaryo: 100 adım hata +1, sonra hata −1.
    Hata döndükten sonra çıktı kaç adım daha +1'de takılı kalıyor?"""
    pid = PID(Kp=0, Ki=1.0, u_max=1.0, anti_windup=anti_windup)
    for _ in range(100):
        pid(+1.0, 0.1)
    for k in range(1000):                # 1000: sonsuz döngüye karşı emniyet
        if pid(-1.0, 0.1) < 1.0:
            return k                     # k adım takılı kaldı
    return 1000


def test_pid_anti_windup():
    assert _stuck_steps(anti_windup=True) == 0     # hemen dönmeli
    assert _stuck_steps(anti_windup=False) > 80    # anti-windup yoksa ~90 adım takılı (karşılaştırma)


def test_pid_derivative_filter():
    """Sabit ham türev 1, tau=0.1 → τ saniye sonra filtre çıkışı 1 − e⁻¹ ≈ 0.632 olmalı"""
    pid = PID(Kp=0, Kd=1.0, tau=0.1)
    for _ in range(100):                             # 100 × 0.001 s = 0.1 s = τ
        u = pid(0.0, 0.001, error_dot=1.0)           # türevi DOĞRUDAN veriyoruz
    assert abs(u - 0.632) < 0.01


# Aşama 3 — AnglePD ve kapalı döngü


def test_anglepd_matches_function():
    """Sınıf (AnglePD + run_closed_loop) ile defterdeki fonksiyon (angle_pd + simulate) AYNI yörüngeyi vermeli.
    Fonksiyonu sınıfa çevirirken bir şeyi bozmadığımızın kanıtı. (tau = 0 → filtre yok → birebir aynı olmalı.)"""
    s0 = [0, 0, 0.1, 0]
    _, S_cls, U_cls = run_closed_loop(AnglePD(Kp=20, Kd=3), s0, T=2.0, F_max=None)
    _, S_fn, U_fn = simulate(FN, s0, lambda t, s: 20 * s[2] + 3 * s[3], 0.02, 2.0, substeps=20)
    np.testing.assert_allclose(S_cls, S_fn, atol=1e-12)
    np.testing.assert_allclose(U_cls, U_fn, atol=1e-12)


def test_run_closed_loop_respects_F_max():
    """Kontrolcü 200·θ istiyor (θ₀ = 0.3 → ilk anda 60 N). Motor sınırı 10 N → hiçbir adımda |F| > 10 olmamalı."""
    _, _, U = run_closed_loop(AnglePD(Kp=200, Kd=10), [0, 0, 0.3, 0], T=2.0, F_max=10.0)
    assert np.abs(U).max() <= 10.0
    assert np.abs(U).max() == 10.0      # sınıra gerçekten dayandı; dayanmasaydı bu test hiçbir şey kanıtlamazdı


def test_controller_reset():
    """Aynı kontrolcü nesnesiyle iki koşu birebir aynı olmalı, çünkü run_closed_loop başta reset() çağırıyor.

    reset() unutulsaydı bu test patlar mıydı? EVET, ama sadece tau > 0 iken:
    AnglePD'de Ki = 0 → integral birikmiyor; error_dot dışarıdan verildiği için prev_error kullanılmıyor.
    Hafızada kalan tek şey türev filtresinin d_filt değeri. tau = 0 olsaydı d_filt her adımda baştan
    yazılır ve reset eksikliği GÖRÜNMEZDİ. Bu yüzden testte tau > 0 seçtik.
    """
    ctrl = AnglePD(Kp=20, Kd=3, tau=0.05)
    s0 = [0, 0, 0.1, 0]
    _, S1, _ = run_closed_loop(ctrl, s0, T=0.3)
    _, S2, _ = run_closed_loop(ctrl, s0, T=0.3)
    np.testing.assert_array_equal(S1, S2)

    # Testin kendisini test et: reset'i ATLAYAN bir koşu farklı çıkmalı (yoksa yukarıdaki test bir şey kanıtlamaz)
    _, S3, _ = simulate(FN, s0, lambda t, s: ctrl.act(s), 0.02, 0.3, substeps=20)   # reset yok
    assert not np.allclose(S1, S3)


# =============================================================================
# Aşama 3 — Kaskad PID
# =============================================================================

def test_cascade_regulates_x_and_theta():
    """Yol haritasının doğrulama senaryosu: θ₀ = 0.1, x₀ = 0.5, motor sınırı AÇIK → hem x hem θ sıfıra."""
    _, S, U = run_closed_loop(CascadePID(), [0.5, 0, 0.1, 0], T=10.0, F_max=10.0)
    assert abs(S[-1, 0]) < 0.01 and abs(S[-1, 2]) < 0.01
    assert np.abs(U).max() <= 10.0


def test_cascade_theta_ref_is_clipped():
    """Dış döngünün çıktısı (θ_ref) theta_ref_max ile sınırlı olmalı. Araba çok uzaktayken (5 m)
    kırpılmasaydı θ_ref = 0.1·5 = 0.5 rad (29°) olurdu. Bu test 89. satırdaki u_max / theta_ref_max
    karışıklığını yakalar."""
    ctrl = CascadePID(theta_ref_max=0.2)
    assert ctrl.outer(-5.0, 0.02, error_dot=0.0) == -0.2


def test_cascade_far_start_does_not_fall():
    """Uzaktan başlangıç (x₀ = 3 m): θ_ref kırpıldığı için sarkaç hiçbir zaman ~0.3 rad'ı geçmemeli
    (43°'lük fiziksel sınırdan uzak) ve araba yine hedefe varmalı."""
    _, S, U = run_closed_loop(CascadePID(), [3.0, 0, 0.1, 0], T=15.0, F_max=10.0)
    assert np.abs(S[:, 2]).max() < 0.35
    assert abs(S[-1, 0]) < 0.01


def test_cascade_inverse_response():
    """Saf ters tepki: sarkaç dik ve hareketsiz, hedef 1 m sağda → araba ÖNCE SOLA gitmeli (x < 0),
    sonra hedefe varmalı. Aşama 2'deki RHP sıfırının (+√(g/l)) zaman domenindeki karşılığı."""
    _, S, _ = run_closed_loop(CascadePID(x_ref=1.0), [0, 0, 0, 0], T=10.0, F_max=10.0)
    assert S[:, 0].min() < -0.02          # ölçülen ≈ −0.063 m
    assert abs(S[-1, 0] - 1.0) < 0.01


def test_cascade_needs_fast_inner_loop():
    """Bant genişliği ayrımı: AYNI dış döngü, yavaş iç döngüyle (Kp=20, Kd=3; ω_n ≈ 4.3) çalışmamalı,
    hızlı iç döngüyle (Kp=40, Kd=5.35; ω_n ≈ 7.6) çalışmalı. Tasarım kuralının belgesi."""
    s0 = [0.5, 0, 0.1, 0]
    _, S_fast, _ = run_closed_loop(CascadePID(inner_Kp=40, inner_Kd=5.35), s0, T=10.0, F_max=10.0)
    _, S_slow, _ = run_closed_loop(CascadePID(inner_Kp=20, inner_Kd=3.0), s0, T=10.0, F_max=10.0)
    assert abs(S_fast[-1, 0]) < 0.01
    assert abs(S_slow[-1, 0]) > 0.1       # yavaş iç döngüyle oturmuyor


# =============================================================================
# Aşama 4 — LQR
# =============================================================================
Q_BRYSON = np.diag([4.0, 1.0, 25.0, 1.0])       # x ≤ 0.5 m, ẋ ≤ 1, θ ≤ 0.2 rad, θ̇ ≤ 1
R_BRYSON = np.array([[0.01]])                    # F ≤ 10 N
DT = 0.02


def test_lqr_continuous_matches_python_control():
    """Kendi lqr_continuous'ımız = control.lqr (bağımsız referans) ve kapalı döngü sol yarı düzlemde."""
    import control
    A, B = analytic_AB(P)
    K, Pm = lqr_continuous(A, B, Q_BRYSON, R_BRYSON)
    K_ref, _, _ = control.lqr(A, B, Q_BRYSON, R_BRYSON)
    np.testing.assert_allclose(K, K_ref, rtol=1e-8)
    assert np.all(np.linalg.eigvals(A - B @ K).real < 0)


def test_lqr_discrete_matches_dlqr_and_riccati_iteration():
    """lqr_discrete = control.dlqr = Riccati'nin elle yinelenmesi (değer iterasyonu). Kapalı döngü birim çemberin içinde."""
    import control
    A, B = analytic_AB(P)
    A_d, B_d = discretize(A, B, DT)
    K, Pm = lqr_discrete(A_d, B_d, Q_BRYSON, R_BRYSON, DT)
    K_ref, _, _ = control.dlqr(A_d, B_d, Q_BRYSON * DT, R_BRYSON * DT)
    np.testing.assert_allclose(K, K_ref, rtol=1e-8)
    assert np.all(np.abs(np.linalg.eigvals(A_d - B_d @ K)) < 1)

    Q_d, R_d = Q_BRYSON * DT, R_BRYSON * DT          # değer iterasyonu: P₀ = Q_d'den başla, yakınsayana kadar yinele
    P_it = Q_d.copy()
    for _ in range(20000):
        K_it = np.linalg.solve(R_d + B_d.T @ P_it @ B_d, B_d.T @ P_it @ A_d)
        P_new = Q_d + A_d.T @ P_it @ A_d - A_d.T @ P_it @ B_d @ K_it
        if np.abs(P_new - P_it).max() < 1e-12:
            break
        P_it = P_new
    np.testing.assert_allclose(P_it, Pm, rtol=1e-6)


def test_lqr_value_function_equals_accumulated_cost():
    """Değer fonksiyonu: ayrık doğrusal modelde gerçekten ödenen toplam ceza = s₀ᵀ P s₀. P pozitif tanımlı."""
    A, B = analytic_AB(P)
    A_d, B_d = discretize(A, B, DT)
    K, Pm = lqr_discrete(A_d, B_d, Q_BRYSON, R_BRYSON, DT)
    assert np.all(np.linalg.eigvalsh(Pm) > 0)
    s0 = np.array([0.3, 0.0, 0.05, 0.0])
    s, J = s0.copy(), 0.0
    for _ in range(5000):
        u = -K @ s
        J += s @ (Q_BRYSON * DT) @ s + u @ (R_BRYSON * DT) @ u
        s = A_d @ s + B_d @ u
    assert abs(J - s0 @ Pm @ s0) < 1e-9


def test_lqr_controller_regulates_and_tracks():
    """LQRController doğrusal olmayan modelde: (1) Aşama 3 senaryosunu kuvvet sınırı açıkken düzeltir,
    (2) s_ref ile 1 m'ye gider ve ters tepki gösterir (önce sola)."""
    A, B = analytic_AB(P)
    K, _ = lqr_discrete(*discretize(A, B, DT), Q_BRYSON, R_BRYSON, DT)
    _, S, U = run_closed_loop(LQRController(K), [0.5, 0, 0.1, 0], T=10.0, F_max=10.0)
    assert abs(S[-1, 0]) < 0.01 and abs(S[-1, 2]) < 0.01 and np.abs(U).max() <= 10.0
    _, S2, _ = run_closed_loop(LQRController(K, s_ref=[1, 0, 0, 0]), [0, 0, 0, 0], T=10.0, F_max=10.0)
    assert S2[:, 0].min() < -0.01 and abs(S2[-1, 0] - 1.0) < 0.01


# =============================================================================
# Aşama 4b — MPC
# =============================================================================
from controllers import MPCController


def _mpc_kur(N=10, **kw):
    A, B = analytic_AB(P)
    A_d, B_d = discretize(A, B, DT)
    K, Pm = lqr_discrete(A_d, B_d, Q_BRYSON, R_BRYSON, DT)
    return MPCController(A_d, B_d, Q_BRYSON, R_BRYSON, Pm, N=N, dt=DT, K_yedek=K, **kw), K


def test_mpc_equals_lqr_when_constraints_inactive():
    """Terminal maliyet = LQR'nin P'si ve kısıtlar aktif değil → MPC'nin ilk kuvveti = −K·s."""
    mpc, K = _mpc_kur()
    rng = np.random.default_rng(0)
    for _ in range(5):
        s = rng.uniform([-0.1, -0.1, -0.03, -0.1], [0.1, 0.1, 0.03, 0.1])
        assert abs(mpc.act(s) - float(-(K @ s)[0])) < 1e-3


def test_mpc_fallback_pushes_the_right_way():
    """Çözümsüz durum: araba rayın ucunda (x = 2.39) ve 3 m/s ile DIŞARI gidiyor → 10 N ile 1 cm'de durmak imkânsız.
    MPC 'infeasible' demeli, yedek LQR devreye girmeli ve doğru yönde (−K·s işaretiyle) itmeli."""
    mpc, K = _mpc_kur()
    s = np.array([2.39, 3.0, 0.0, 0.0])
    F = mpc.act(s)
    assert mpc.basarisiz == 1                                   # yedek plan çalıştı
    assert np.sign(F) == np.sign(float(-(K @ s)[0]))            # yedek LQR ile aynı yönde

# =============================================================================
# Aşama 5 — Kalman ve LQG
# =============================================================================
from controllers import KalmanFilter, LQGController, FiniteDiffLQR
from cartpole import H_XTHETA, make_sensor


def _kf_kur(sigma=0.005, sigma_w=0.3):
    A, B = analytic_AB(P)
    A_d, B_d = discretize(A, B, DT)
    Q_kf = B_d @ B_d.T * sigma_w**2 + 1e-8 * np.eye(4)
    R_kf = np.eye(2) * sigma**2
    kf = KalmanFilter(A_d, B_d, H_XTHETA, Q_kf, R_kf, s0=np.zeros(4), P0=np.diag([sigma**2, 1.0, sigma**2, 1.0]))
    return kf, A_d, B_d, Q_kf, R_kf


def test_kalman_predict_accepts_scalar_force():
    """predict(u) düz bir sayıyla çağrılabilmeli (LQGController böyle çağırıyor)."""
    kf, A_d, B_d, *_ = _kf_kur()
    kf.predict(2.0)
    np.testing.assert_allclose(kf.s, B_d[:, 0] * 2.0, atol=1e-12)


def test_kalman_covariance_converges_to_riccati():
    """Yinelemeli filtrenin P'si (tahmin sonrası), Kalman Riccati'sinin (LQR'nin ikizi: A→Aᵀ, B→Hᵀ) çözümüne yakınsamalı."""
    from scipy.linalg import solve_discrete_are
    kf, A_d, B_d, Q_kf, R_kf = _kf_kur()
    for _ in range(2000):
        kf.predict(0.0)
        P_prior = kf.P.copy()
        kf.update(np.zeros(2))
    P_dare = solve_discrete_are(A_d.T, H_XTHETA.T, Q_kf, R_kf)
    np.testing.assert_allclose(P_prior, P_dare, rtol=1e-6, atol=1e-12)


def test_lqg_separation_principle():
    """Ayrılma ilkesi: LQG kapalı döngü özdeğerleri = eig(A−BK) ∪ eig(A−LH)."""
    from scipy.linalg import solve_discrete_are
    kf, A_d, B_d, Q_kf, R_kf = _kf_kur()
    K, _ = lqr_discrete(A_d, B_d, Q_BRYSON, R_BRYSON, DT)
    H = H_XTHETA
    P_kf = solve_discrete_are(A_d.T, H.T, Q_kf, R_kf)
    L = A_d @ P_kf @ H.T @ np.linalg.inv(H @ P_kf @ H.T + R_kf)
    A_kd = np.block([[A_d - B_d @ K, B_d @ K], [np.zeros((4, 4)), A_d - L @ H]])
    birlesik = np.sort_complex(np.r_[np.linalg.eigvals(A_d - B_d @ K), np.linalg.eigvals(A_d - L @ H)])
    np.testing.assert_allclose(np.sort_complex(np.linalg.eigvals(A_kd)), birlesik, atol=1e-9)


def test_lqg_beats_finite_difference_under_noise():
    """Aynı gürültü dizisi (seed) ile: LQG başarılı ve ∫F² sonlu farktan çok küçük (kuvvet titremiyor)."""
    kf, A_d, B_d, *_ = _kf_kur(sigma=0.005)
    K, _ = lqr_discrete(A_d, B_d, Q_BRYSON, R_BRYSON, DT)
    s0 = [0.3, 0, 0.1, 0]
    _, S_l, U_l = run_closed_loop(LQGController(K, kf), s0, T=10.0, F_max=10.0, substeps=5,
                                  sensor=make_sensor(H_XTHETA, 0.005, seed=0))
    _, S_f, U_f = run_closed_loop(FiniteDiffLQR(K), s0, T=10.0, F_max=10.0, substeps=5,
                                  sensor=make_sensor(H_XTHETA, 0.005, seed=0))
    assert abs(S_l[-1, 2]) < 0.05 and np.abs(S_l[:, 0]).max() <= 2.4
    assert np.sum(U_l**2) * DT < 0.1 * np.sum(U_f**2) * DT

# =============================================================================
# Aşama 6 — Değerlendirme düzeneği
# =============================================================================
import evaluate as ev


def test_scenarios_reproducible():
    """Aynı tohum → aynı senaryolar; farklı tohum → farklı senaryolar."""
    a, b, c = ev.senaryolar("nominal", 5, seed=0), ev.senaryolar("nominal", 5, seed=0), ev.senaryolar("nominal", 5, seed=1)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x["s0"], y["s0"]); assert x["sensor_seed"] == y["sensor_seed"]
    assert not np.allclose(a[0]["s0"], c[0]["s0"])


def test_wilson_known_values():
    """50/50 → alt sınır ≈ 0.929 (asla 1.0 değil); 45/50 → ≈ [0.786, 0.957]."""
    lo, hi = ev.wilson(50, 50)
    assert abs(lo - 0.9287) < 1e-3 and hi <= 1.0 + 1e-12
    lo, hi = ev.wilson(45, 50)
    assert abs(lo - 0.7864) < 1e-3 and abs(hi - 0.9565) < 1e-3


def test_metrics_settling_time_in_seconds():
    """Oturma süresi SANİYE cinsinden (indis değil) ve başarısızlıkta NaN."""
    t = np.arange(501) * 0.02
    S = np.zeros((501, 4)); S[:100, 0] = 0.5          # x, 2 s boyunca dışarıda (son dışarıda indisi 99 → t = 1.98 s)
    U = np.zeros(500)
    m = ev.olcutler(t, S, U)
    assert m["basari"] and abs(m["oturma"] - 1.98) < 1e-9
    S[-1, 2] = 0.3                                     # sonda sarkaç yatık → başarısız
    assert np.isnan(ev.olcutler(t, S, U)["oturma"])

def test_evaluate_reproducible():
    """Aynı blok + aynı kontrolcü (gürültülü sensörle bile) → birebir aynı sonuçlar (ortak rastgele sayılar)."""
    from controllers import LQGController
    blok = ev.senaryolar("nominal", 3, seed=0)
    def yap():
        kf, A_d, B_d, *_ = _kf_kur(sigma=0.005, sigma_w=1.0)
        K, _ = lqr_discrete(A_d, B_d, Q_BRYSON, R_BRYSON, DT)
        return LQGController(K, kf)
    r1 = ev.degerlendir("LQG", yap, blok, sensor_sigma=0.005)
    r2 = ev.degerlendir("LQG", yap, blok, sensor_sigma=0.005)
    for a, b in zip(r1, r2):
        assert a["efor"] == b["efor"] and a["basari"] == b["basari"]

# =============================================================================
# Aşama 7 — Gymnasium ortamı
# =============================================================================
import envs


def test_env_passes_check_env():
    from gymnasium.utils.env_checker import check_env
    check_env(envs.CartPoleContinuousEnv(), skip_render_check=True)


def test_env_reset_reproducible():
    """Aynı tohum → aynı başlangıç (ve gürültülü gözlem); options['s0'] verilen durumu aynen kurar."""
    e1, e2 = envs.CartPoleContinuousEnv(obs_noise=0.01), envs.CartPoleContinuousEnv(obs_noise=0.01)
    o1, _ = e1.reset(seed=7); o2, _ = e2.reset(seed=7)
    np.testing.assert_array_equal(o1, o2)
    o3, _ = envs.CartPoleContinuousEnv().reset(seed=0, options={"s0": [0.1, 0.0, 0.2, 0.0]})
    np.testing.assert_allclose(o3, [0.1, 0.0, 0.2, 0.0], atol=1e-7)


def test_env_lqr_matches_run_closed_loop():
    """Ortamın içinden LQR (a = −K·o / F_max) = Aşama 4'teki run_closed_loop ile aynı yörünge.
    Ortamın fiziği, ölçeklemesi ve kırpması doğru taşıdığının kanıtı."""
    A, B = analytic_AB(P)
    K, _ = lqr_discrete(*discretize(A, B, DT), Q_BRYSON, R_BRYSON, DT)
    s0 = [0.3, 0.0, 0.25, -0.5]
    env = envs.CartPoleContinuousEnv()
    o, _ = env.reset(seed=0, options={"s0": s0})
    S_env = [np.array(s0)]
    for _ in range(200):
        a = np.clip(-(K @ o.astype(float)) / 10.0, -1, 1).astype(np.float32)
        o, r, te, tr, _ = env.step(a)
        S_env.append(env.s.copy())
        if te or tr:
            break
    _, S_ref, _ = run_closed_loop(LQRController(K), s0, T=len(S_env) * DT - DT, F_max=10.0, substeps=5)
    np.testing.assert_allclose(np.array(S_env), S_ref, atol=1e-5)


def test_env_reward_and_termination_flags():
    """Ödül her zaman (0, 1]; devrilme → terminated (truncated değil); süre dolması → truncated (terminated değil)."""
    env = envs.CartPoleContinuousEnv(max_adim=500)
    env.reset(seed=0, options={"s0": [0, 0, 0.5, 2.0]})               # hızla devrilen sarkaç, F = 0
    for _ in range(500):
        _, r, te, tr, _ = env.step(np.zeros(1, dtype=np.float32))
        assert 0 < r <= 1
        if te or tr:
            break
    assert te and not tr
    env = envs.CartPoleContinuousEnv(max_adim=3)
    env.reset(seed=0, options={"s0": [0, 0, 0, 0]})                   # tam dengede, 3 adımda süre dolar
    for _ in range(3):
        _, r, te, tr, _ = env.step(np.zeros(1, dtype=np.float32))
    assert tr and not te and r == 1.0
