"""Denetleyiciler. Hepsi aynı arayüzü kullanır: reset(), act(s) -> F."""
import numpy as np
import cartpole as cp 
from scipy.linalg import solve_continuous_are
from scipy.linalg import solve_discrete_are   
import cvxpy as cvx
import time

class PID:
    """Ayrık PID.  e = referans − ölçüm.  u = Kp·e + Ki·Σe·dt + Kd·ė,  |u| ≤ u_max."""

    def __init__(self, Kp, Ki=0.0, Kd=0.0, u_max=None, tau=0.0, anti_windup=True):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.u_max = u_max
        self.anti_windup = anti_windup
        self.tau = tau                # türev filtresi zaman sabiti (tau=0 → filtre yok)
        self.reset()

    def reset(self):
        self.integral = 0.0          # # Ki ile çarpılmış birikmiş integral
        self.prev_error = None       # önceki hata (ilk çağrıda yok)
        self.d_filt = 0.0            # filtrelenmiş türev (filtrenin hafızası)

    def __call__(self, error, dt, error_dot=None):
        # 1) Türev: error_dot verilmişse onu kullan; yoksa farkla hesapla.
        #    İlk çağrıda (prev_error None) türev tekmesi olmasın.
        if error_dot is None:
            error_dot = 0.0 if self.prev_error is None else (error - self.prev_error) / dt
        self.prev_error = error

        # 2) Türev filtresi (birinci dereceden lowpass)
        alpha = self.tau / (self.tau + dt)                 # tau = 0 → alpha = 0 → filtre yok
        self.d_filt = alpha * self.d_filt + (1 - alpha) * error_dot

        # 3) İntegralin ADAY değeri. Henüz kabul etmiyoruz!
        integral_new = self.integral + self.Ki * error * dt

        # 4) Doymamış çıktı (sınırı yokmuş gibi)
        u = self.Kp * error + integral_new + self.Kd * self.d_filt

        # 5) Doyma ve anti-windup
        if self.u_max is not None and abs(u) > self.u_max:          # sınır aşılıyor mu?
            hata_ayni_yonde = np.sign(error) == np.sign(u)          # hata u'yu daha da büyütmek mi istiyor?

            if not (self.anti_windup and hata_ayni_yonde):
                self.integral = integral_new                         # sadece işe yarıyorsa kabul et, güncellemeyi kabul ediyor muyuz etmiyor muyuz önce buna bakıyoruz
            u = np.clip(u, -self.u_max, self.u_max)                  # çıktıyı sınırla
        else:
            self.integral = integral_new                             # doyma yok → her zaman kabul

        return float(u)




class AnglePD:
    """Sadece açıya PD:  F = Kp·(θ − θ_ref) + Kd·θ̇. """

    def __init__(self, Kp, Kd, u_max=None, tau=0.0, dt=0.02, theta_ref=0.0):
        self.pid = PID(Kp=Kp, Ki=0.0, Kd=Kd, u_max=u_max, tau=tau)
        self.dt = dt
        self.theta_ref = theta_ref

    def reset(self):
        self.pid.reset()                                   # içteki PID'nin hafızasını sıfırla

    def act(self, s):
        theta, theta_dot = s[2], s[3]           # durumdan çek: s = [x, ẋ, θ, θ̇]
        error = theta - self.theta_ref        # ters etkili süreç: F>0 θ'yı azaltır → hata işareti çevrildi
        return self.pid(error, self.dt, error_dot=theta_dot)


def run_closed_loop(controller, s0, T, dt=0.02, params=None, F_max=10.0, substeps=20, sensor=None):
    """Kontrolcüyü sistemle çalıştırır. Motor sınırı |F| ≤ F_max SİSTEM tarafında uygulanır.
    Dönüş: t, S, U  (cp.simulate ile aynı)."""
    params = params if params is not None else cp.CartPoleParams()
    controller.reset()                                        # önceki deneyin hafızası sızmasın

    def u_fn(t, s):
        obs = sensor(s) if sensor is not None else s      # sensör varsa kontrolcü ölçümü görür yoksa gerçek durumu
        F = controller.act(obs)
        return np.clip(F, -F_max, F_max) if F_max is not None else F

    return cp.simulate(cp.make_f(params), s0, u_fn, dt, T, substeps=substeps)


class CascadePID:
    """Kaskad: dış döngü (x → θ_ref, yavaş) + iç döngü (θ → F, hızlı, AnglePD).
    θ_ref = k_px·(x_ref − x) − k_dx·ẋ   (doğrudan etkili: hedefe doğru yat, hıza karşı fren)"""

    def __init__(self, inner_Kp=40.0, inner_Kd=5.35, outer_Kp=0.1, outer_Kd=0.15,
                 theta_ref_max=0.2, u_max=None, tau=0.0, dt=0.02, x_ref=0.0):
        self.inner = AnglePD(Kp=inner_Kp, Kd=inner_Kd, u_max=u_max, tau=tau, dt=dt)
        self.outer = PID(Kp=outer_Kp, Ki=0.0, Kd=outer_Kd, u_max=theta_ref_max)      # çıktısı θ_ref → ±theta_ref_max'a kırpılır
        self.dt = dt
        self.x_ref = x_ref

    def reset(self):
        self.inner.reset()
        self.outer.reset()

    def act(self, s):
        x, x_dot = s[0], s[1]
        # Dış döngü: hata e = x_ref − x, türevi ė = −ẋ (x_ref sabit). Sonuç: θ_ref
        theta_ref = self.outer(self.x_ref - x, self.dt, error_dot=-x_dot)
        # İç döngü: hedef açıyı ver, kuvveti al
        self.inner.theta_ref = theta_ref
        return self.inner.act(s)





def lqr_continuous(A, B, Q, R):
    """Sürekli LQR:  AᵀP + PA − PBR⁻¹BᵀP + Q = 0,  K = R⁻¹BᵀP.  Dönüş: K (1×n), P (n×n)."""
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P) # # R⁻¹BᵀP
    return K, P


class LQRController:
    """F = −K·s.  Ortak arayüz: reset(), act(s)."""

    def __init__(self, K, s_ref=None):
        self.K = np.atleast_2d(K)
        self.s_ref = np.zeros(self.K.shape[1]) if s_ref is None else np.asarray(s_ref, float)

    def reset(self):
        pass          # LQR'nin hafızası yok: F yalnız şu anki duruma bağlı (PD gibi, PID gibi değil)

    def act(self, s):
        s = np.asarray(s, float)
        u = -self.K @ (s - self.s_ref)
        return float(u[0])                      # −K·(s − s_ref);  K 1×4 → sonuç 1 elemanlı dizi


def lqr_discrete(A_d, B_d, Q, R, dt):
    """Ayrık LQR (ZOH, dt). Maliyet her adımda × dt ile ölçeklenir (sürekli ∫(...)dt'nin ayrık hâli).
    Dönüş: K (1×n), P (n×n)."""
    Q_d, R_d = Q * dt, R * dt
    P = solve_discrete_are(A_d, B_d, Q_d, R_d) # # solve_discrete_are(A_d, B_d, Q_d, R_d)
    K = np.linalg.solve(R_d + B_d.T @ P @ B_d, B_d.T @ P @ A_d)
    return K, P

class MPCController:
    """Doğrusal MPC (kayan ufuk). Maliyet: Σ (sᵀQ_d s + uᵀR_d u) + s_Nᵀ P s_N,  Q_d = Q·dt, R_d = R·dt.
    Kısıtlar: s_{k+1} = A_d s_k + B_d u_k, |u_k| ≤ F_max, |x_k| ≤ x_max.
    P: lqr_discrete'in P'si (aynı Q, R, dt ile) → kısıtlar aktif değilken MPC = LQR.
    Çözüm bulunamazsa K_yedek ile LQR'ye düşer."""

    def __init__(self, A_d, B_d, Q, R, P, N, dt=0.02, F_max=10.0, x_max=2.4, K_yedek=None):
        n, m = B_d.shape
        self.s0 = cvx.Parameter(n)                    # HER ADIMDA güncellenecek tek şey: şu anki durum
        S = cvx.Variable((n, N + 1))                  # öngörülen durumlar s_0 … s_N  (sütun sütun)
        U = cvx.Variable((m, N))                      # planlanan kuvvetler u_0 … u_{N-1}

        maliyet = 0
        kisitlar = [S[:, 0] == self.s0]
        for k in range(N):
            maliyet += cvx.quad_form(S[:, k], Q * dt) + cvx.quad_form(U[:, k], R * dt)
            kisitlar += [S[:, k + 1] == A_d @ S[:, k] + B_d @ U[:, k]]       # model = öngörü
            if F_max is not None:
                kisitlar += [cvx.abs(U[:, k]) <= F_max]                     # kuvvet sınırı
            if x_max is not None:
                kisitlar += [cvx.abs(S[0, k + 1]) <= x_max]                 # ray sınırı (S[0] = x)
        maliyet += cvx.quad_form(S[:, N], P)                                 # terminal maliyet = V(s_N)

        self.problem = cvx.Problem(cvx.Minimize(maliyet), kisitlar)
        self.S, self.U = S, U
        self.K_yedek, self.F_max = K_yedek, F_max
        self.reset()

    def reset(self):
        self.cozum_sureleri = []                     # her act()'teki çözüm süresi [s] (Ders 4b.2'de ölçeceğiz)
        self.basarisiz = 0                          # kaç adımda çözüm bulunamadı

    def act(self, s):
        self.s0.value = np.asarray(s, dtype=float)
        t0 = time.perf_counter()
        self.problem.solve(solver=cvx.OSQP, warm_start=True, eps_abs=1e-7, eps_rel=1e-7, max_iter=20000)
        self.cozum_sureleri.append(time.perf_counter() - t0)

        if self.problem.status not in ("optimal", "optimal_inaccurate"):
            # Çözüm yok (kısıtlar sağlanamıyor) → güvenli yedek: kırpılmış LQR
            self.basarisiz += 1
            F = - self.K_yedek @ np.asarray(s, dtype=float) if self.K_yedek is not None else 0.0
            F = np.clip(F, -self.F_max, self.F_max) if self.F_max is not None else F
            return float(F)
        return float(self.U.value[0, 0])              # planın SADECE ilk kuvveti


class FiniteDiffLQR:
    """Sadece z = [x, θ] ölçülüyor. Hızlar sonlu farkla: (z_k − z_{k−1}) / dt. Saf yaklaşım (karşılaştırma için)."""

    def __init__(self, K, dt=0.02):
        self.K = np.atleast_2d(K)
        self.dt = dt
        self.reset()

    def reset(self):
        self.z_onceki = None                    # önceki ölçüm (ilk adımda yok)

    def act(self, z):
        x, th = z[0], z[1]
        if self.z_onceki is None:
            xd, thd = 0.0, 0.0                  # ilk adım: hız bilinmiyor = 0 
        else:
            xd  = (x - self.z_onceki[0]) / self.dt  # (x − önceki x) / dt
            thd = (th - self.z_onceki[1]) / self.dt

        z = np.array(z, dtype=float)
        self.z_onceki = z                     # bir sonraki adım için sakla
        s_tahmin = np.array([x, xd, th, thd])
        return float(-(self.K @ s_tahmin)[0])


class KalmanFilter:
    """Ayrık Kalman filtresi  s_{k+1} = A_d s_k + B_d u_k + w,  z_k = H s_k + v.
    w ~ N(0, Q_kf) (model hatası),  v ~ N(0, R_kf) (sensör gürültüsü)."""

    def __init__(self, A_d, B_d, H, Q_kf, R_kf, s0, P0):
        self.A, self.B, self.H = A_d, B_d, H
        self.Q, self.R = Q_kf, R_kf
        self.s0, self.P0 = np.asarray(s0, float), np.asarray(P0, float)
        self.reset()

    def reset(self):
        self.s = self.s0.copy()               # kestirim ŝ
        self.P = self.P0.copy()               # kestirimin kovaryansı (ne kadar emin değilim)
        self.gecmis_s, self.gecmis_P = [], [] # her update'ten sonra kaydedilir (tanılama için)

    def predict(self, u):
        """Tahmin: model konuşur, belirsizlik büyür."""
        self.s = self.A @ self.s + self.B @ np.atleast_1d(u)                           # A ŝ + B u   (u bir sayı  B[:, 0] * u)
        self.P = self.A @ self.P @ self.A.T + self.Q                          # A P Aᵀ + Q

    def update(self, z):
        """Düzeltme: ölçüm konuşur, belirsizlik azalır."""
        S = self.H @ self.P @ self.H.T + self.R                               # yenilik kovaryansı: H P Hᵀ + R
        K = np.linalg.solve(S, self.H @ self.P).T                             # P Hᵀ S⁻¹  , np.linalg.solve(S, H @ self.P).T  (simetri sayesinde)
        self.s = self.s + K @ (z - self.H @ self.s)                           # ŝ + K (z − H ŝ)
        #self.P = (np.eye(*self.P.shape) - K @ self.H) @ self.P                # (I − K H) P
        IKH = np.eye(len(self.s)) - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T     # Joseph formu: her zaman simetrik ve pozitif tanımlı kalır
        self.P = 0.5 * (self.P + self.P.T)                  
        self.gecmis_s.append(self.s.copy()); self.gecmis_P.append(self.P.copy())
        return self.s

class LQGController:
    """LQG = Kalman kestirimi + LQR:  F = −K·ŝ.  Ölçüm z = [x, θ] alır."""

    def __init__(self, K, kf, F_max=10.0):
        self.K = np.atleast_2d(K)
        self.kf = kf
        self.F_max = F_max
        self.reset()

    def reset(self):
        self.kf.reset()
        self.u_onceki = None                              # önceki adımda UYGULANAN kuvvet

    def act(self, z):
        if self.u_onceki is None:
            self.kf.s = np.array([z[0], 0.0, z[1], 0.0])  # ilk ölçümle başlat (hızlar bilinmiyor, 0)
        else:
            self.kf.predict(self.u_onceki)                # model: önceki kuvvetle bir adım ilerle
        s_hat = self.kf.update(z)                         # ölçümle düzelt
        u = -self.K @ s_hat                                             # −K·ŝ, ±F_max'a kırpılır (filtre GERÇEK kuvveti bilmeli)
        u = np.clip(u, -self.F_max, self.F_max)
        self.u_onceki = u
        return float(u)


class RLPolitika:
    """SB3 politikasını Aşama 6 düzeneğine uyduran sarmalayıcı: act(s) → F [N].
    vecnorm: PPO'nun VecNormalize nesnesi (SAC'ta None)."""
    def __init__(self, model, vecnorm=None, F_max=10.0):
        self.model, self.vecnorm, self.F_max = model, vecnorm, F_max

    def reset(self):
        pass                                     # LQR gibi: iç durumu yok (ağ hafızasız)

    def act(self, s):
        obs = np.asarray(s, dtype=np.float32)
        if self.vecnorm is not None:
            obs = self.vecnorm.normalize_obs(obs)        # eğitimdeki μ, σ ile normalize et — istatistikleri GÜNCELLEMEDEN
        a, _ = self.model.predict(obs, deterministic=True)  # SB3: a ∈ [−1, 1]
        return float(a[0]) * self.F_max   # a ∈ [−1, 1] → Newton  (ortam da tam böyle çeviriyor: envs.py step)