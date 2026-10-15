import argparse, json, os, subprocess, time

import torch

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from envs import CartPoleContinuousEnv

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import evaluate as ev
from controllers import RLPolitika


ALGOS = {"ppo": PPO, "sac": SAC}
DEFAULT_STEPS = {"ppo": 300_000, "sac": 100_000}    # KARARLAR: 2026-10-15 revizyonu
SINAV_ARALIGI = {"ppo": 5_000, "sac": 2_000}

def arguments():
    p = argparse.ArgumentParser(description="Cartpole için PPO / SAC eğitimi")
    p.add_argument("--algo", choices=list(ALGOS), required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--distribution", default="zor")
    p.add_argument("--out", default="runs")
    p.add_argument("--sinav_araligi", type=int, default=None, help="boşsa SINAV_ARALIGI[algo]")
    return p.parse_args()


def git_surumu():
    """Kod sürümü: commit hash'i + commit'lenmemiş değişiklik var mı."""
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        kirli = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return h + ("-dirty" if kirli else "")
    except Exception:
        return "git-yok"


def config_yaz(config, run_dir):
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def main():
    torch.set_num_threads(1)   # küçük ağ: ek thread hız kazandırmıyor; paralel koşularda çekirdek yarışmasını önler
    a = arguments()
    adim = a.steps or DEFAULT_STEPS[a.algo]
    run_dir = os.path.join(a.out, f"{a.algo}_seed{a.seed}")
    os.makedirs(run_dir, exist_ok=True)

    env = make_vec_env(CartPoleContinuousEnv, n_envs=1, seed=a.seed,
                       env_kwargs=dict(distribution=a.distribution))


    if a.algo == "ppo":
        env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.99)

    # SB3 defaults
    model = ALGOS[a.algo]("MlpPolicy", env, gamma=0.99, seed=a.seed,
                            tensorboard_log=run_dir, device="cpu", verbose=1)

    # Config 
    config = dict(algo=a.algo, seed=a.seed, steps=adim, distribution=a.distribution, gamma=0.99,
                  git=git_surumu(),
                  baslangic=time.strftime("%Y-%m-%d %H:%M:%S"))
    config_yaz(config, run_dir)            # eğitimden ÖNCE: koşu çökse de ne koştuğu bilinsin

    # Periyodik sınav: zor geliştirme bloğu (50 senaryo, seed = 0), deterministik politika
    sinav = SinavCallback(a.sinav_araligi or SINAV_ARALIGI[a.algo], run_dir,
                          ev.senaryolar(a.distribution, 50, 0))

    # 5) Eğit ve süreyi ölç
    t0 = time.time()
    model.learn(total_timesteps=adim, tb_log_name="tb", callback=sinav)
    ttime = time.time() - t0

    # 6) Kaydet: model + (PPO ise) normalizasyon istatistikleri + duvar saati
    model.save(os.path.join(run_dir, "model"))
    if a.algo == "ppo":
        env.save(os.path.join(run_dir, "vecnormalize.pkl"))
    config["wall_time"] = ttime
    config["gercek_adim"] = int(model.num_timesteps)   # PPO bütçeyi n_steps katına yuvarlar
    config["duvar_egitim_s"] = sinav.kayit[-1]["duvar_egitim_s"]   # sınav süresi düşülmüş eğitim süresi
    config["en_iyi_basari"] = sinav.en_iyi
    config_yaz(config, run_dir)
    print(f"bitti: {run_dir}  ({ttime/60:.1f} dk)")


    
class SinavCallback(BaseCallback):
    def __init__(self, aralik, run_dir, blok):
        super().__init__()
        self.aralik, self.run_dir, self.blok = aralik, run_dir, blok
        self.kayit = []                # her sınav bir dict → sinav.csv
        self.en_iyi = -1               # şimdiye kadarki en yüksek başarı sayısı
        self.sinav_suresi = 0.0        # duvar saatinden düşülecek

    def _on_training_start(self):
        self.t_bas = time.time()
        self._sinav()                  # 0. adım: rastgele politikanın puanı

    def _on_step(self):
        if self.num_timesteps % self.aralik == 0:
            self._sinav()
        return True

    def _sinav(self):
        t0 = time.time()
        vecnorm = self.model.get_vec_normalize_env()          # PPO → VecNormalize, SAC → None
        satirlar = ev.degerlendir("rl", lambda: RLPolitika(self.model, vecnorm), self.blok)
        k = len([r for r in satirlar if r["basari"]])                                              # başarılı senaryo sayısı
        oturma = np.nanmedian([r["oturma"] for r in satirlar]) if k else np.nan   # hepsi NaN ise uyarı vermesin
        efor = float(np.median([r["efor"] for r in satirlar]))                   # bütün senaryolarda eforun medyanı (başarısızlarda da tanımlı)
        self.sinav_suresi += time.time() - t0

        self.kayit.append(dict(adim=self.num_timesteps, basari=k, n=len(self.blok),
                               oturma_med=oturma, efor_med=efor,
                               duvar_egitim_s=time.time() - self.t_bas - self.sinav_suresi))
        self.logger.record("sinav/basari_orani", k / len(self.blok))   # TensorBoard'a da
        self.logger.record("sinav/efor_med", efor)
        ev.csv_yaz(self.kayit, os.path.join(self.run_dir, "sinav.csv"))   # her sınavda yaz: koşu çökse de eğri kalır

        if k > self.en_iyi:            # en iyi modeli ayrıca sakla
            self.en_iyi = k
            self.model.save(os.path.join(self.run_dir, "model_en_iyi"))
            if vecnorm is not None:
                vecnorm.save(os.path.join(self.run_dir, "vecnormalize_en_iyi.pkl"))
        print(f"[sınav] adım {self.num_timesteps:>7d}: {k}/{len(self.blok)}  efor_med {efor:.1f}")

if __name__ == "__main__":
    main()


