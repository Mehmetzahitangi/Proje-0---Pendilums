import argparse, json, os, subprocess, time

import torch

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from envs import CartPoleContinuousEnv

ALGOS = {"ppo": PPO, "sac": SAC}
DEFAULT_STEPS = {"ppo": 1_000_000, "sac": 300_000}


def arguments():
    p = argparse.ArgumentParser(description="Cartpole için PPO / SAC eğitimi")
    p.add_argument("--algo", choices=list(ALGOS), required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--distribution", default="zor")
    p.add_argument("--out", default="runs")
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

    # 5) Eğit ve süreyi ölç
    t0 = time.time()
    model.learn(total_timesteps=adim, tb_log_name="tb")
    ttime = time.time() - t0

    # 6) Kaydet: model + (PPO ise) normalizasyon istatistikleri + duvar saati
    model.save(os.path.join(run_dir, "model"))
    if a.algo == "ppo":
        env.save(os.path.join(run_dir, "vecnormalize.pkl"))
    config["wall_time"] = ttime
    config["gercek_adim"] = int(model.num_timesteps)   # PPO bütçeyi n_steps katına yuvarlar
    config_yaz(config, run_dir)
    print(f"bitti: {run_dir}  ({ttime/60:.1f} dk)")


if __name__ == "__main__":
    main()
