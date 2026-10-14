"""Ortak değerlendirme düzeneği (Aşama 6). Kararlar: KARARLAR.md → Aşama 6."""
import numpy as np
from controllers import run_closed_loop
import cartpole as cp 
import csv

DT, T_BOLUM, X_SINIR = 0.02, 10.0, 2.4

# Senaryo dağılımları: [x, ẋ, θ, θ̇] için alt ve üst sınırlar
DAGILIMLAR = {
    "nominal": (np.array([-0.5, -0.1, -0.2, -0.1]), np.array([0.5, 0.1, 0.2, 0.1])),
    "zor":     (np.array([-1.0, -0.5, -0.6, -1.0]), np.array([1.0, 0.5, 0.6, 1.0])),
}


def senaryolar(dagilim, n, seed):
    """Tekrarlanabilir senaryo bloğu. Her senaryo: başlangıç durumu + kendi sensör tohumu."""
    alt, ust = DAGILIMLAR[dagilim]
    rng = np.random.default_rng(seed)
    blok = []
    for i in range(n):
        s0 = rng.uniform(alt, ust)                     # rng.uniform(alt, ust)
        sensor_seed = int(rng.integers(2**31))          # int(rng.integers(2**31)) → bu senaryonun gürültü dizisi
        blok.append(dict(id=i, s0=s0, sensor_seed=sensor_seed))
    return blok


def olcutler(t, S, U, tol=0.02):
    """Bir bölümün ölçütleri (KARARLAR.md'de yazılı)."""
    sonlu = bool(np.all(np.isfinite(S)))
    basari = (sonlu
              and np.abs(S[:, 0]).max() <= X_SINIR
              and abs(S[-1, 2]) < 0.05
              and abs(S[-1, 3]) < 0.1)
    disarida = np.where((np.abs(S[:, 0]) > tol) | (np.abs(S[:, 2]) > tol))[0]
    oturma = (float(t[disarida[-1]]) if len(disarida) else 0.0) if basari else np.nan   # en son dışarıda olunan an (hiç dışarıda değilse 0.0); başarısızsa np.nan
    return dict(basari=bool(basari),
                oturma=oturma,
                efor=float(np.sum(U**2) * DT),
                th_rms=float(np.sqrt(np.mean(S[:, 2]**2))) if sonlu else np.nan,
                max_x=float(np.nanmax(np.abs(S[:, 0]))))


def wilson(k, n, z=1.96):
    """Başarı oranı için %95 Wilson güven aralığı → (alt, üst)."""
    p = k / n
    payda = 1 + z**2 / n
    merkez = (p + z**2 / (2 * n)) / payda
    yaricap = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / payda
    return merkez - yaricap, merkez + yaricap

def degerlendir(ad, kontrolcu_yap, blok, sensor_sigma=None, params=None, F_max=10.0, T=T_BOLUM, substeps=5):
    """Bir kontrolcüyü bir senaryo bloğunda koşturur → her senaryo için bir satır (dict).
    kontrolcu_yap: her senaryoda TAZE kontrolcü üreten fonksiyon (lambda).  sensor_sigma=None → tam durum."""
    satirlar = []
    for sc in blok:
        sensor = None if sensor_sigma is None else cp.make_sensor(cp.H_XTHETA, sensor_sigma, seed=sc["sensor_seed"])
        with np.errstate(all="ignore"):
            t, S, U = run_closed_loop(kontrolcu_yap(), sc["s0"], T=T, dt=DT, params=params,
                                      F_max=F_max, substeps=substeps, sensor=sensor)
        satirlar.append(dict(kontrolcu=ad, senaryo=sc["id"], **olcutler(t, S, U)))
    return satirlar


def ozet(satirlar):
    """Kontrolcü başına özet: başarı (Wilson aralığıyla) ve başarılı bölümlerde medyan ölçütler."""
    tablo = {}
    for ad in dict.fromkeys(r["kontrolcu"] for r in satirlar):        # sırayı koruyarak tekil isimler
        R = [r for r in satirlar if r["kontrolcu"] == ad]
        k, n = sum(r["basari"] for r in R), len(R)
        lo, hi = wilson(k, n)
        tablo[ad] = dict(n=n, basari=k, oran=k / n, wilson_alt=lo, wilson_ust=hi,
                         oturma_medyan=np.nanmedian([r["oturma"] for r in R]),
                         efor_medyan=np.median([r["efor"] for r in R]),
                         max_x=np.nanmax([r["max_x"] for r in R]))
    return tablo


def ozet_yazdir(tablo):
    print(f"{'kontrolcü':16s} | {'başarı':>7s} | {'%95 Wilson':>13s} | {'oturma med':>10s} | {'efor med':>8s} | {'max|x|':>6s}")
    for ad, o in tablo.items():
        print(f"{ad:16s} | {o['basari']:3d}/{o['n']:<3d} | [{o['wilson_alt']:.2f}, {o['wilson_ust']:.2f}] | "
              f"{o['oturma_medyan']:8.2f} s | {o['efor_medyan']:8.1f} | {o['max_x']:6.2f}")


def csv_yaz(satirlar, yol):
    """Senaryo başına satırları CSV'ye yazar (sonradan analiz / RL ile karşılaştırma için)."""
    with open(yol, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(satirlar[0].keys()))
        w.writeheader(); w.writerows(satirlar)


def eslestirilmis(satirlar, a, b, olcut="oturma", n_boot=2000, seed=0):
    """A ile B'yi AYNI senaryolarda karşılaştırır. Başarı tablosu + ikisinin de başarılı olduğu senaryolarda
    (b − a) farkının ortalaması ve %95 bootstrap aralığı."""
    A = {r["senaryo"]: r for r in satirlar if r["kontrolcu"] == a}
    B = {r["senaryo"]: r for r in satirlar if r["kontrolcu"] == b}
    ortak = sorted(set(A) & set(B))
    sa = np.array([A[i]["basari"] for i in ortak]); sb = np.array([B[i]["basari"] for i in ortak])
    ikisi = sa & sb
    d = np.array([B[i][olcut] - A[i][olcut] for i, ok in zip(ortak, ikisi) if ok])
    rng = np.random.default_rng(seed)
    boot = [rng.choice(d, len(d)).mean() for _ in range(n_boot)]          # senaryoları yerine koyarak yeniden örnekle
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"{a} vs {b}: ikisi de başarılı {ikisi.sum()}, sadece {a} {(sa & ~sb).sum()}, sadece {b} {(~sa & sb).sum()}")
    print(f"  {olcut} farkı ({b} − {a}): ort {d.mean():+.3f}, %95 aralık [{lo:+.3f}, {hi:+.3f}], "
          f"{b} daha küçük olduğu senaryo oranı {np.mean(d < 0):.2f}")
    return d, (lo, hi)