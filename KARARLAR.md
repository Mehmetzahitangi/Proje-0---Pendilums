# KARARLAR

Her önemli karar tarihiyle. Deneylerden önce tahmin, sonra sonuç.

---

## 2026-10-07 — Aşama 1: model ve entegrasyon

**Açı kuralı:** θ = 0 yukarı, θ > 0 sağa yatık; x > 0 sağ; F > 0 sağa (Ek B).

**Model:** noktasal sarkaç kütlesi, kütlesiz çubuk, sürtünme yok. M = 1.0 kg, m = 0.1 kg, l = 0.5 m, g = 9.81 m/s².
Gymnasium `CartPole-v1` dağılmış kütleli çubuk kullanır, onunla aynı değil.

**Entegratör: RK4.** Euler sürtünmesiz sistemde enerji pompalıyor (F = 0, θ₀ = 0.3, 10 s):

| yöntem | max ‖ΔE/E₀‖ |
|---|---|
| Euler, dt = 0.02 | 2.2 |
| Euler, dt = 0.001 | 7.8e-2 |
| RK4, dt = 0.02 | 7.4e-6 |
| RK4, dt = 0.02, 20 alt adım | 1.8e-11 |

**Kontrol periyodu dt = 0.02 s (50 Hz), fizik alt adımı h = 0.001 s (`substeps=20`).**
Gerekçe: dt = 0.02'de RK4 hatası sarkacın en alttan hızlı geçtiği anlarda (|θ̇| ≈ 9 rad/s) sıçrıyor ve 1e-6 hedefini
aşıyor. Alt adımlar kontrol periyodunu değiştirmeden fiziği hassaslaştırır. Dik denge civarında (yavaş hareket)
dt = 0.02 tek adımla da yeterli: 5 s'de dt = 0.02 ile dt = 0.001 arasındaki fark θ'da 2e-4 rad.

## 2026-10-07 — Aşama 2: doğrusallaştırma

**Yukarı denge:** özdeğerler 0, 0, ±4.646 → kararsız; açı ~0.15 s'de ikiye katlanır. Simülasyondan ölçülen
log θ eğimi 4.57.
**Kontrol edilebilir** (rank 𝒞 = 4, det 𝒞 = g²/(M⁴l⁴) ≠ 0).
**F → x:** sıfırlar ±√(g/l) = ±4.43 → RHP sıfırı, ters tepki. z/p = √(M/(M+m)) = 0.95 < 1.
**Gözlemlenebilirlik:** yalnız x ölçülürse rank 4, yalnız θ ölçülürse rank 2.

**Doğrusal model geçerlilik eşiği:** θ'da %10 göreli hata = "ayrışma". Üç başlangıçta da (0.05, 0.2, 0.5 rad)
ayrışma θ ≈ 77–90°'de. x çok daha erken ayrışıyor.

## Aşama 4 — LQR
# Ders 4.1 — LQR fikri
(a) LQR’nin K’sındaki x ve ẋ katsayılarının işareti ne olacak? Kaskadın eşdeğer K’sı [−4, −6, −40, −5.35] idi.
    Artı olarak kalırlar sanırım.
(b) Hangisi daha hızlı oturur?
    PID daha hızlı oturur gibi çünkü LQR de bir de ayarlama süreci var.
(c) Hangisi daha çok toplam kuvvet harcar? Ölçü olarak ∫F² dt kullan: np.sum(U**2) * 0.02.
    Bilmiyorum.

1.  q_θ × 10'un yavaşlatacağını beklemezdim
    R*10 en şaşırtıcı bulgu oldu.
    Bundan sonra Q/R olarak: karşılaştırmalarda referans olarak Bryson, “gerçek uygulama” tercihi olarak R × 10 kullanacağım.

2. Fizik üzerinden birbirine bağlı. “Sarkacı dik tut” ile “arabayı hızlıca götür” birbiriyle çelişen hedefler. Birine verdiğin ağırlık ötekinin bedelini artırıyor.
    Doyma gerçekleşiyor,  R × 10'un ucuz olmasının sebebi de bu: kısılan kuvvet zaten işe yaramayan kuvvetti.

Bizim tanımımız (yol haritasındaki ortam sınırlarıyla uyumlu):

şart	neden
Sayılar sonlu (NaN/sonsuz yok)	Sınırsız kuvvette büyük açılar simülasyonu “patlatıyor” (sayılar taşıyor)
Bütün bölüm boyunca |x| ≤ 2.4 m	Ray bitiyor. Gerçek sistemde araba duvara çarpar
10 s sonunda |θ| < 0.05 rad ve |θ̇| < 0.1 rad/s	Sarkaç gerçekten dik ve sakin

Bu tanımı KARARLAR.md’ye yaz: Aşama 9'da RL’yi aynı tanımla ölçeceğiz. Tanım değişirse karşılaştırma çöker.

🎯 Adım 1: Tahmin (koşmadan, KARARLAR.md’ye)

Dört kontrolcü, θ₀ = 0°, 1°, 2°, … 90° (diğer durumlar sıfır). Hangi açıda ilk kez başarısız olur?

kontrolcü	F_max = 10 N	sınırsız kuvvet
AnglePD (20, 3)	?	?
Kaskad PID	?	?
LQR Bryson (R = 0.01)	?	?
LQR R × 10 (R = 0.1)	?	?

AnglePD     | F_max=10.0  | ilk başarısız açı: 5°
AnglePD     | F_max=None  | ilk başarısız açı: 5°
Kaskad      | F_max=10.0  | ilk başarısız açı: 36°
Kaskad      | F_max=None  | ilk başarısız açı: 50°
LQR Bryson  | F_max=10.0  | ilk başarısız açı: 32°
LQR Bryson  | F_max=None  | ilk başarısız açı: 61°
LQR R×10    | F_max=10.0  | ilk başarısız açı: 36°
LQR R×10    | F_max=None  | ilk başarısız açı: 57°
## 2026-10-13 — Aşama 6: değerlendirme düzeneği (kararlar)

| karar | seçim | gerekçe |
|---|---|---|
| Başarı tanımı | sonlu; bütün bölüm boyunca \|x\| ≤ 2.4 m; sonda \|θ\| < 0.05 rad ve \|θ̇\| < 0.1 rad/s | Aşama 4.4 ROA ile aynı; RL de bununla ölçülecek. "\|θ\| < 12° boyunca" şartı yok (zor blokta daha büyük başlangıçlar var) |
| Ölçütler | başarı, oturma süresi (x ve θ ±0.02'ye girip bir daha çıkmadığı an; başarısızsa NaN), efor ∫F² dt, θ RMS, en büyük \|x\| | Aşama 3–5 ile tutarlı |
| Bölüm süresi | 10 s (500 adım, dt = 0.02), fizik alt adımı 5 | substeps = 5 ile ROA sınırları 20 ile aynı çıkmıştı (Aşama 4.4) |
| Senaryo blokları | **nominal**: x₀ ∼ U(±0.5), ẋ₀ ∼ U(±0.1), θ₀ ∼ U(±0.2), θ̇₀ ∼ U(±0.1). **zor**: x₀ ∼ U(±1), ẋ₀ ∼ U(±0.5), θ₀ ∼ U(±0.6), θ̇₀ ∼ U(±1) | Prototipte nominal blokta bütün tam durumlu kontrolcüler 50/50 → ayırt etmiyor. Zor blok ROA kenarını yokluyor |
| Blok boyutları ve tohumlar | geliştirme: 50 senaryo, seed = 0. **Nihai: 200 senaryo, seed = 1000, projenin sonunda BİR KEZ bakılacak** | Ayar sırasında teste göre ayar yapmamak |
| Sensör koşulları | (a) tam durum, (b) z = [x, θ], σ = 0.005, her senaryonun kendi sensor_seed'i | Aşama 5; ortak rastgele sayılar |
| Belirsizlik | başarı oranı için %95 Wilson aralığı; ikili karşılaştırma eşleştirilmiş (aynı senaryo farkları), bootstrap aralığı | az örnekle "A > B" dememek |

## 2026-10-13 — Aşama 7: Gymnasium ortamı (kararlar)

| karar | seçim | gerekçe |
|---|---|---|
| Gözlem | [x, ẋ, θ, θ̇], float32; isteğe bağlı `obs_noise` (4 bileşene Gauss) | Tam durum temel. Yalnız x, θ ölçen kısmi gözlem tekrarlayan (recurrent) ağ ister → ileri bir projeye |
| Gözlem uzayı sınırları | **±∞** | hızların ve gürültünün doğal sınırı yok; bölümü bitiren son gözlem tanım gereği sınırın dışında. PPO/SAC sınırları kullanmıyor; ölçek için `VecNormalize` düşünülecek. `check_env` bunun için sezgisel bir uyarı veriyor (bilinçli tercih) |
| Aksiyon | `Box(−1, 1)`, ortam içinde F = clip(a, −1, 1) · 10 N | ağlar için ~1 ölçeği; motor sınırı ortamda (Aşama 3 kararı) |
| Ödül | **r = 1 / (1 + c)**, c = sᵀQs + R·F², Q = diag[4, 1, 25, 1], R = 0.01 (Bryson) | her zaman pozitif → hayatta kalmak her zaman ölmekten iyi; maliyetle sürekli azalıyor → dikten uzakta da öğrenme sinyali var |
| Reddedilen ödül A: r = −c·dt | — | "kötü ama hayatta" politika (c = 2.25, 500 adım) −22.5 alıyor, intihar (~19 adımda raydan çıkış) −20.2 → **erken bitirip cezadan kaçma teşviki** |
| Reddedilen ödül B: r = 1 − min(c, 2)/2 | — | c ≥ 2'de ödül 0, ölmekle aynı → dikten uzakta öğrenme sinyali yok |
| terminated | \|x\| > 2.4 veya \|θ\| > π/2 | ray ve yatayın ötesine devrilme. 12° değil: zor blokta 34°'ye kadar başlangıç var |
| truncated | 500 adım (10 s) | Aşama 6 ile aynı; süre dolması ≠ ölüm (bootstrapping) |
| Başlangıç dağılımı | eğitimde **zor** dağılım (nominali kapsıyor); `reset(options={"s0": ...})` ile belirli başlangıç | ROA kenarını görsün; Aşama 6 senaryolarını aynen oynatabilmek |
| Fizik | Aşama 1'in RK4'ü, dt = 0.02, substeps = 5 | Aşama 6 ile aynı; ~11 000 adım/s |
| RL'nin değerlendirilmesi | **ortamın ödülüyle değil, Aşama 6 düzeneğiyle** (başarı tanımı + Wilson + eşleştirme) | ödül eğitim sinyali, başarı tanımı ölçüt. `terminated` "sakin" şartını içermiyor (sonlu farklı LQR örneği) |

**Alt referanslar (zor geliştirme bloğu, 50 senaryo, bu ödülle):**
| politika | ort. ödül | ort. uzunluk | düştü |
|---|---|---|---|
| rastgele | 2.9 | 27.5 | 50/50 |
| sıfır | 4.0 | 29.6 | 50/50 |
| LQR (Bryson) | 384.2 | 426.3 | 8/50 |

## 2026-10-14 — Aşama 8 → Aşama 9 için RL eğitim kararları

| karar | seçim | gerekçe |
|---|---|---|
| Algoritmalar | **PPO ve SAC** (Stable-Baselines3 2.9) | on-policy ↔ off-policy karşılaştırması (yol haritası) |
| γ | **0.99** | etkin ufuk ~100 adım = 2 s; ters tepki süresini (~0.5 s, Ders 4.2 miyop kontrolcü) rahatça kapsıyor, 10 s bölümden kısa |
| PPO ayarları | SB3 varsayılanları: n_steps = 2048, batch = 64, epochs = 10, GAE λ = 0.95, clip ε = 0.2, lr = 3e-4, ağ [64, 64] | iyi bilinen başlangıç; Ders 8.2'nin parçaları |
| SAC ayarları | SB3 varsayılanları: buffer 1M, batch 256, lr = 3e-4, τ = 0.005, otomatik α (hedef entropi −1), ağ [256, 256] | aynı |
| Normalizasyon | PPO: `VecNormalize` (gözlem + ödül); SAC: yok | PPO ölçeğe duyarlı; SAC'ta buffer + kayan istatistik sorunu |
| Eğitim bütçesi (ilk tur) | PPO 1M adım, SAC 300k adım | öğrenme eğrisine göre ayarlanacak |
| Tohum | algoritma başına 5 | RL seed'e duyarlı (Agarwal ve ark. 2021) |
| Eğitim ortamı | `CartPoleContinuousEnv`, zor dağılım, tam durum, gürültüsüz | Aşama 7; gürültülü sürüm Aşama 12 |
| Değerlendirme | Aşama 6 düzeneği (zor + nominal geliştirme blokları), deterministik politika | ödül ≠ ölçüt; klasiklerle aynı terazi |
| Loglama | TensorBoard; her koşu `runs/<algo>_seed<k>/` | tekrarlanabilirlik |

**Tahmin (koşudan önce):** LQR seviyesine (zor blokta ~42/50) **daha az ortam adımıyla** da, **daha kısa duvar saatinde** de **SAC** ulaşır.
