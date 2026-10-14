# Aşama 8 Ders Notları: LQR'den RL'ye — Değer, Policy Gradient, PPO, SAC

> Kısa plan (3 ders). Her bölüm: **Köprü → Fikir → Deney → Mülakatta ne derim.**
> Deneyler: `notebooks/08_rl_theory.ipynb`. Kendi kodların: `C:\Zahit\RL Algorithm Designs` (REINFORCE, A2C,
> Continuous Action Space, Trust Regions/PPO, SAC, TRPO). Kararlar: `KARARLAR.md` → Aşama 8 → 9.

---

## İçindekiler
1. [Ders 8.1 — Bellman, V ve Q: LQR'den RL'ye](#1-ders-81--bellman-v-ve-q-lqrden-rle)
2. [Ders 8.1 — Veriden öğrenmek ve üç dert](#2-ders-81--veriden-öğrenmek-ve-üç-dert)
3. [Ders 8.2 — Policy gradient ve taban çizgisi](#3-ders-82--policy-gradient-ve-taban-çizgisi)
4. [Ders 8.2 — Avantaj ve GAE](#4-ders-82--avantaj-ve-gae)
5. [Ders 8.2 — PPO kırpması](#5-ders-82--ppo-kırpması)
6. [Ders 8.3 — Off-policy ve replay buffer](#6-ders-83--off-policy-ve-replay-buffer)
7. [Ders 8.3 — Aşırı tahmin ve iki Q ağı](#7-ders-83--aşırı-tahmin-ve-iki-q-ağı)
8. [Ders 8.3 — Entropi, α ve yeniden parametrelendirme](#8-ders-83--entropi-α-ve-yeniden-parametrelendirme)
9. [PPO ile SAC](#9-ppo-ile-sac)
10. [Aşama 9 kararları](#10-aşama-9-kararları)
11. [Kontrol ↔ RL sözlüğü](#11-kontrol--rl-sözlüğü)
12. [Mülakat soruları ve örnek cevaplar](#12-mülakat-soruları-ve-örnek-cevaplar)
13. [Kaynak haritası](#13-kaynak-haritası)
14. [Kendini sına](#14-kendini-sına)

---

## 1. Ders 8.1 — Bellman, V ve Q: LQR'den RL'ye

$$ V^\pi(s) = c(s,\pi(s)) + V^\pi(s'),\qquad Q^\pi(s,a) = c(s,a) + V^\pi(s'),\qquad \pi_{\text{yeni}}(s) = \arg\min_a Q(s,a) $$

LQR'de hepsi karesel. z = [s; a]:
$$ Q(s,a) = z^\top H z,\quad H = \begin{bmatrix} Q_d + A_d^\top P A_d & A_d^\top P B_d \\ B_d^\top P A_d & R_d + B_d^\top P B_d \end{bmatrix},\quad
a^* = -(R_d + B_d^\top P B_d)^{-1} B_d^\top P A_d\, s = -Ks $$

**Ders 4.2'deki K formülü = "Q'nun minimumu".** LQR'nin kazancı ile RL'nin "Q'ya göre açgözlü aksiyon" kuralı aynı şey.
Fark, Q'nun **modelden mi veriden mi** geldiğinde.

| V'yi öğrenmenin yolu | ne gerekiyor | RL karşılığı | kod |
|---|---|---|---|
| Dinamik programlama (Riccati) | model A, B | değer iterasyonu | Aşama 4.2 |
| Monte Carlo | tam bölümler, gerçekleşen getiri | getiri | `REINFORCE/` |
| TD | tek adımlık geçişler (s, a, c, s'), bootstrap | kritik | `A2C/`, `SAC/` |

**Öznitelik hilesi:** sᵀWs, W'nin elemanlarına göre doğrusal (v_i², 2v_iv_j). "Karesel fonksiyon öğrenmek" en küçük
karelere dönüşüyor.

---

## 2. Ders 8.1 — Veriden öğrenmek ve üç dert

| deney | sonuç |
|---|---|
| Doğrusal model, Monte Carlo | **N = 9: %59 hata, N = 10: 6e-14.** W'nin tam 10 bilinmeyeni var, veri hatasız |
| Doğrusal model, TD (2000 tek adım geçiş) | 7e-14 |
| **Modelsiz politika iterasyonu** (A, B'ye bakmadan, kaskadın K'sından başla) | 4 iterasyonda Riccati'nin K'sı [−14.71, −17.61, −96.15, −21.15]. Aktör (K) + kritik (H) = actor-critic'in en saf hâli (Bradtke 1994) |
| Aynısı **keşifsiz** (a = −Ks tam) | denklem rankı **10 / 15**; yeni K **[+4.0, +3.2, −0.96, +0.99]**, anlamsız |
| Doğrusal olmayan + gürültü, V'yi karesel kalıpla öğren | nominal σ_F = 0: ~%10; nominal σ_F = 1: ~%30–40; **zor: ~%100**, veri artınca da düşmüyor |

**Üç dert:**
| | dert | açıklama | algoritmaların cevabı |
|---|---|---|---|
| 1 | örnek verimliliği | gereken veri ≈ parametre sayısı × gürültü | replay buffer, taban çizgisi / avantaj |
| 2 | **keşif** | a hep −Ks ise "başka aksiyon yapsaydım" verisi yok → Q'nun a'ya bağlılığı öğrenilemez. Kontroldeki karşılığı: **sürekli uyarım** (persistent excitation, sistem tanılama) | Gauss politika (PPO), entropi (SAC), aksiyon gürültüsü (DDPG/TD3) |
| 3 | **fonksiyon yaklaştırma** | doğrusal olmayan bölgede gerçek V karesel değil (sinθ ≠ θ, doyma, ROA sınırında **uçurum**). Sorun veri değil, kalıp | sinir ağı (esnek ama veri aç, eğitim yavaş ve kararsız) |

**Gürültüde sabit terim:** stokastik sistemde V = sᵀPs + sabit. Kalıpta sabit yoksa hata kalır.

**Projeyle bağ:** LQR ve MPC zor bölgede tam olarak "dünya doğrusal, maliyet karesel" varsayımı yüzünden takıldı. Sinir
ağı kritiği gerçek, karesel olmayan V'yi öğrenebilir. Teoride; Aşama 9 pratiği gösterecek.

---

## 3. Ders 8.2 — Policy gradient ve taban çizgisi

**Köprü:** argmin Q kapalı formda bulunamıyorsa politikayı **gradyanla** iyileştir.

**Log-türev hilesi:**
$$ \nabla_\mu \mathbb E[r(a)] = \mathbb E\big[r(a)\,\nabla_\mu \log\pi(a)\big],\qquad \pi = \mathcal N(\mu,\sigma^2):\ \nabla_\mu\log\pi = \frac{a-\mu}{\sigma^2} $$
r'nin türevi gerekmiyor (kara kutu simülatör). "İyi sonuç veren aksiyonların olasılığını artır." `REINFORCE/`'daki
`loss = -log_prob * return`.

**Taban çizgisi:** (r − b), b ≈ ortalama. Beklenen gradyan aynı, varyans düşük. Soru "ödül yüksek mi?"den "ortalamadan
iyi mi?"ye dönüşüyor.

**Deney** (r = −(a − 2)², μ = 0, σ = 1, gerçek gradyan 4):
| N | tabansız std | tabanlı std |
|---|---|---|
| 10 | 2.99 | 1.94 |
| 100 | 0.93 | 0.63 |
| 1000 | 0.29 | 0.20 |

~%35 düşük std ≈ yarı örnek. **İncelik:** N = 10'da tabanlı ortalama 3.58 (yanlı). Taban çizgisi **aynı örneklerden**
hesaplandı. Gerçek algoritmalar bu yüzden ayrı bir modelden alıyor: **öğrenilmiş V(s), yani kritik.**

**Neden tabansız da öğrenir ama yavaş?** Ortalama yön doğru, tek tek tahminler savruk. Telafi: daha çok örnek ya da daha
küçük adım. (Sisli dağda titreyen pusula benzetmesi.)

---

## 4. Ders 8.2 — Avantaj ve GAE

**Avantaj:** A(s, a) = Q(s, a) − V(s), "bu aksiyon, bu durumda ortalamadan ne kadar iyi?"

| | formül | yanlılık | varyans |
|---|---|---|---|
| TD (tek adım) | δ_t = r_t + γV(s_{t+1}) − V(s_t) | var (V yanlışsa) | düşük |
| Monte Carlo | G_t − V(s_t) | yok | yüksek |
| **GAE(λ)** | Â_t = Σ (γλ)^k δ_{t+k} | λ ile ayarlı | λ ile ayarlı |

λ = 0 → TD, λ = 1 → Monte Carlo, pratikte 0.95. **Deney:** beş adımlık sahte bölümde λ = 0 satırı δ'ya, λ = 1 satırı G − V'ye
birebir eşit. Hesap **sondan başa** (Â_t = δ_t + γλÂ_{t+1}), Riccati iterasyonu gibi.

---

## 5. Ders 8.2 — PPO kırpması

**Problem:** gradyan yön veriyor, adım boyunu değil. Büyük adım → politika çöker → kötü veri → toparlanamaz. (8.1'deki
iterasyon 2. adımda K'yı hedefin ötesine fırlatmıştı.)

$$ \rho = \frac{\pi_{\text{yeni}}(a|s)}{\pi_{\text{eski}}(a|s)},\qquad L = \min\big(\rho\hat A,\ \text{clip}(\rho, 1-\varepsilon, 1+\varepsilon)\hat A\big) $$

**Grafik okuması:**
- Â > 0: ρ > 1 + ε'da düz (gradyan 0), "yeterince artırdın".
- Â < 0: ρ < 1 − ε'da düz, "yeterince azalttın".
- **Asimetri:** kötü aksiyon yanlışlıkla arttıysa (Â < 0, ρ > 1 + ε) düzleşmiyor; düzeltme hep açık. `min` = kötümser seçim.

Sonuç: aynı veriyle ~10 geçiş yapılabiliyor, politika eskisinden çok uzaklaşmıyor. TRPO güven bölgesinin basit yaklaşığı.

| ε | sonuç |
|---|---|
| 0.01 | güvenli ama öğrenme neredeyse durur |
| 0.1–0.3 | denge (genelde 0.2) |
| 2.0 | kırpma fiilen kapalı → düz PG + 10 epoch → dev adımlar → kararsızlık |

Genel ders: "güvenlik payı" parametrelerinde hep hız ↔ kararlılık takası var (Kalman σ_w, LQR R gibi).

---

## 6. Ders 8.3 — Off-policy ve replay buffer

- **PPO (on-policy):** sadece şu anki politikanın verisi; güncellemeden sonra atılıyor.
- **SAC (off-policy):** her geçiş havuzda (1M), rastgele örneklenip yüzlerce kez kullanılıyor.

**Neden mümkün?** TD denklemi Q(s, a) = c + V(s') **her** geçiş için geçerli, veriyi kim toplamış olursa olsun. 8.1 Hücre 3'te
gürültülü politikanın verisiyle K'nın Q'sunu doğru öğrendik: **Q öğrenmek doğası gereği off-policy.** PPO'nun gradyanı ise
∇log π üzerinden "bu politikanın aksiyonlarına" bağlı.

**Sonuç:** SAC daha az ortam adımı; bedeli güncelleme başına hesap ve bellek.

---

## 7. Ders 8.3 — Aşırı tahmin ve iki Q ağı

Gürültülü tahminlerin maksimumu yukarı yanlı. Gerçek max Q = 0, σ = 1:

| | tahmin |
|---|---|
| tek Q: E[max] (10 aksiyon) | **+1.54** |
| max min(Q₁, Q₂) (TD3 / SAC) | +0.68 |
| çift Q (Q₁ ile seç, Q₂ ile değerlendir) | ≈ 0.00 |

| aksiyon sayısı | 2 | 10 | 100 |
|---|---|---|---|
| E[max] | 0.56 | 1.54 | 2.51 |

**Tehlike:** Bellman hedefi kendi tahminini kullanıyor (bootstrap) → iyimserlik birikiyor → Q sonsuza kaçabiliyor (DDPG
kararsızlığı). **Çözüm:** iki Q, hedefte küçüğü. Yanlılık yarıdan fazla azalıyor. PPO'daki `min` ile aynı felsefe:
şüphedeysen kötümser ol.

---

## 8. Ders 8.3 — Entropi, α ve yeniden parametrelendirme

$$ \max\ \mathbb E\Big[\sum_t \gamma^t\big(r_t + \alpha\,\mathcal H(\pi(\cdot|s_t))\big)\Big] $$

- Keşif şansa bırakılmıyor, **amacın içinde** (8.1'deki keşif derdinin cevabı).
- α büyük → çok keşif, az performans; küçük → erken kesinleşme, yerel çözüm.
- **Otomatik α:** hedef entropi −(aksiyon boyutu) = −1; politika fazla kesinse α artar, fazla rastgeleyse azalır.
- **Yeniden parametrelendirme:** a = tanh(μ + σε), ε ∼ N(0, 1); gradyan öğrenilmiş Q'nun içinden geçiyor. 8.2'deki
  "kara kutu ödül" problemi kritik sayesinde kalkıyor → düşük varyans.

---

## 9. PPO ile SAC

| | PPO | SAC |
|---|---|---|
| veri | on-policy, kullan-at | off-policy, replay buffer |
| ortam adımı | çok | az |
| güncelleme başına hesap | ucuz | pahalı |
| kararlılık | kırpma ile çok kararlı | iki Q + entropi; hiperparametreye duyarlı olabilir |
| keşif | Gauss (σ öğreniliyor) | entropi, otomatik α |
| ne zaman | ucuz simülasyon, paralel ortam, kararlılık | pahalı ortam adımı, sürekli kontrol |

Bizim simülatör ucuz (~11 000 adım/s): PPO'nun veri israfı az önemli. SAC'ın sürekli kontroldeki gücü zor blokta fark
yaratabilir.

---

## 10. Aşama 9 kararları

PPO + SAC (SB3), γ = 0.99, SB3 varsayılanları, PPO'da `VecNormalize` (gözlem + ödül), SAC'ta yok, bütçe PPO 1M / SAC 300k,
5 seed, zor dağılım + tam durum, değerlendirme Aşama 6 düzeneğiyle, TensorBoard + `runs/<algo>_seed<k>/`. Ayrıntı ve
gerekçeler: `KARARLAR.md`.

**γ = 0.99 neden?** Etkin ufuk 1/(1 − γ) = 100 adım = 2 s. Ders 4.2: ufuk ters tepki süresini (~0.5 s ≈ 25 adım) kapsamazsa
politika miyop olur ("araba sağda → sola it", sarkaç düşer).

**Tahmin (koşudan önce):** hem daha az ortam adımıyla hem daha kısa duvar saatinde **SAC**.

---

## 11. Kontrol ↔ RL sözlüğü

| kontrol | RL |
|---|---|
| maliyet c = sᵀQs + uᵀRu | (eksi) ödül |
| kontrolcü K | politika π (aktör) |
| V(s) = sᵀPs | değer fonksiyonu (kritik) |
| Riccati iterasyonu | değer iterasyonu (dinamik programlama) |
| K = (R + BᵀPB)⁻¹BᵀPA | argmin_a Q(s, a) |
| ufuk N, terminal maliyet (MPC) | γ, bootstrap |
| sürekli uyarım (sistem tanılama) | keşif |
| LQR'nin kararlılık garantisi | (genelde) yok |
| model (A, B) | simülatör / deneyim |

---

## 12. Mülakat soruları ve örnek cevaplar

**S1. LQR ile RL arasındaki ilişki ne?**
LQR'de V(s) = sᵀPs ve K, Q fonksiyonunun minimumu; Riccati iterasyonu değer iterasyonu. Bunu veriden öğrendim: doğrusal
modelde Monte Carlo ve TD P'yi 10 bölümle buluyor, modelsiz politika iterasyonu A ve B'ye bakmadan Riccati'nin K'sına 4
iterasyonda ulaşıyor. Keşifsiz problem rank-eksik, doğrusal olmayan bölgede karesel kalıp yetersiz. Derin RL'nin keşif
mekanizmaları ve sinir ağı kritiği bu iki sorunun cevabı.

**S2. PPO ile SAC'ın farkı ne, hangisini ne zaman seçersin?**
PPO on-policy, veriyi atıyor, kırpmayla kararlı; simülasyon ucuzsa ve paralelleştirilebiliyorsa iyi. SAC off-policy,
replay buffer ile örnek verimli, iki Q ile aşırı tahmini azaltıyor, entropiyle keşfi canlı tutuyor; ortam adımı pahalıysa
ve sürekli kontrolde iyi.

**S3. PPO'daki kırpma ne işe yarıyor?**
Oran 1 ± ε dışına çıkınca amaç düzleşiyor, gradyan sıfır; aynı veriyle birçok güncelleme yapılırken politika eskisinden
çok uzaklaşmıyor. Kötümser min sayesinde kötü aksiyonun yanlışlıkla artışı hep düzeltiliyor. ε küçükse yavaş, büyükse
kararsız.

**S4. On-policy ve off-policy ne demek?**
On-policy, gradyan şu anki politikanın verisini gerektiriyor (∇log π). Off-policy, TD ile Q öğrenildiği için her geçiş
kullanılabiliyor; veriyi kimin topladığı önemli değil.

**S5. Keşfi nasıl sağlıyorsun? Neden gerekli?**
Keşifsiz Q'nun aksiyona bağlılığı öğrenilemiyor: LQR deneyimde denklem rankı 15'ten 10'a düştü ve K anlamsızlaştı. PPO'da
Gauss gürültüsü, SAC'ta entropi terimi ve otomatik α.

**S6. SAC'ta neden iki Q ağı?**
Gürültülü Q'ların maksimumu yukarı yanlı (10 aksiyonda +1.54σ) ve bootstrap ile birikiyor. İki Q'nun küçüğünü kullanmak
yanlılığı yarıdan fazla azaltıyor.

**S7. Değer fonksiyonu ile LQR'nin P matrisi arasında ne ilişki var?**
P, doğrusal-karesel durumda değer fonksiyonunun kapalı formu: V(s) = sᵀPs. Simülasyonla ödenen toplam cezanın s₀ᵀPs₀'a eşit
olduğunu ve veriden öğrenilen karesel V'nin P'ye yakınsadığını doğruladım.

---

## 13. Kaynak haritası

| konu | kaynak |
|---|---|
| MDP, Bellman, DP | Sutton & Barto bölüm 3–4; Spinning Up "Key Concepts" |
| Policy gradient, REINFORCE, taban çizgisi | Sutton & Barto bölüm 13; Spinning Up "Intro to Policy Optimization"; `REINFORCE/` |
| Actor-critic, GAE | Schulman ve ark. (2016) GAE; `A2C/` |
| PPO | Spinning Up PPO; Schulman ve ark. (2017); **"The 37 Implementation Details of PPO"** |
| DDPG / TD3, aşırı tahmin | Spinning Up DDPG, TD3; `Continuous Action Space/` |
| SAC | Spinning Up SAC; Haarnoja ve ark. (2018) iki makale; `Trust Regions/SAC` |
| LQR için Q-öğrenme | Bradtke (1994); Tedrake LQR bölümü |
| Az seed ile raporlama | Agarwal ve ark. (2021) |

---

## 14. Kendini sına

1. LQR'nin K formülünü "Q'nun minimumu" olarak türet (H'nin bloklarından).
2. Monte Carlo ile V'yi öğrenmek için neden en az 10 bölüm gerekti? 5 boyutlu bir sistemde kaç olurdu?
3. Keşif gürültüsünü sıfırlayınca neden rank 15'ten 10'a düşüyor? Hangi 5 yön belirsiz kalıyor?
4. Zor bölgede karesel V'nin hatası veri artınca neden düşmüyor?
5. Taban çizgisi beklenen gradyanı neden değiştirmez? (b sabitken E[b·∇log π] = ?)
6. GAE'de λ = 0 ve λ = 1 neye eşit, neden pratikte 0.95?
7. PPO'da Â < 0 ve ρ = 1.5 iken gradyan sıfır mı? Neden?
8. SAC eski veriyi neden kullanabiliyor da PPO kullanamıyor?
9. 100 aksiyonlu bir problemde aşırı tahmin neden 10 aksiyonludan büyük?
10. γ = 0.9 seçseydin cartpole'da ne bekledin? Ders 4.2 ile ilişkilendir.
