# Aşama 5 Ders Notları: Kalman Filtresi ve LQG

> Her bölüm: **Amaç → Sezgi → Matematik → Kodda nerede → Mülakatta ne derim.**
> Kod: `cartpole.py` (`H_XTHETA`, `make_sensor`), `controllers.py` (`FiniteDiffLQR`, `KalmanFilter`, `LQGController`,
> `run_closed_loop(..., sensor=)`), testler: `test_all.py` (Aşama 5), deneyler: `notebooks/05_kalman.ipynb`.
> Ders notu bağlantısı: **hafta 10–11 gözlemci** (x̂̇ = Ax̂ + Bu + K_e(y − ŷ)), **hafta 11 gözlemlenebilirlik**.

---

## İçindekiler
1. [Problem: gerçek sensörler](#1-problem-gerçek-sensörler)
2. [Sonlu fark neden çöküyor?](#2-sonlu-fark-neden-çöküyor)
3. [Kalman filtresi: sezgi](#3-kalman-filtresi-sezgi)
4. [Kalman filtresi: denklemler ve gözlemci bağlantısı](#4-kalman-filtresi-denklemler-ve-gözlemci-bağlantısı)
5. [Ölçülmeyen hız nasıl kestiriliyor?](#5-ölçülmeyen-hız-nasıl-kestiriliyor)
6. [Q_kf, R_kf ve NEES](#6-q_kf-r_kf-ve-nees)
7. [LQG ve ayrılma ilkesi](#7-lqg-ve-ayrılma-ilkesi)
8. [LQG'nin kırılganlığı: kestirici kontrolcüden hızlı olmalı](#8-lqgnin-kırılganlığı-kestirici-kontrolcüden-hızlı-olmalı)
9. [Sayısal kararlılık: Joseph formu](#9-sayısal-kararlılık-joseph-formu)
10. [Yazılım ve deney tasarımı dersleri](#10-yazılım-ve-deney-tasarımı-dersleri)
11. [Mülakat soruları ve örnek cevaplar](#11-mülakat-soruları-ve-örnek-cevaplar)
12. [Tuzaklar](#12-tuzaklar)
13. [Kaynak haritası](#13-kaynak-haritası)
14. [Kendini sına](#14-kendini-sına)

---

## 1. Problem: gerçek sensörler

Aşama 3–4b'de kontrolcüler gerçek durumu (s) kusursuz alıyordu. Gerçekte:
- **Ölçülen:** x (araba enkoderi), θ (mafsal enkoderi). **Ölçülmeyen:** ẋ, θ̇.
- Ölçülenler **gürültülü**.

$$ z = H s + v,\qquad H = \begin{bmatrix}1&0&0&0\\0&0&1&0\end{bmatrix},\qquad v \sim \mathcal N(0,\sigma^2 I) $$

H, ders notundaki **C**'nin aynısı (y = Cx). Aşama 2'de bu H ile **gözlemlenebilirlik rank 4**: ölçülmeyen hızları
ölçülenlerden çıkarmak teoride mümkün.

**Sensör kodu `cartpole.py`'de**, çünkü donanımın parçası, kontrolcünün değil. `make_sensor(H, σ, seed)`, seed sayesinde
tekrarlanabilir. Üç kontrolcüyü aynı gürültü dizisiyle karşılaştırmak (**ortak rastgele sayılar**) için şart.

`run_closed_loop(..., sensor=None)`: sensör verilirse kontrolcü **ölçümü** görür, verilmezse gerçek durumu.
Geriye uyumlu değişiklik: eski kodlar aynen çalışıyor (pytest ile doğrulandı).

---

## 2. Sonlu fark neden çöküyor?

θ̇ ≈ (θ_k − θ_{k−1}) / dt. İki bağımsız gürültünün farkı √2·σ, sonra dt'ye bölünüyor:

$$ \sigma_{\dot\theta} = \frac{\sqrt2\,\sigma}{dt} = \frac{1.41 \times 0.005}{0.02} = 0.354\ \text{rad/s} $$

**Ölçülen 0.348**, formülle birebir uyuşuyor. 0.3°'lik gürültü hızda **70 kat** büyüyor. LQR'nin θ̇ kazancı ~21 olduğu için
kuvvette ~7 N'luk titreme demek (sınır 10 N).

| σ | ideal LQR ∫F² | sonlu farklı LQR ∫F² | sonlu fark başarı (10 seed) |
|---|---|---|---|
| 0.001 | 5.9 | 59 (10×) | — |
| 0.005 | 5.9 | **550** (90×) | 10/10 (ama motor titriyor) |
| 0.01 | — | 755 | 6/10 |
| 0.02 | 5.9 | 976 | **0/10** |

**Ders notunda (hafta 10) bu cümle var:** *"Eğer sensörün çıktısı gürültülü ise, sensörün çıktısından türev almak mümkün
değildir."* Bu tablo onun sayısal kanıtı. Çözüm de aynı sayfada: **gözlemci.**

---

## 3. Kalman filtresi: sezgi

İki kusurlu bilgi kaynağı:
- **Model:** "Fizik kurallarına göre buradayım" (kusuru: bilinmeyen kuvvetler, doğrusallaştırma hatası).
- **Ölçüm:** "Sensör burada diyor" (kusuru: gürültülü, hızları söylemiyor).

**Tek boyut örneği:** model 22 °C (varyans 4), termometre 20 °C (varyans 1).
$$ K = \frac{P}{P+R} = 0.8,\qquad \hat T = 22 + 0.8(20 - 22) = 20.4,\qquad P_{\text{yeni}} = (1-K)P = 0.8 $$
Varyansların tersiyle ağırlıklı ortalama. **Birleşik belirsizlik (0.8) ikisinden de küçük.** İki kusurlu kaynak, her
birinden daha iyi bir tahmin veriyor.

**Sisli havada gemi benzetmesi:** hızdan ve rotadan konumu hesaplarsın (model), ara sıra bir fener görürsün (ölçüm).
Fener netse ona, puslu görünüyorsa hesabına güvenirsin.

---

## 4. Kalman filtresi: denklemler ve gözlemci bağlantısı

**Tahmin (model konuşur, belirsizlik büyür):**
$$ \hat s^- = A_d \hat s + B_d u,\qquad P^- = A_d P A_d^\top + Q_{kf} $$
**Düzeltme (ölçüm konuşur, belirsizlik azalır):**
$$ S = HP^-H^\top + R_{kf},\quad K = P^-H^\top S^{-1},\quad \hat s = \hat s^- + K(z - H\hat s^-),\quad P = (I-KH)P^-\ \text{(→ Joseph, Bölüm 9)} $$

- **z − Hŝ⁻: yenilik** (innovation). Ölçümün tahminden ne kadar farklı olduğu.
- **K:** "ölçüme ne kadar güveniyorum". 0 → hiç, büyük → çok.

**Gözlemciyle (H10–11) aynı yapı:**
$$ \hat x_{k+1} = A\hat x_k + Bu_k + K_e(y_k - C\hat x_k) \quad\Longleftrightarrow\quad \text{tahmin} + K(z - H\hat s) $$
Derste K_e'yi "A − K_eC şu kutuplara sahip olsun" diye **sen** seçiyordun. Kalman'da K, P'den, yani modele ve sensöre ne
kadar güvenildiğinden **hesaplanıyor**. Gürültü varsayımları doğruysa kestirim hatası varyansını en küçük yapan K bu.

**İkizlik (dualite):** Kalman'ın kararlı hâl P'si, LQR Riccati'sinin **ikizi**. A → Aᵀ, B → Hᵀ:
`solve_discrete_are(A_d.T, H.T, Q_kf, R_kf)`. Ders notundaki kontrol edilebilirlik ↔ gözlemlenebilirlik ikiliği
(H8 ↔ H11) burada LQR ↔ Kalman ikiliği oluyor. Yinelemeli filtrenin P'si bu çözüme yakınsıyor (test:
`test_kalman_covariance_converges_to_riccati`).

**Kod:**
```python
def predict(self, u):
    self.s = self.A @ self.s + self.B @ np.atleast_1d(u)    # u düz sayı da olabilir
    self.P = self.A @ self.P @ self.A.T + self.Q
def update(self, z):
    S = self.H @ self.P @ self.H.T + self.R
    K = np.linalg.solve(S, self.H @ self.P).T                # P Hᵀ S⁻¹ (simetri sayesinde)
    self.s = self.s + K @ (z - self.H @ self.s)
    IKH = np.eye(len(self.s)) - K @ self.H
    self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T         # Joseph formu
    self.P = 0.5 * (self.P + self.P.T)
```

---

## 5. Ölçülmeyen hız nasıl kestiriliyor?

**P'nin köşegen dışı elemanları sayesinde.** Model "x bir sonraki adımda ẋ·dt kadar değişecek" diyor. Tahmin adımı
P'de x ile ẋ arasında bir **kovaryans** kuruyor: "x'teki belirsizliğimin bir kısmı ẋ'teki belirsizliğimden geliyor."
Ölçüm x'in tahminden farklı olduğunu söyleyince, düzeltme bu bağ üzerinden **ẋ'yi de** düzeltiyor. K'nın ẋ ve θ̇
satırları sıfır değil. Gözlemlenebilirlik (rank 4) bunun **mümkün olduğunun** garantisi.

**Ölçüm (filtre tek başına, kaydedilmiş yörünge üzerinde):**

| | x | ẋ | θ | θ̇ |
|---|---|---|---|---|
| Kalman RMS hata | 0.0017 | 0.0079 | 0.0022 | **0.0164** |
| sonlu fark θ̇ hatası | | | | 0.348 |

**θ̇'da 21 kat iyileşme.** İlk ~0.1 s'deki sapma: hızlar bilinmiyor (P0'da varyans 1), filtre birkaç ölçümde "öğreniyor".

---

## 6. Q_kf, R_kf ve NEES

⚠ **LQR'nin Q, R'si ile karıştırma.** Kodda `Q_kf`, `R_kf`.

| | anlamı | seçimimiz |
|---|---|---|
| R_kf | sensör gürültü varyansı | veri sayfasından: σ²·I = 0.005² |
| Q_kf | modelin her adımda ne kadar yanılabileceği | modelde olmayan kuvvet σ_w: B_d B_dᵀ σ_w² (+ 1e-8·I) |

**Oran filtrenin karakterini belirliyor:** R_kf ↑ → sensöre az güven → K küçük → yumuşak ama **gecikmeli**.
Q_kf ↑ → modele az güven → K büyük → hızlı ama **gürültülü**.

### NEES (normalleştirilmiş kestirim hatası karesi)
$$ \text{NEES} = e^\top P^{-1} e,\qquad e = s - \hat s $$
Filtrenin **kendi belirsizliği hakkında dürüst olup olmadığını** ölçer. Tutarlı bir filtrede ortalama NEES = **durum
sayısı (4)**.
- **≫ 4:** aşırı özgüvenli (tehlikeli: kestirim yanlış ama emin, ölçümleri dinlemiyor).
- **≪ 4:** kötümser (güvenli ama gereğinden gürültülü kalabilir).
- Sadece simülasyonda ölçülebilir (gerçek durum gerekli).

| ayar | NEES | θ̇ hatası | yorum |
|---|---|---|---|
| doğru (σ_w = 0.3) | 2.05 | 0.016 | biraz kötümser: simülasyonda 0.3 N'luk bilinmeyen kuvvet **yok** |
| R_kf × 0.1 | **20.6** | 0.035 | sensöre fazla güveniyor, gürültüyü takip ediyor, aşırı özgüvenli |
| R_kf × 10 | 2.4 | 0.013 | sensöre az güveniyor; burada zarar vermedi, model zaten çok iyi |

---

## 7. LQG ve ayrılma ilkesi

**LQG = Kalman + LQR:** F = −K·ŝ.

**Ayrılma ilkesi:** doğrusal sistem ve Gauss gürültü için LQR'yi ve Kalman'ı **ayrı ayrı** tasarlayıp birleştirmek, bütün
problemi birlikte çözmekle **aynı** en iyi sonucu verir.
$$ \begin{bmatrix} s \\ e \end{bmatrix}_{k+1} = \begin{bmatrix} A - BK & BK \\ 0 & A - LH \end{bmatrix}\begin{bmatrix} s \\ e \end{bmatrix}_k
\;\Rightarrow\; \text{eig}(\text{LQG}) = \text{eig}(A-BK)\cup\text{eig}(A-LH) $$
**Blok üst üçgen** matrisin özdeğerleri köşegen bloklarınkiler. Ders notunda (H10–11) gözlemcili kontrolde kutupları
ayrı ayrı yerleştirebilmenin sebebi bu.

**Ölçüm:** 8 kapalı döngü özdeğeri = {0.638, 0.905, 0.968 ± 0.017i} ∪ {0.830 ± 0.128i, 0.950 ± 0.028i} ✓.

**`LQGController` tasarım kararları:**
| karar | neden |
|---|---|
| Filtreyi **ilk ölçümle başlat** (ŝ = [z₀, 0, z₁, 0]) | ŝ = 0 ve P0'da "x'ten eminim" denince filtre ilk ölçümlere inanmadı, ∫F² 8 → 40. **P0 başlangıç tahminine gerçek güveni yansıtmalı** |
| Kuvveti **kontrolcünün içinde kırp**, `predict`'e kırpılmış u'yu ver | Filtre **gerçekten uygulanan** kuvveti bilmeli. 40 N istenip 10 N uygulandıysa 40 N ile tahmin yanlış olur |
| Sıra: (önceki u ile) predict → update(z) → u = −Kŝ | ölçüm geldiğinde en güncel kestirimle karar ver |

---

## 8. LQG'nin kırılganlığı: kestirici kontrolcüden hızlı olmalı

**10 seed'li karşılaştırma** (Bryson LQR, başlangıç [0.3, 0, 0.1, 0]):

| σ | sonlu fark | LQG σ_w = 0.3 | LQG σ_w = 1.0 |
|---|---|---|---|
| 0.005 | 10/10, ∫F² 550 | 10/10, **8.4** | 10/10, 12.6 |
| 0.01 | 6/10, 755 | 8/10, 36.2 | **10/10**, 19.1 |
| 0.02 | 0/10, 976 | **1/10**, 110 | **10/10**, 46.3 |

**Teşhis, kutup hızları** (σ = ln|z|/dt):

| | hızlı kutuplar |
|---|---|
| LQR (A − BK) | −22.5, **−4.98**, −1.63, −1.63 |
| Kalman σ = 0.005, σ_w = 0.3 | −8.73, −8.73, −2.56, −2.56 ✅ |
| **Kalman σ = 0.02, σ_w = 0.3** | **−5.29**, −5.29, −2.28, −2.28 ❌ |
| Kalman σ = 0.02, σ_w = 1.0 | −8.06, −8.06, −2.50, −2.50 ✅ |

- Gürültü artınca (R_kf ↑) filtre ölçüme az güveniyor, **yavaşlıyor**. σ = 0.02, σ_w = 0.3'te kestirici (−5.3), kontrolcünün
  ikinci modu (−5.0) kadar yavaş. Kontrolcü eski bilgiyle karar veriyor ve sistem kararsızlaşıyor.
- **Doyle (1978), "Guaranteed Margins for LQG Regulators", özet: "There are none."** LQR'nin sonsuz kazanç / ≥ 60° faz
  payı garantisi LQG'de **yok**.
- **Aşama 3'teki kaskad kuralının aynısı:** iç döngü dış döngüden hızlı olmalı → **kestirici kontrolcüden hızlı olmalı.**
- **Çare: Q_kf'yi büyütmek** (σ_w = 1): "modele daha az güven, ölçümü hızlı takip et". Kestirici −8'e hızlanıyor. Bedeli:
  düşük gürültüde ∫F² 8.4 → 12.6 (daha çok gürültü geçiyor). **Takas: gürültü bastırma ↔ gecikme.**
- **Q_kf bir tasarım düğmesi:** sadece "gerçek rüzgâr" değil; doğrusallaştırma hatasını, modellenmemiş etkileri ve sağlamlık
  payını da temsil ediyor. Sistematik hâli: **Loop Transfer Recovery** (Doyle & Stein, 1979). Yapay süreç gürültüsü
  LQR'nin sağlamlık paylarını geri kazandırır.
- **İlk hipotez yanlıştı:** "filtre aşırı özgüvenli" açıklaması NEES ile **doğrulanmadı** (iki ayarda da NEES ≈ 2.2). Doğru
  açıklama kutup hızlarıyla bulundu. Hipotezi ölçerek eleme dersi.

---

## 9. Sayısal kararlılık: Joseph formu

**Hata (testle bulundu):** standart P = (I − KH)P⁻ matematiksel olarak doğru ama sayısal olarak kırılgan.
- Yuvarlama hataları (~1e-17) P'yi simetriklikten çıkarıyor.
- A'nın kararsız modu (|z| = 1.097) A P Aᵀ'de bunu her adımda büyütüyor.
- ~500. adımda asimetri 1e-5, en küçük özdeğer **negatif**, ~1000'de 10⁹, **1205. adımda çöküş**.
- Kontrol deneyleri tam 500 adımdı, yani sınırın kenarında. Fark edilmedi.

**Çözüm:**
$$ P = (I-KH)P^-(I-KH)^\top + KRK^\top,\qquad P \leftarrow \tfrac12(P + P^\top) $$
Matematiksel olarak aynı sonuç, ama "pozitif + pozitif" biçimi P'yi asla negatif yapamaz. **Joseph formuyla 5000 adım
sorunsuz, DARE çözümüne 3e-16 hassasiyetle yakınsıyor.**

**Ders:** "kısa testte çalışıyor" ≠ "doğru". Uzun çalışacak yinelemeli algoritmaları (filtreler, entegratörler, öğrenme
döngüleri) uzun süre test et.

---

## 10. Yazılım ve deney tasarımı dersleri

| ders | nerede |
|---|---|
| **Birden çok seed, medyan ve başarı oranı raporla** | σ = 0.02'de LQG tek seed'de "başarılı", 10 seed'de 1/10 |
| **Kestiriciyi kontrol döngüsüne bağlamadan önce tek başına test et** | kaydedilmiş yörünge üzerinde filtre: RMS ve NEES |
| **Ortak rastgele sayılar** | `make_sensor(..., seed)`, her kontrolcü aynı gürültüyü görüyor |
| **Geriye uyumlu değişiklik** | `run_closed_loop(sensor=None)`; ardından pytest |
| **Savunmacı yazım** | `np.atleast_1d(u)`, `np.array(z, dtype=float)` (referans saklama) |
| **Hipotezi ölç** | "aşırı özgüven" → NEES ile elendi; "kestirici yavaş" → kutuplarla doğrulandı |

---

## 11. Mülakat soruları ve örnek cevaplar

**S1. Hızı neden sonlu farkla hesaplamadın?**
Türev gürültüyü √2/dt kat büyütüyor: 0.005 rad gürültü hızda 0.35 rad/s oldu (ölçtüm, formülle uyuşuyor), kuvvette ±7 N
titreme. Sonlu farklı LQR ∫F²'yi 90 kat artırdı ve σ = 0.02'de hiç başaramadı. Kalman θ̇'yı 21 kat daha doğru kestirdi.

**S2. Kalman filtresinde Q ve R ne anlama geliyor, nasıl ayarladın?**
R sensör gürültüsü (veri sayfasından), Q modelin her adımdaki hatası (bilinmeyen kuvvet olarak modelledim). Oranları
filtrenin hızını belirliyor. Q'yu bir tasarım düğmesi olarak kullandım: yüksek gürültüde kestirici kontrolcüden yavaş
kalınca Q'yu büyütüp kestiriciyi hızlandırdım.

**S3. Filtrenin doğru çalıştığını nasıl anlarsın?**
Simülasyonda gerçek durumu bildiğim için RMS kestirim hatası ve NEES'e baktım. NEES ≈ durum sayısıysa filtre kendi
belirsizliği konusunda dürüst. R'yi 10 kat küçük verince NEES 20'ye çıktı (aşırı özgüven). Ayrıca kararlı hâl P'nin
Riccati çözümüne yakınsadığını test ettim. Bu test sayısal bir hatayı (Joseph formu gereksinimi) yakaladı.

**S4. Ölçmediğin bir durumu nasıl kestirirsin?**
Gözlemlenebilirlik şart (rank 4). Kalman'da P'nin köşegen dışı elemanları ölçülen ile ölçülmeyen durumlar arasındaki
bağı tutuyor; yenilik bu bağ üzerinden ölçülmeyenleri de düzeltiyor.

**S5. Ayrılma ilkesi nedir? LQG'nin zayıflığı ne?**
Doğrusal-Gauss durumda LQR ve Kalman ayrı tasarlanıp birleştirilebilir; kapalı döngü özdeğerleri ikisinin birleşimi
(doğruladım). Zayıflığı: LQR'nin sağlamlık payları korunmuyor (Doyle 1978). Bende yüksek gürültüde kestirici
yavaşlayınca sistem 10'da 9 kez kararsızlaştı; süreç gürültüsü kovaryansını büyüterek düzelttim (Loop Transfer Recovery).

**S6. Kalman ile senin derste gördüğün gözlemci arasındaki fark?**
Yapı aynı: tahmin + kazanç × yenilik. Luenberger'de kazancı hata dinamiğinin kutuplarını yerleştirerek seçiyorsun;
Kalman'da gürültü kovaryanslarından optimal olarak hesaplanıyor. Kararlı hâl Kalman kazancı, LQR Riccati'sinin ikizinden
geliyor.

**S7. Kalman'ı uzun süre çalıştırırken neye dikkat edersin?**
Kovaryans güncellemesinin sayısal kararlılığına. Standart formülle P 1200 adımda simetrikliğini ve pozitifliğini
kaybedip çöktü; Joseph formu ve simetrikleştirme ile 5000 adım sorunsuz.

---

## 12. Tuzaklar

| tuzak | belirti | önlem |
|---|---|---|
| `B @ u`, u düz sayı | `matmul` hatası | `np.atleast_1d(u)` |
| Filtreyi 0'dan başlatıp P0'da "eminim" demek | ilk ölçümlere inanmıyor, ∫F² 5 kat | ilk ölçümle başlat; P0 dürüst |
| `predict`'e kırpılmamış u | model yanlış ilerliyor | kontrolcü içinde kırp, uygulanan u'yu ver |
| Standart P güncellemesi | ~1200 adımda çöküş | Joseph + simetrikleştirme |
| Tek seed ile sonuç | "başarılı" sanılan 1/10 | 10 seed, medyan ve başarı oranı |
| LQR Q/R ile Kalman Q/R karışıklığı | yanlış matris | `Q_kf`, `R_kf` isimleri |
| "Gerçek gürültü kadar Q_kf" ısrarı | kestirici yavaş, LQG kararsız | Q_kf'yi tasarım düğmesi olarak büyüt |
| `z_onceki = z` (referans) | arabellek değişirse sessiz hata | `np.array(z, dtype=float)` |
| Doğrulanmamış hipotez | NEES açıklamayı desteklemedi | ölçerek ele, kutup hızlarına bak |

---

## 13. Kaynak haritası

| konu | kaynak |
|---|---|
| Gözlemci, gözlemlenebilirlik | Ders notları hafta 10–11; Brunton: "Observability", "Full-State Estimation" |
| Kalman filtresi | Brunton: "The Kalman Filter"; **Labbe, *Kalman and Bayesian Filters in Python*** → g-h, 1D, Multivariate Kalman |
| Gaussian, Bayes | Labbe: "Probabilities, Gaussians, and Bayes' Theorem", "Multivariate Gaussians" |
| NEES / filtre performansı | Labbe: "Evaluating Filter Performance" |
| LQG, ayrılma ilkesi | Brunton: "LQG"; Åström & Murray, "Output Feedback" |
| LQG sağlamlığı | Doyle (1978) "Guaranteed Margins for LQG Regulators"; Doyle & Stein (1979) LTR |
| Joseph formu | Labbe (Kalman denklemleri bölümü); herhangi bir sayısal Kalman kaynağı |

---

## 14. Kendini sına

1. σ = 0.002 ve dt = 0.01 için sonlu fark θ̇ gürültüsü kaç rad/s? dt'yi küçültmek yardım eder mi?
2. Tek boyut örneğinde termometrenin varyansı 4 olsaydı K ve birleşik varyans kaç olurdu?
3. Kalman ölçülmeyen ẋ'yi hangi matrisin hangi elemanları sayesinde düzeltiyor?
4. NEES ortalaması 12 çıktı. Ne düşünürsün, hangi ayarı değiştirirsin?
5. Ayrılma ilkesinin kapalı döngü matrisinde sol alt blok neden sıfır?
6. σ = 0.02'de LQG neden σ_w = 0.3 ile çöktü, σ_w = 1 ile çalıştı? Kutup sayılarıyla anlat.
7. Q_kf'yi sonsuz büyütürsen filtre neye dönüşür? Bu iyi bir fikir mi?
8. Filtreye kırpılmamış kuvvet verilseydi hangi durumda en çok zarar verirdi?
9. Standart kovaryans güncellemesi neden kararsız bir sistemde daha hızlı bozuluyor?
10. Doyle'un makalesinin başlığı ve özeti neydi, bu projede nerede gördün?
