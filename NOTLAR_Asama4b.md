# Aşama 4b Ders Notları: MPC (Model Öngörülü Kontrol)

> Her bölüm: **Amaç → Sezgi → Matematik → Kodda nerede → Mülakatta ne derim.**
> Kod: `controllers.py` (`MPCController`), testler: `test_all.py` (Aşama 4b), deneyler: `notebooks/04b_mpc.ipynb`.
> Parametreler: M = 1, m = 0.1, l = 0.5, g = 9.81, F_max = 10 N, x_max = 2.4 m, dt = 0.02 s,
> Q = diag[4, 1, 25, 1], R = 0.01 (Bryson). Önceki aşama: `NOTLAR_Asama4.md`.

---

## İçindekiler
1. [MPC fikri: kayan ufuk](#1-mpc-fikri-kayan-ufuk)
2. [Matematik: kısıtlı, sonlu ufuklu LQR](#2-matematik-kısıtlı-sonlu-ufuklu-lqr)
3. [Terminal maliyet ve MPC = LQR](#3-terminal-maliyet-ve-mpc--lqr)
4. [Kod: MPCController](#4-kod-mpccontroller)
5. [Kayan ufku gözle görmek](#5-kayan-ufku-gözle-görmek)
6. [Ufuk N ve gerçek zaman bütçesi](#6-ufuk-n-ve-gerçek-zaman-bütçesi)
7. [Kuvvet sınırı senaryosu ve ablasyon dersi](#7-kuvvet-sınırı-senaryosu-ve-ablasyon-dersi)
8. [Çekim bölgesi: neden sadece 2°?](#8-çekim-bölgesi-neden-sadece-2)
9. [MPC, LQR ve RL karşılaştırması](#9-mpc-lqr-ve-rl-karşılaştırması)
10. [Mülakat soruları ve örnek cevaplar](#10-mülakat-soruları-ve-örnek-cevaplar)
11. [Tuzaklar](#11-tuzaklar)
12. [Kaynak haritası](#12-kaynak-haritası)
13. [Kendini sına](#13-kendini-sına)

---

## 1. MPC fikri: kayan ufuk

**Amaç:** LQR kuvvet sınırını ve ray sınırını bilmiyor; F = −Ks hesaplanıp sonradan kırpılıyor. Bedeli çekim bölgesinde
görüldü (sınırsızken 61°, 10 N ile 32°). MPC kısıtları **planın içine** koyuyor.

Her kontrol adımında (her 0.02 s):
1. **Öngör:** şu anki durumdan başlayarak N adımı modelle hesapla.
2. **Planla:** N adımlık maliyeti en küçük yapan kuvvet dizisini u₀ … u_{N−1}, **kısıtlara uyarak** bul.
3. **Sadece ilk hamleyi oyna:** u₀'ı uygula, gerisini at.
4. Bir sonraki adımda yeni ölçümle **baştan** yap.

**Satranç benzetmesi:** birkaç hamle ilerisini düşün, en iyi planı seç, ilk hamleyi oyna, rakip oynayınca yeniden düşün.
Planın geri kalanını atmak israf değil, **geri besleme**: her adımda yeni ölçümle plan düzeltiliyor.

---

## 2. Matematik: kısıtlı, sonlu ufuklu LQR

$$ \min_{u_0..u_{N-1}} \sum_{k=0}^{N-1} \big( s_k^\top Q_d s_k + u_k^\top R_d u_k \big) + s_N^\top P\, s_N $$
$$ s_0 = \text{ölçülen durum},\quad s_{k+1} = A_d s_k + B_d u_k,\quad |u_k| \le F_{\max},\quad |x_k| \le x_{\max} $$

| parça | nereden tanıdık |
|---|---|
| Σ (sᵀQ_ds + uᵀR_du), Q_d = Q·dt | Ders 4.2'deki ayrık LQR maliyeti, N adımla sınırlı |
| s_{k+1} = A_d s_k + B_d u_k | Aşama 2'nin ayrık modeli. MPC'nin "öngörüsü" |
| \|u_k\| ≤ F_max, \|x_k\| ≤ x_max | **yeni:** LQR'nin bilmediği kısıtlar |
| s_Nᵀ P s_N | Ders 4.2'deki değer fonksiyonu V(s) = sᵀPs |

**Problem tipi:** karesel maliyet + doğrusal kısıtlar = **karesel program (QP)**. Dışbükey, tek bir global çözümü var,
hızlı çözülür. Kurulum `cvxpy`, çözücü OSQP.

---

## 3. Terminal maliyet ve MPC = LQR

**Terminal maliyetin anlamı:** MPC sadece N adım görüyor. N. adımdan sonrası için ödenecek bedel "LQR devralırsa
ödeyeceğim toplam ceza" = V(s_N) = s_Nᵀ P s_N. **Ufkun ötesinin değerini LQR'nin değer fonksiyonuyla tahmin ediyoruz.**
Ders 4.2'deki miyop kontrolcü sorununun çözümü: ufuk kısa olsa da terminal maliyet sayesinde sonsuz ufuk "görülüyor".

**Sonuç:** kısıtlar aktif değilken (küçük sapmalar) MPC'nin ilk kuvveti **LQR'nin −K·s'si ile birebir aynı**. Çünkü
kısıtsız problem, Bellman denkleminin N kez açılmış hâli; terminal maliyet de tam sonsuz ufuklu değer.

**Ölçüm (5 rastgele küçük durum):** fark ~1e-5, bir satırda 4e-4. OSQP yinelemeli (ADMM) bir çözücü; `eps=1e-7`
kısıt ihlali eşiği, kuvvette 1e-7 doğruluk değil. Testte tolerans **1e-3**.

**Ölçek tuzağı:** P, adım maliyetiyle **aynı ölçekte** olmalı. `lqr_discrete` P'yi Q·dt, R·dt ile hesapladığı için MPC'de
de Q·dt, R·dt kullanılır. Uyuşmazsa terminal maliyet ile adım maliyetleri "farklı para biriminde" olur ve MPC = LQR
testi tutmaz.

**MPC, LQR'nin kısıtları bilen genellemesidir.**

---

## 4. Kod: MPCController

```python
self.s0 = cvx.Parameter(n)                         # HER ADIMDA değişen tek şey
S = cvx.Variable((n, N + 1)); U = cvx.Variable((m, N))
maliyet += cvx.quad_form(S[:, k], Q * dt) + cvx.quad_form(U[:, k], R * dt)
kisitlar += [S[:, k + 1] == A_d @ S[:, k] + B_d @ U[:, k], cvx.abs(U[:, k]) <= F_max, cvx.abs(S[0, k + 1]) <= x_max]
maliyet += cvx.quad_form(S[:, N], P)
...
self.problem.solve(solver=cvx.OSQP, warm_start=True, eps_abs=1e-7, eps_rel=1e-7, max_iter=20000)
return float(self.U.value[0, 0])                   # planın SADECE ilk kuvveti
```

**Tasarım kararları:**
| karar | neden |
|---|---|
| Problem **bir kez** kurulur, başlangıç durumu `cvx.Parameter` | Her adımda yeniden kurmak 10–100 kat yavaş |
| `warm_start=True` | Ardışık planlar çok benzer; önceki çözümden başlamak hızlandırır |
| Yüksek hassasiyet (`eps=1e-7`) | Varsayılanla MPC = LQR ancak 3. basamakta tutar |
| **Yedek plan**: çözümsüzse kırpılmış −K·s | Sert ray kısıtı bazı durumlarda sağlanamaz (rayın ucunda dışarı hızla gidiş). Kontrolcü susamaz |
| `cozum_sureleri`, `basarisiz` sayaçları, `reset()` ile sıfırlanır | Gerçek zaman ölçümü ve tanılama |
| (öneri) `try / except cvx.SolverError` | Çözücü hata fırlatırsa kontrolcü çökmemeli |

**Yedek plandaki işaret hatası (gerçekten yaşandı):** `F = self.K_yedek @ s` yazılmıştı, doğrusu **−K·s**. Normal
çalışmada **hiç görünmüyor**, sadece çözücü başarısız olunca, yani tam en zor anda, ters yöne itiyor. Testi:
`test_mpc_fallback_pushes_the_right_way`. x = 2.39, ẋ = 3 ile çözümsüzlüğü **bilerek** tetikliyor ve yönü kontrol ediyor.
**Kural:** nadir çalışan kod yolları (hata işleme, yedek planlar) en az test edilen ve en çok hata barındıran yerlerdir;
onları bilerek tetikleyen testler yaz.

---

## 5. Kayan ufku gözle görmek

**Tek plan (θ₀ = 0.3 rad ≈ 17°, N = 30):**
- İlk 6 adım (0.12 s) **tam 10 N**. MPC sınırı biliyor, planı sınırın içinde. LQR burada ~30 N isteyip kırpılırdı.
- Sonra kuvvet **negatif** (≈ −4 N): sarkaç dikleşince araba frenleniyor. Öngörülen θ da 0'ın altına iniyor (−0.12):
  sarkacı biraz ters yatırmak frenin kendisi (yatıklık = gaz pedalı, Aşama 3).

**Kayan ufuk (6 adım):** ardışık planlar neredeyse üst üste. Gerçek θ (doğrusal olmayan sistem) planların **tam
üstünde**. 17°'de doğrusal öngörü modeli çok iyi.

---

## 6. Ufuk N ve gerçek zaman bütçesi

Senaryo x₀ = 0.5, θ₀ = 0.1, 5 s. Süreler ilk çağrı hariç.

| N | ufuk | ort. | %95 | en kötü | ilk çağrı | oturma |
|---|---|---|---|---|---|---|
| 5 | 0.10 s | 1.32 ms | 1.69 ms | 2.14 ms | 17.6 ms | 2.50 s |
| 10 | 0.20 s | 1.84 ms | 2.36 ms | 2.71 ms | 27.7 ms | 2.50 s |
| 30 | 0.60 s | 5.37 ms | 7.87 ms | 8.82 ms | 72.5 ms | 2.50 s |
| 60 | 1.20 s | 14.06 ms | 19.73 ms | **27.20 ms** | 147.2 ms | 2.50 s |

1. **Süre N ile büyüyor** (QP boyutu ∝ N). **N = 60 bütçeyi aşıyor**: ortalama sığıyor gibi ama en kötü 27 ms.
   **Gerçek zamanlı sistemde ortalama değil en kötü durum önemli.**
2. **İlk çağrı** = cvxpy'nin problemi çözücü biçimine derlemesi. Bir kez, kontrol döngüsünden önce ödenir.
3. **Oturma N'den bağımsız (2.50 s = LQR'ninki).** Terminal maliyet ufkun ötesini zaten hesaba katıyor; kısıt bu
   senaryoda sadece başta kısa süre aktif. **Kural:** N'i kısıtların aktif kaldığı süreyi kapsayacak kadar seç, fazlası
   gereksiz. Bizde N = 10–30.
4. Gerçek uygulamada MPC Python'da çalışmaz. Çözücü C koduna derlenir (OSQP kod üretimi, `acados`) ve milisaniyenin
   altına iner.

---

## 7. Kuvvet sınırı senaryosu ve ablasyon dersi

Senaryo: x₀ = 1.8 m, ẋ₀ = +1 m/s (rayın sağ ucuna doğru), sarkaç dik. N = 30.
- **İkisi de** ilk 0.5 s'de aynı şeyi yapıyor, en fazla ~2.3 m'ye gidiyor.
- **LQR** −10 N'u **0.75 s** tutuyor (kırpıldığından habersiz). Sarkaç ~1 rad'a yatıyor, kontrol kayboluyor, araba +10 N
  ile kaçıp raydan **sonradan** çıkıyor.
- **MPC** −10 N'u **0.6 s'de** bırakıyor, sarkaç en fazla 0.51 rad, araba 0'a oturuyor.

**Ablasyon:** MPC'yi `x_max=None` ile tekrar koşunca sonuç **birebir aynı** (en büyük x 2.294; ray kısıtı hiç aktif
olmadı). **Bu senaryoda MPC'yi kurtaran ray bilgisi değil, kuvvet sınırını bilmesi.** Raydan çıkmak sebep değil sonuç.

**Kural:** "X sayesinde oldu" demeden önce X'i kapatıp tekrar dene. Sonuç değişmiyorsa sebep X değildir. (İlk
açıklamada bu atlanıp "rayı gördüğü için" denmişti, sonra düzeltildi.)

---

## 8. Çekim bölgesi: neden sadece 2°?

Başarı tanımı Aşama 4 ile aynı (sonlu, |x| ≤ 2.4 m, sonda |θ| < 0.05, |θ̇| < 0.1). θ̇₀ = 0, F_max = 10 N.

| kontrolcü | ilk başarısız açı |
|---|---|
| LQR Bryson | 32° |
| MPC N = 10 | 33° |
| MPC N = 30 | 34° |
| fiziksel sınır | 43° |

**Teşhis, öngörü ile gerçek (34°):** her plan gerçeği **geriden** takip ediyor. MPC **iyimser**:

| | t = 0.2 s | t = 0.4 s |
|---|---|---|
| adım 0'daki plan | θ = 0.44 | θ = −0.11 |
| gerçek | θ = 0.50 | θ = +0.09 |

Her yeni planda gecikme fark edilip düzeltiliyor (planlar sağa kayıyor), sonunda sarkaç ters yöne fazla savruluyor
(−0.5 rad) ve kaybediliyor. Kısıtlar aktif değilken kaybedilen (`x_max=None`) koşularda da çözücü hep "optimal" diyor.
"infeasible" durumları ancak sarkaç düştükten **sonra** çıkıyor; sebep değil sonuç.

**Neden iyimser? B matrisi:**
$$ \text{model: } \frac{\partial\ddot\theta}{\partial F} = -\frac{1}{Ml},\qquad \text{gerçek: } -\frac{\cos\theta}{l\,(M + m\sin^2\theta)} $$
cos 34° = 0.83 → arabayı itmenin sarkaca etkisi modelin sandığından **%17 zayıf**. Yatık sarkaç yatay itişi daha az
hisseder; kuvvetin bir kısmı çubuk boyunca gider. (Yerçekimi tarafında sinθ/θ = 0.94; o da %6 fark.)

**Sonuç:** MPC kısıtları biliyor ama **yanlış fizikle** planlıyor. Büyük açıda çekim bölgesini sınırlayan artık kısıt
bilgisi değil, **modelin doğruluğu**.

| kontrolcü | biliyor | bilmiyor | 10 N ile |
|---|---|---|---|
| LQR | doğrusal model | kısıtlar | 32° |
| MPC | doğrusal model + kısıtlar | doğrusal olmayan fizik | 34° |
| ? | gerçek fizik + kısıtlar | — | ≤ 43° |

**Boşluğu doldurma yolları:**
1. **Doğrusal olmayan MPC (NMPC):** öngörü gerçek f(s, F) ile. Dışbükey değil, yavaş, yerel çözümler. Ara yol: her adımda
   o anki duruma göre yeniden doğrusallaştırma (LTV-MPC). Planda yok, isteğe bağlı.
2. **RL (Aşama 9):** doğrudan doğrusal olmayan simülatörde öğreniyor, aksiyon ±10 N sınırlı, çalışırken sadece ağ.
   Bedeli: çok sayıda deneme, garanti yok, simülatör–gerçek farkı (Aşama 12).

---

## 9. MPC, LQR ve RL karşılaştırması

| | LQR | MPC | RL |
|---|---|---|---|
| model gerekli mi? | evet (doğrusal) | evet (öngörü) | hayır (simülatör / deneyim) |
| kısıtlar | bilmez, sonradan kırpılır | **açıkça** optimizasyonda | aksiyon sınırı doğal; durum kısıtı ödülle |
| çalışma anı hesap | F = −Ks (μs) | her adımda QP (ms) | ağ ileri geçişi (μs) |
| hesap ne zaman? | çevrimdışı (Riccati) | **çevrimiçi** | **çevrimdışı** (eğitim) |
| optimallik / garanti | doğrusal + kısıtsız için optimal, kararlılık garantili | kısıtlı doğrusal problem için optimal; terminal maliyetle kararlılık | garanti yok, deneysel |
| değer fonksiyonu | V = sᵀPs (kapalı form) | terminal maliyet olarak P | öğrenilen V / Q ağı |

**Ortak çizgi:** üçünün de kalbinde aynı Bellman fikri var. LQR onu kapalı formda çözüyor, MPC sonlu ufukta çevrimiçi
çözüp ufkun ötesini LQR'nin değeriyle kapatıyor, RL ise veriden yaklaşık olarak öğreniyor.

---

## 10. Mülakat soruları ve örnek cevaplar

**S1. MPC ile LQR arasındaki fark ne, ne zaman hangisi?**
İkisi aynı maliyeti minimize ediyor. LQR sonsuz ufuklu ve kısıtsız problemi kapalı formda çözüp sabit bir K veriyor;
MPC sonlu ufukta, kısıtlarla her adımda bir QP çözüyor. Terminal maliyet olarak LQR'nin P'sini koyunca kısıtlar aktif
değilken ikisi birebir aynı kontrolcü (bunu test ettim). Kısıtlar önemliyse ve hesap bütçesi yetiyorsa MPC, yoksa LQR.

**S2. MPC'de kararlılığı nasıl sağlarsın?**
Terminal maliyet olarak LQR'nin değer fonksiyonu sᵀPs (gerekirse terminal kısıt kümesiyle). Ufkun ötesinin maliyeti
LQR'ninkiyle tahmin edildiği için kısa ufukta bile miyop davranmıyor. Ufku 5'ten 60'a çıkarmak oturmayı değiştirmedi.

**S3. Gerçek zamanda çalışır mı, nasıl ölçtün?**
Her `act`'te çözüm süresini ölçüp ortalama, %95 ve en kötüyü raporladım; ilk çağrıyı (derleme) ayırdım. dt = 20 ms'de
N ≤ 30 rahat sığıyor (en kötü 8.8 ms), N = 60 aşıyor (27 ms). Gerçek sistemde çözücü C'ye derlenir.

**S4. MPC ile RL'yi nasıl karşılaştırırsın?**
Model gereksinimi (MPC ister), kısıt yönetimi (MPC açıkça), çalışma anı hesap (MPC çevrimiçi optimizasyon, RL ağ
çıkarımı), garanti (MPC'de var, RL'de yok) ve **model uyumsuzluğu**. MPC'nin çekim bölgesi doğrusal öngörünün büyük
açıdaki iyimserliği yüzünden 34°'de kaldı; RL doğrusal olmayan simülatörde öğrendiği için bu boşlukta değerlendirdim.

**S5. MPC neden çekim bölgesini çok az genişletti?**
Planlar gerçeği sistematik olarak geriden takip etti. Doğrusal model kontrol etkinliğini cosθ = 1 varsayıyor; 34°'de
gerçek etkinlik %17 düşük. Kısıt bilgisi doğruydu ama fizik yanlıştı. Teşhisi öngörü ile gerçeği aynı grafikte çizip
yaptım.

**S6. Çözücü çözüm bulamazsa ne olur?**
Kontrolcü susamaz. Yedek plan olarak kırpılmış LQR'ye düşüyor. Bu yolu bilerek tetikleyen bir test yazdım ve ilk
denemede yedek planın işaret hatasını bu test yakaladı.

---

## 11. Tuzaklar

| tuzak | belirti | önlem |
|---|---|---|
| Problemi her adımda yeniden kurmak | çok yavaş | bir kez kur, `cvx.Parameter` güncelle |
| P ile adım maliyetinin ölçek uyuşmazlığı | MPC ≠ LQR | ikisi de Q·dt, R·dt ölçeğinde |
| Varsayılan çözücü hassasiyeti | MPC = LQR ancak 3. basamakta | `eps_abs/eps_rel` küçült, testte tolerans 1e-3 |
| Yedek planda +K·s | sadece zor anda ters itme | yedek yolu bilerek tetikleyen test |
| Ortalama süreye bakmak | N = 60 "sığıyor" sanıldı | en kötü ve %95 |
| İlk çağrıyı ölçüme katmak | yanıltıcı en kötü süre | ısınmayı ayrı raporla |
| Ablasyonsuz sebep iddiası | "rayı gördüğü için" | özelliği kapatıp tekrar dene |
| "infeasible"ı sebep sanmak | sarkaç düştükten sonra çıkıyor | ilk başarısız adımın zamanına bak |
| `basarili` fonksiyonunu defterden deftere kopyalamak | tanım sessizce ayrışır | `cartpole.py` / `evaluate.py`'ye taşı |

---

## 12. Kaynak haritası

| konu | kaynak |
|---|---|
| MPC fikri, kayan ufuk | Brunton: "Model Predictive Control" |
| Doğrusal MPC'yi QP olarak kurmak | cvxpy dokümanı → örnekler → "Control" |
| Terminal maliyet ve kararlılık | Borrelli, Bemporad, Morari, *Predictive Control for Linear and Hybrid Systems* (ücretsiz PDF), doğrusal MPC ve kararlılık bölümleri |
| Gömülü / hızlı MPC | OSQP kod üretimi, `acados` dokümanları |

---

## 13. Kendini sına

1. Terminal maliyeti P yerine 0 alsaydın, N = 5'te MPC = LQR testi tutar mıydı? Neden?
2. Kısıtlar aktif değilken MPC ile LQR neden aynı? Bir cümlede Bellman ile açıkla.
3. N'i 30'dan 60'a çıkarmak bu senaryoda neyi değiştirdi, neyi değiştirmedi? Neden?
4. Gerçek zaman bütçesini değerlendirirken neden ortalamaya değil en kötü süreye bakıyoruz?
5. Kuvvet sınırı senaryosunda MPC'yi kurtaranın ray bilgisi olmadığını nasıl kanıtladık?
6. 34°'de MPC'nin planları neden gerçeğin önünde gidiyor? Hangi trigonometrik fonksiyon?
7. Yedek plan neden −K·s ve neden bir test bunu ayrıca kontrol etmeli?
8. NMPC boşluğu doldurabilir mi? Bedeli ne?
9. LQR, MPC ve RL'yi "çevrimiçi / çevrimdışı hesap" açısından karşılaştır.
10. Ray kısıtının **gerçekten** işe yaradığı bir senaryo tasarla. Başlangıç durumu ne olmalı?
