# Aşama 4 Ders Notları: LQR, Değer Fonksiyonu ve Çekim Bölgesi

> Her bölüm: **Amaç → Sezgi → Matematik → Kodda nerede → Mülakatta ne derim.**
> Kod: `controllers.py` (`lqr_continuous`, `lqr_discrete`, `LQRController`), testler: `test_all.py` (Aşama 4),
> deneyler: `notebooks/04_lqr.ipynb`. Parametreler: M = 1, m = 0.1, l = 0.5, g = 9.81, F_max = 10 N, dt = 0.02 s.
> Ders notu bağlantısı: LQR = hafta 7–8'deki kutup yerleştirmenin "optimal" versiyonu (`DERS_PROJE_UYUMLULUGU.md`).

---

## İçindekiler
1. [LQR fikri: kazancı değil cezayı seç](#1-lqr-fikri-kazancı-değil-cezayı-seç)
2. [Bryson kuralı ve ilk tasarım](#2-bryson-kuralı-ve-ilk-tasarım)
3. [LQR ile kaskad PID karşılaştırması](#3-lqr-ile-kaskad-pid-karşılaştırması)
4. [Riccati = değer iterasyonu (RL bağlantısı)](#4-riccati--değer-iterasyonu-rl-bağlantısı)
5. [Sürekli ve ayrık LQR](#5-sürekli-ve-ayrık-lqr)
6. [Q / R takası](#6-q--r-takası)
7. [Çekim bölgesi](#7-çekim-bölgesi)
8. [Mülakat soruları ve örnek cevaplar](#8-mülakat-soruları-ve-örnek-cevaplar)
9. [Tuzaklar](#9-tuzaklar)
10. [Kaynak haritası](#10-kaynak-haritası)
11. [Kendini sına](#11-kendini-sına)

---

## 1. LQR fikri: kazancı değil cezayı seç

**Amaç:** Aşama 3'te iki döngü kurup dört kazancı elle ayarladık; yavaş iç döngü kaskadı bozdu, işaretler kafa
karıştırdı. LQR'de kazançları seçmiyorsun: **neyi ne kadar önemsediğini** söylüyorsun, matematik en iyi K'yı buluyor.

$$ J = \int_0^\infty \big( s^\top Q s + F^\top R F \big)\, dt, \qquad Q = \mathrm{diag}(q_x, q_{\dot x}, q_\theta, q_{\dot\theta}) $$

- **Q:** her durumun hatasına ne kadar ceza verdiğin. **R:** kuvvete ne kadar ceza verdiğin (enerji, ısınma, doyma).
- **Kare** almanın iki sebebi var: işaret önemsiz olur, büyük hataya orantısız ceza verilir.
- **Sadece oranlar önemli:** Q ve R'yi aynı sayıyla çarparsan K değişmez.

**Sonuç her zaman doğrusal durum geri beslemesi:** F = −K·s. Aşama 3'teki A − BK dersi aynen geçerli.

$$ A^\top P + PA - PBR^{-1}B^\top P + Q = 0, \qquad K = R^{-1}B^\top P $$

**Garantiler** (mülakatta mutlaka):
1. (A, B) kontrol edilebilir ise kapalı döngü **garantili kararlı** (Aşama 2'deki rank 4'ün ödülü).
2. Sürekli zamanlı, tek girdili LQR'de **kazanç payı sonsuz, faz payı en az 60°**.
3. **Ama:** garantiler doğrusal model + kısıtsız girdi + tam durum bilgisi için geçerli. Üçü de gerçekte bozuluyor:
   büyük açı (Bölüm 7), 10 N sınırı (Bölüm 7), gürültülü ölçüm (Aşama 5).

**Kodda:**
```python
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.solve(R, B.T @ P)        # R⁻¹BᵀP: tersini alma, denklemi çöz (daha doğru, daha hızlı)
```
`LQRController.act(s) = −K(s − s_ref)`. `reset()` boş (`pass`), çünkü LQR'nin hafızası yok.

---

## 2. Bryson kuralı ve ilk tasarım

$$ Q_{ii} = \frac{1}{(\text{izin verilen en büyük } s_i)^2}, \qquad R = \frac{1}{F_{\max}^2} $$

Fikir: her terimi "kabul edilebilir büyüklüğüne" göre **birimsizleştirmek**. 0.5 m konum hatası ile 0.2 rad açı
hatası aynı derecede "kötü" sayılır.

| durum | izin | Q_ii |
|---|---|---|
| x | 0.5 m | 4 |
| ẋ | 1 m/s | 1 |
| θ | 0.2 rad | 25 |
| θ̇ | 1 rad/s | 1 |
| F | 10 N | R = 0.01 |

Sonuç (sürekli): **K = [−20, −23.7, −124.2, −27.3]** → F = **+20x + 23.7ẋ** + 124θ + 27.3θ̇.

**x ve ẋ katsayıları pozitif**, yani "sezgiye ters". Aşama 3'te K_x deneyi ve kaskad yapıyla zor bulduğumuz işareti
Riccati **kendiliğinden** buldu: RHP sıfırının bilgisi A ve B'nin içinde.

**Kapalı döngü özdeğerleri:** −22.69, −4.98, −1.63 ± 0.90i. Hafta 7'nin sorusu "kutuplar nereye gitti?": LQR seçti.

---

## 3. LQR ile kaskad PID karşılaştırması

Senaryo: x₀ = 0.5, θ₀ = 0.1, F_max = 10 N.

| | Kaskad (elle) | LQR (Riccati) |
|---|---|---|
| oturma (±2 cm) | 3.42 s | **2.50 s** |
| ∫F² dt | 15.18 | **9.17** |
| en büyük \|F\| | 6 N | 10 N (çok kısa) |
| baskın kutup | −1.14 ± **6.78i** (ζ = 0.17) | −1.63 ± 0.90i (ζ = 0.88) |

**LQR hem daha hızlı hem daha tutumlu.** Sebebi:
- **Kaskadın kuvveti salınıyor** (+6, −5.5, +3.7 …). Birbirini iptal eden itişler boşa gidiyor, ama ∫F² her birini sayıyor.
- **LQR'nin kuvveti kısa, sert bir itiş ve sonra sakin bir dönüş.** Kuvveti en etkili anda harcıyor.

**Kural: oturma ≈ 4 / |en yavaş kutbun gerçel kısmı|.** Kaskad için 4/1.14 = 3.5 s (ölçülen 3.42), LQR için 4/1.63 = 2.45 s
(ölçülen 2.50). Grafiği çizmeden tahmin edilebiliyor.

**Dürüst not:** LQR'nin hedefinde R·F² zaten var, yani ∫F²'yi düşürmek için tasarlanmış. Kaskad ise optimize edilmedi.
Asıl mesaj: **optimizasyon problemini doğru kurarsan, elle ayara göre çok iyi bir başlangıç bedavaya gelir.**

**"LQR'nin ayarlama süreci var" yanılgısı:** K bir kez, önceden hesaplanır (milisaniye). Çalışırken sadece F = −Ks
çarpımı yapılır; yükü PD'den bile hafiftir. "Ayarlama" ise elle PID'de mühendisin harcadığı zamandır.

---

## 4. Riccati = değer iterasyonu (RL bağlantısı)

### 4.1 Değer fonksiyonu
Ayrık zamanda toplam ceza J = Σ (s_kᵀQ_d s_k + u_kᵀR_d u_k). **Değer fonksiyonu** V(s): s'den başlayıp **en iyi** kontrolle
ödenecek toplam ceza. LQR'nin kapalı form cevabı:

$$ V(s) = s^\top P\, s $$

RL'deki V(s) / Q(s, a)'nın doğrusal-karesel durumdaki **kapalı formu**.

### 4.2 Bellman ve Riccati
$$ V_{\text{yeni}}(s) = \min_u \big[\, s^\top Q_d s + u^\top R_d u + V_{\text{eski}}(A_d s + B_d u) \,\big] $$

V karesel olduğu için minimum kapalı formda bulunuyor ve matris güncellemesine dönüşüyor (Ek A.3):

$$ K = (R_d + B_d^\top P B_d)^{-1} B_d^\top P A_d, \qquad P \leftarrow Q_d + A_d^\top P A_d - A_d^\top P B_d K $$

P₀ = Q_d'den başlayıp yinelemek = **değer iterasyonu**. Her tur kontrolcünün bir adım daha ileriye bakması demek.

### 4.3 Doğrulamalar (defter 04 + `test_all.py`)
| kontrol | sonuç |
|---|---|
| elle yineleme = `solve_discrete_are` = `control.dlqr` | fark ~1e-9 |
| yakınsama | **386 iterasyon**. En yavaş kapalı döngü modu her adımda sadece %3.2 sönüyor (\|z\| = 0.968); 0.968^(2k) ≈ 1e-10 → k ≈ 350 |
| P pozitif tanımlı | özdeğerler [0.056, 0.603, 3.09, 23.5]. V bir ceza toplamı: negatif olamaz, s ≠ 0 için sıfır olamaz |
| **değer fonksiyonu** | 5000 adımda gerçekten ödenen ceza = s₀ᵀPs₀ = **0.6446394190** (fark ~1e-13) |

**P'nin özdeğerleri:** en büyüğünün özvektörü dengeden "en pahalı" sapma yönü (sarkaç açısı ağırlıklı), en küçüğü en ucuz
yön. **V(s) her adımda azalıyor** → V bir **Lyapunov fonksiyonu**: kapalı döngü kararlılığının kanıtı.

### 4.4 Miyop kontrolcü ve RL'deki γ
Değer iterasyonunun k. turundaki K, "k adım ileriye bakan" kontrolcü.

| ileriye bakış | K'da x, ẋ | kuvvette | davranış |
|---|---|---|---|
| 1–20 adım (≤ 0.4 s) | +, + | −x, −ẋ | "araba sağda → sola it" (açgözlü, sezgisel) |
| 30+ adım (≥ 0.6 s) | −, − | +x, +ẋ | ters tepki stratejisi |

Kısa ufuklu kontrolcü geri itmenin 0.5 s sonra sarkacı devireceğini **göremiyor**. Ufuk ters tepki süresini aşınca strateji
değişiyor. **RL karşılığı:** indirim faktörü γ, etkin ufuk ≈ 1/(1 − γ) adım. γ = 0.9 (~10 adım) bir ajan tam bu miyop
kontrolcü gibi davranır. **Aşama 9 için kural:** γ, ters tepki süresini (~25 adım) rahatça kapsamalı → γ ≥ 0.98.

---

## 5. Sürekli ve ayrık LQR

| | x | ẋ | θ | θ̇ |
|---|---|---|---|---|
| sürekli K | −20.0 | −23.7 | −124.2 | −27.3 |
| ayrık K_d (dt = 0.02) | −14.7 | −17.6 | −96.2 | −21.2 |

Ayrık kazançlar **%22–26 daha yumuşak**. Sürekli LQR kuvvetin her an değişebildiğini varsayıyor ve −22.7'lik bir mod
tasarlamış: zaman sabiti 0.044 s ≈ iki örnekleme adımı. Gerçekte kuvvet 0.02 s sabit tutuluyor (ZOH). Ayrık LQR bunu
**biliyor** (A_d, B_d) ve dijital kontrolcü için gerçekten optimal olan kazançları seçiyor.

**Kural:** En hızlı kapalı döngü modunda λ·dt ≪ 1 ise ikisi neredeyse aynı. Bizde 22.7 × 0.02 = 0.45, küçük değil.
**Bilgisayarda çalışacak kontrolcüyü ayrık zamanda tasarla.** dt → 0 iken K_d → K.

**Ayrık maliyet ölçeklemesi:** Q_d = Q·dt, R_d = R·dt. Sürekli ∫(…)dt'nin ayrık toplamı.

---

## 6. Q / R takası

Ayrık LQR, senaryo x₀ = 0.5, θ₀ = 0.1, F_max = 10.

| ayar | K | oturma | ∫F² | yavaş kutup | yorum |
|---|---|---|---|---|---|
| Bryson | [−14.7, −17.6, −96.1, −21.1] | 2.50 s | 8.00 | −1.63 | başlangıç noktası |
| **q_θ × 10** | [−14.0, −21.4, −154, −24.9] | **3.30 s** | 8.87 | **−0.86** | **yavaşladı** |
| q_x × 10 | [−45.5, −35.1, −138, −30.8] | **1.40 s** | **25.78** | −2.84 | hızlı ama 3.2× enerji, θ −0.21'e kadar |
| **R × 10** | [−5.4, −7.0, −49.4, −10.8] | 2.66 s | **3.96** | −1.49 | yarı enerji, %6 yavaş, **hiç doymuyor** |
| R ÷ 10 | [−30.1, −35.1, −175, −38.4] | 2.48 s | 10.38 | −1.65 | 2× kazanç, **hız yok** |

**Üç ders:**
1. **q_θ'yu artırmak yavaşlatır:** sarkacın yatıklığı arabanın gaz pedalı (ẍ ≈ gθ). θ'ya ağır ceza = "gaz pedalına hafif bas".
   **Q'nun elemanları fizik üzerinden birbirine bağlı**; "sarkacı dik tut" ile "arabayı hızlı götür" çelişiyor.
2. **Kuvveti ucuzlatmak (R ↓) hız satın almıyor:** (i) doyma zaten kırpıyor, (ii) oturmayı belirleyen yavaş kutup kuvvetle
   değil **fizikle** sınırlı: RHP sıfırı (z = 4.43) ile kararsız kutup (p = 4.65) yakın. Aşama 3'teki kaskadın ~−1.3
   duvarıyla aynı duvar. **R × 10'un ucuz olmasının sebebi de bu: kısılan kuvvet zaten işe yaramayan kuvvetti.**
3. **"En iyi" ayar yok, gereksinime göre seçilir.** "Oturma ≤ 3 s, en az enerji" → R × 10. "Oturma ≤ 1.5 s" → q_x × 10
   ve daha büyük bir motor (sistem ekibine götürülecek soru).

**Bryson R = 1/F_max² motor sınırını garanti etmiyor.** Beş ayarın dördü başta 10 N'a dayandı. LQR sınırı bilmiyor,
sadece kuvvete "pahalı" diyebiliyor (→ Aşama 4b, MPC).

**Ölçü seçimi de tasarım kararı:** `max|θ|` her satırda 0.100 çıktı (başlangıç açısı). Fark ters yöndeki yatıştaydı.
"Başlangıçtan sonraki en büyük |θ|" daha anlamlı bir ölçü olurdu.

---

## 7. Çekim bölgesi

### 7.1 Tanım ve neden yerel?
**Çekim bölgesi (ROA):** kontrolcünün kurtarabildiği başlangıç durumlarının kümesi. LQR'nin kararlılık garantisi
**yerel**: doğrusal model (sinθ ≈ θ) ve sınırsız kuvvet varsayımları büyük açıda bozuluyor. Kontrolcüden bağımsız bir
fiziksel sınır da var: 10 N ile durağan bir sarkacı tutabileceğin en büyük açı **43°** (Aşama 3).

### 7.2 Başarı tanımı: taramadan ÖNCE sabitle
| şart | neden |
|---|---|
| sayılar sonlu | sınırsız kuvvette büyük açılar simülasyonu patlatıyor |
| bütün bölüm boyunca \|x\| ≤ 2.4 m | ray sınırı |
| 10 s sonunda \|θ\| < 0.05, \|θ̇\| < 0.1 | gerçekten dik ve sakin |

**Bu tanım Aşama 9'da RL için de aynen kullanılacak.** İlk ölçümde (sadece "sarkaç dik mi?") AnglePD 42°'yle **en iyi**
göründü; ray şartı eklenince **5°'yle en kötü** oldu. Tanım sıralamayı tamamen değiştiriyor.

### 7.3 1B tarama (θ̇₀ = 0, ilk başarısız açı)
| kontrolcü | 10 N | sınırsız |
|---|---|---|
| AnglePD (20, 3) | 5° (ray) | 5° (ray) |
| Kaskad | **36°** | 50° |
| LQR Bryson | **32°** | **61°** |
| LQR R × 10 | **36°** | 57° |
| fiziksel sınır | 43° | — |

- **Sınırsızken** sınırı doğrusal olmayan fizik belirliyor. En agresif (Bryson) en uzağa gidiyor ve 43°'yi geçiyorlar,
  çünkü sınırsız kuvvette o sınır yok.
- **10 N ile "optimal" LQR en kötü.** 32°'de kontrolcü ~60 N istiyor, 10 N alıyor. Kontrolcü bambaşka bir sisteme göre
  hesap yaptığını sanıyor; ters tepki stratejisi de kısıtlı kuvvet bütçesinden pay alıyor. Optimallik garantisi
  kısıtsız girdi varsayımına dayanıyor.
- **Yumuşak (R × 10) daha iyi**: daha az ve daha kısa doyuyor, tasarlandığı gibi davranmaya daha yakın kalıyor.
- **Kaskad, R × 10 kadar iyi**: küçük kazançlar ve θ_ref'in ±0.2'de kırpılması bir tür "kısıt bilinci".

### 7.4 2B harita (θ₀ × θ̇₀, F_max = 10)
| kontrolcü | kurtarılan alan |
|---|---|
| Kaskad | %54 |
| LQR R × 10 | %53 |
| LQR Bryson | %48 |

- **Şekil: çapraz şerit.** θ₀ ile θ̇₀'ın işaretleri **zıtsa** sarkaç dik konuma doğru geliyor: 60° bile kurtarılabiliyor.
  İşaretler **aynıysa** sarkaç zaten devriliyor: θ₀ = 0'da bile 3–4 rad/s'lik hız kurtarılamaz.
- **Şeridin merkez hattı θ̇ = −λθ** (λ = 4.65): kararlı manifold, yani Aşama 2'deki sönen modun (e^{−λt}) var olduğu
  başlangıçlar. Bu çizgideki sarkaç kontrolsüz bile (doğrusal modelde) dik konuma gelir. Şeridin genişliği, kontrolcünün
  büyüyen modu (e^{+λt}) kuvvet sınırı içinde ne kadar yenebildiği. Hafta 5–6'daki modal ayrıştırmanın uygulaması.
- Sıralama 1B ile aynı.

### 7.5 Projenin tezi
```
0°                        36°        43°                       61°
|——— klasik kontrolcüler ———|  BOŞLUK |                         |
     (10 N ile)                 ≤ 7°   fiziksel sınır          doğrusallık sınırı (sınırsız kuvvet)
```
Boşluğu kuvvet sınırını **bilerek** plan yapan bir kontrolcü alabilir: **MPC** (Aşama 4b) ve **RL** (Aşama 9). Dürüst not:
43° yerinde tutma sınırı. Tutmak için araba sürekli g·tanθ ≈ 8 m/s² ivmelenmeli, 2.4 m'lik ray ise hızla bitiyor.
Gerçek ulaşılabilir sınır muhtemelen 43°'nin altında.

### 7.6 Kod notları
- `basarili(S)`: tanımın kodu. NaN kontrolü ilk sırada.
- Her koşuda `lambda: Kontrolcu(...)` ile taze nesne → hafıza paylaşılmıyor.
- `substeps=5`: 728 simülasyon ~30 s. **Hızlandırmadan önce sonucun değişmediğini doğruladık** (sınır açıları 5 ve 20
  alt adımda aynı).
- `np.errstate(all="ignore")`: patlayan simülasyonların beklenen uyarılarını susturuyor.

---

## 8. Mülakat soruları ve örnek cevaplar

**S1. Q ve R'yi nasıl seçtin, değiştirince ne oldu?**
Bryson kuralıyla başladım: her durumu kabul edilebilir en büyük değerine göre birimsizleştirdim. Sonra takas tablosu
çıkardım. Açı ağırlığını artırmak sistemi yavaşlattı, çünkü sarkacın yatıklığı arabanın gaz pedalı. R'yi 10 katına
çıkarmak enerjiyi yarıya indirdi, hızdan sadece %6 kaybettirdi. R'yi azaltmak hız kazandırmadı: yavaş kutup kuvvetle
değil, RHP sıfırı ile kararsız kutbun yakınlığıyla sınırlı.

**S2. LQR hangi varsayımlarla optimal, gerçekte hangileri bozuluyor?**
Doğrusal model, kısıtsız girdi, tam ve gürültüsüz durum bilgisi. 10 N sınırıyla LQR'nin çekim bölgesi 61°'den 32°'ye
düştü; daha yumuşak bir LQR ve kaskad 36°'de kaldı. Ölçüm gürültüsü Aşama 5'in (LQG) konusu.

**S3. LQR ile RL arasındaki ilişki ne?**
Ayrık Riccati yinelemesi, doğrusal-karesel durumda değer iterasyonu; P değer fonksiyonu, V(s) = sᵀPs. Simülasyonla
ödenen toplam cezanın s₀ᵀPs₀'a eşit olduğunu doğruladım. Kısa ufuklu iterasyonlarda kontrolcü arabayı açgözlüce geri
itiyor, ufuk ters tepki süresini aşınca minimum fazlı olmayan stratejiyi buluyor. Bu RL'de γ seçimine doğrudan çeviriliyor.

**S4. Sürekli ve ayrık LQR farkı? Hangisini kullandın?**
Ayrık. dt = 0.02'de sürekli LQR'nin en hızlı modu λ·dt = 0.45; ZOH'yi hesaba katan ayrık LQR kazançları %25 daha
yumuşak buldu. Dijital uygulama için doğru olan bu.

**S5. P neden pozitif tanımlı?**
sᵀPs, s'den başlayınca ödenecek karesel cezaların toplamı: negatif olamaz, sıfırdan farklı bir sapma için sıfır olamaz.
Ayrıca her adımda azaldığı için bir Lyapunov fonksiyonu.

**S6. Çekim bölgesini neler sınırlıyor?**
Üç şey: doğrusal modelin geçerliliği (sınırsız kuvvette 61°), kuvvet sınırı (10 N ile 32–36°) ve fiziksel tutma sınırı
(43°). Ray sınırı da bir dördüncü kısıt. 2B haritada bölge, kararlı manifold θ̇ = −λθ etrafında bir şerit.

**S7. LQR'yi PID ile nasıl karşılaştırdın?**
Aynı senaryo, aynı kuvvet sınırı ve aynı başarı tanımıyla. LQR %27 daha hızlı oturdu ve %40 daha az enerji harcadı;
kaskadın zayıf sönümlü modu (ζ = 0.17) kuvveti salınımla boşa harcıyordu. Ama 10 N sınırlı çekim bölgesinde kaskad
(%54) Bryson LQR'den (%48) iyiydi. Hangi ölçüte baktığın sonucu değiştiriyor.

**S8. LQR'nin K'sı neden kaskadınkine benziyor?**
İkisi de x ve ẋ için pozitif kuvvet katsayısı veriyor (sezgiye ters işaret). İkisi de aynı fiziği (RHP sıfırı, ters
tepki) yansıtıyor. LQR bunu Riccati'den, kaskad gaz pedalı benzetmesinden buluyor.

---

## 9. Tuzaklar

| tuzak | belirti | önlem |
|---|---|---|
| Parametreleri `P` adlı Riccati matrisiyle ezmek | simülasyon bozuluyor | Riccati çözümüne `Pmat` / `P_it` de |
| `float(u)`, u tek elemanlı dizi | numpy uyarısı | `float(u[0])` |
| R'yi skaler vermek | `solve_*_are` hata | `np.array([[r]])` |
| Ayrık LQR'de Q, R'yi dt ile ölçeklememek | sürekliyle kıyaslanamayan K | Q_d = Q·dt, R_d = R·dt |
| Değer fonksiyonunu doğrusal olmayan simülasyonla test etmek | fark doğrusallaştırmadan geliyor | sadece test ettiğin şeyi test et: doğrusal model |
| "Optimal = her durumda en iyi" | doymada LQR en kötü çıktı | optimallik varsayımlarına bak |
| Başarı tanımını taramadan sonra seçmek | AnglePD 42° → 5° | tanımı önce yaz, KARARLAR.md |
| Doğrulamadan hızlandırmak (`substeps`) | sınırlar kayabilir | 5 ve 20 alt adımda sınır aynı mı kontrol et |
| `max|θ|` ölçüsü | her satır 0.100 | ölçüyü fizikle seç |

---

## 10. Kaynak haritası

| konu | kaynak |
|---|---|
| Kutup yerleştirme → LQR geçişi | Ders notları hafta 7–8; Brunton: "Pole Placement for the Inverted Pendulum on a Cart" |
| LQR, Riccati, K = R⁻¹BᵀP | Brunton: "LQR Control for an Inverted Pendulum on a Cart"; Åström & Murray, "State Feedback" → "Linear Quadratic Regulators" |
| Dinamik programlama, değer fonksiyonu, RL bağlantısı | Tedrake, "Linear Quadratic Regulators" bölümü |
| Ayrık LQR | Tedrake LQR (ayrık kısım); `scipy.linalg.solve_discrete_are` dokümanı |
| Çekim bölgesi, Lyapunov | Tedrake, "Lyapunov Analysis" bölümünün başı |

---

## 11. Kendini sına

1. Q ve R'nin ikisini de 100 ile çarparsan K ne olur? Neden?
2. q_x × 10 neden θ'yı −0.21'e kadar yatırdı? Gaz pedalı benzetmesiyle açıkla.
3. Değer iterasyonu neden 386 adımda yakınsadı da 30'da değil? Hangi özdeğer belirliyor?
4. γ = 0.95 seçilmiş bir RL ajanının etkin ufku kaç saniye? Cartpole'da ters tepkiyi "görür" mü?
5. dt = 0.005 olsaydı sürekli ve ayrık LQR arasındaki fark büyür mü küçülür mü?
6. 10 N sınırıyla neden yumuşak LQR (R × 10) agresif olandan (Bryson) daha geniş bir çekim bölgesine sahip?
7. 2B haritada θ₀ = 0, θ̇₀ = +3.5 neden kırmızı? θ₀ = 40°, θ̇₀ = −3 neden yeşil?
8. P'nin en büyük özdeğerinin özvektörü neden sarkaç açısı ağırlıklı?
9. "LQR kaskaddan iyi" cümlesi hangi ölçütte doğru, hangisinde yanlış?
10. MPC çekim bölgesini genişletebilir mi? Hangi varsayımı düzelttiği, hangisini düzeltmediği için?
