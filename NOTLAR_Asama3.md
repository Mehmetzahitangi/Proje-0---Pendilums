# Aşama 3 Ders Notları: PID, Kapalı Döngü Analizi ve Kaskad Kontrol

> Her bölüm: **Amaç → Sezgi → Matematik → Kodda nerede → Mülakatta ne derim.**
> Kod: `controllers.py` (PID, AnglePD, CascadePID, run_closed_loop), testler: `test_all.py` (Aşama 3 bölümleri),
> deneyler: `notebooks/03_pid_analysis.ipynb`. Parametreler: M = 1, m = 0.1, l = 0.5, g = 9.81, F_max = 10 N, dt = 0.02 s.
> VS Code'da `Ctrl+Shift+V` ile önizleme.

---

## İçindekiler
1. [Açık döngü, kapalı döngü ve işaret](#1-açık-döngü-kapalı-döngü-ve-işaret)
2. [Sadece açıya PD: kütle–yay–amortisör](#2-sadece-açıya-pd-kütleyayamortisör)
3. [A − BK: kapalı döngüyü önceden hesaplamak](#3-a--bk-kapalı-döngüyü-önceden-hesaplamak)
4. [Kararlılık haritası ve "daha çok kazanç her zaman iyi mi?"](#4-kararlılık-haritası-ve-daha-çok-kazanç-her-zaman-iyi-mi)
5. [Ayrık PID: türev tekmesi, türev filtresi, anti-windup](#5-ayrık-pid-türev-tekmesi-türev-filtresi-anti-windup)
6. [Doyma ve fiziksel sınır (43°)](#6-doyma-ve-fiziksel-sınır-43)
7. [Kaskad PID](#7-kaskad-pid)
8. [Yazılım tasarımı ve test stratejisi](#8-yazılım-tasarımı-ve-test-stratejisi)
9. [Mülakat soruları ve örnek cevaplar](#9-mülakat-soruları-ve-örnek-cevaplar)
10. [Tuzaklar (bu aşamada gerçekten yaşananlar)](#10-tuzaklar-bu-aşamada-gerçekten-yaşananlar)
11. [Kaynak haritası](#11-kaynak-haritası)
12. [Kendini sına](#12-kendini-sına)

---

## 1. Açık döngü, kapalı döngü ve işaret

- **Açık döngü:** kuvvet önceden belli (F = 0, F = sabit). Sistem ne yaparsa yapar. Aşama 1–2'de sarkaç düştü.
- **Kapalı döngü:** her dt'de **ölç → hesapla → uygula**. Kuvvet durumun fonksiyonu: F = k(s).

**İşaret (süpürge kuralı):** sarkaç sağa yatarsa (θ > 0) arabayı sağa it (F > 0), altına gir.
Aşama 2'deki B ile tutarlılık: θ̈ ≈ 21.58·θ − 2·F. F > 0, θ̈'ya **negatif** katkı yapar ve sarkacı dik konuma döndürür.

**"Ters etkili süreç":** F'den θ'ya kazanç negatif (−2). Standart PID kuralı e = ref − ölçüm ve pozitif kazançlarla
yanlış yöne iter. İki çözüm var: kazançları negatif vermek ya da hatayı ters tanımlamak. Bizim seçimimiz: `AnglePD`
içinde `error = θ − θ_ref` ve bir yorum satırı. İşaret inceliği sınıfın içinde kalıyor, kullanıcı pozitif kazanç veriyor.

---

## 2. Sadece açıya PD: kütle–yay–amortisör

$$ F = K_p\,\theta + K_d\,\dot\theta $$

θ̈ = aθ − bF (a = (M+m)g/(Ml) = 21.58, b = 1/(Ml) = 2) denkleminde yerine koy:

$$ \ddot\theta + \underbrace{b K_d}_{\text{amortisör}}\,\dot\theta + \underbrace{(b K_p - a)}_{\text{yay}}\,\theta = 0 $$

| terim | anlam |
|---|---|
| bK_d·θ̇ | hıza karşı fren (amortisör) |
| (bK_p − a)·θ | **iki kuvvetin yarışı**: senin itişin bK_p dik konuma çekiyor, yerçekimi a devirmeye çalışıyor ("negatif yay") |

**İkinci dereceden sistemde kararlılık = bütün katsayılar pozitif** (Routh gerekmez):
- **K_p > a/b = (M+m)g ≈ 10.8 N/rad**: senin yayın yerçekiminin negatif yayından güçlü olmalı.
- **K_d > 0**: sönüm olmalı.

**Ayar dili:** ω_n² = bK_p − a, 2ζω_n = bK_d. Örnek: K_p = 20, K_d = 3 → ω_n = 4.29 rad/s, ζ = 0.70.

### Üç deney (θ₀ = 0.1)
| | K_p | K_d | yay / fren | sonuç |
|---|---|---|---|---|
| A | 5 | 1 | −11.6 / 2 | **düştü** (~1.2 s). Yay negatif |
| B | 20 | 0 | +18.4 / 0 | **salındı**, periyot 2π/4.29 = 1.46 s. **Ama genlik büyüdü** (5 s'de ±0.24, 10 s'de ±0.77) |
| C | 20 | 3 | +18.4 / 6 | θ ~1 s'de oturdu, **araba 0.30 m/s ile kayıp gitti** |

**B'deki büyüme neden?** Teoride σ = 0, yani sınırda. Ama kontrolcü kuvveti 0.02 s **sabit tutuyor** (ZOH). Bu,
ortalama ~dt/2'lik bir gecikme demek. Sönümsüz salınımda gecikme enerjiyi besler (salıncağı yanlış anda itmek).
Kontrolü 0.001 s'de güncelleyince etki kayboluyor (10 s'de ±0.11). **Ders:** kâğıtta sınırda kararlı olan sistem
gerçekte kararsızdır. Her zaman pay bırak.

**C'de araba neden kaydı?** Denklemde x yok: kontrolcü arabaya bakmıyor. Sarkacı yakalarken araba hız kazandı. Sarkaç
dikleşince F = 0 oldu, sürtünme de yok. Aşama 2'deki **çift integratör**, yani iki sıfır özdeğer.

---

## 3. A − BK: kapalı döngüyü önceden hesaplamak

Her doğrusal durum geri beslemesi F = −K·s biçiminde yazılır. Yerine koy:

$$ \dot s = A s + B(-Ks) = (A - BK)\,s $$

Kontrolcü sistemi **yeni bir A**'ya dönüştürür. Aşama 2'nin bütün araçları (özdeğer, kararlılık) aynen geçerli.

**Kazançları K'ya çevirmek** (s = [x, ẋ, θ, θ̇] sırasıyla):

$$ F = K_x x + K_p\theta + K_d\dot\theta \;\Longleftrightarrow\; K = [\,-K_x,\; 0,\; -K_p,\; -K_d\,] $$

**Kural seç ve karıştırma:** F = −Ks ile A − BK *ya da* F = +Ks ile A + BK. İkisi aynı matrisi verir. Eksili K'yı
A + BK ile kullanırsan **ters sistemi** analiz edersin ve kod hata vermez. Projenin kuralı: **F = −Ks, A − BK**
(LQR'nin dili).

**Boyut kuralı:** (a×b)@(b×c) = (a×c). K **1×4** olmalı. Kontrolcü bir duruma bakmıyorsa o yerde **0** durur;
eleman atlanamaz, sıra durumla aynı olmalı.

### Özdeğer okuma rehberi (λ = σ ± iω)
| özdeğer | davranış |
|---|---|
| σ < 0, ω = 0 | salınmadan söner; hız ≈ 1/\|σ\| |
| σ < 0, ω ≠ 0 | salınarak söner; periyot 2π/ω |
| σ = 0, ω ≠ 0 | sönmeyen salınım (pratikte kararsız) |
| σ > 0, ω = 0 | salınmadan kaçar; ikiye katlanma ln2/σ |
| σ > 0, ω ≠ 0 | büyüyen salınım; genlik her 1/σ'da e katı |
| 0 (tekrarlı) | serbest kayma (çift integratör) |

**Asimptotik kararlılık: BÜTÜN özdeğerlerde σ < 0.** Tek bir sıfır bile bunu bozar.

### Sonuçlarımız
| durum | özdeğerler | yorum |
|---|---|---|
| (K_p, K_d, K_x) = (5, 1, 0) | 0, 0, **+2.55**, −4.55 | kararsız; açı ~0.27 s'de ikiye katlanır |
| (20, 0, 0) | 0, 0, **±4.29i** | sınırda; ZOH ile pratikte kararsız |
| (20, 3, 0) | **0, 0**, −3 ± 3.07i | sarkaç kararlı, **sistem değil** (araba kayar) |
| (20, 3, −1) "geri it" | **+0.878**, −1.2, −2.84 ± 3.25i | kararsız; sarkaç yavaşça devriliyor (~0.8 s'de ikiye katlanma) |
| (20, 3, +1) | **+0.179 ± 1.01i**, −3.18 ± 2.92i | büyüyen yavaş salınım, **periyot 6.2 s** |

**K_x'in işareti neden "ters"?** Arabayı sola götürmek için önce sağa itip sarkacı sola yatırmak gerekiyor (RHP sıfırı,
ters tepki). K_x = −1 ("geri it") kontrolcüyü kendi kendine karşı çalıştırıyor. Sadece x'e bakmak da yetmiyor: ẋ'ye
bakılmazsa arabanın tarafında amortisör olmuyor (+1'deki büyüyen salınım).

**Simülasyon süresi en yavaş moda göre seçilir.** (20, 3, +1) 5 s'de "dengede" gibi göründü, çünkü periyot 6.2 s ve
genlik ancak her 1/0.179 = 5.6 s'de e katına çıkıyor. Kural: en az 2–3 periyot ya da 3–5 × 1/|σ|.

---

## 4. Kararlılık haritası ve "daha çok kazanç her zaman iyi mi?"

K_p × K_d ızgarasında sarkaç alt sisteminin (A'nın sağ alt 2×2'si) en büyük özdeğer gerçel kısmı. Dört boyutlu
A − BK'da iki sıfır özdeğer her yeri "kararsız" gösterirdi.

- Mavi bölge = K_p > 10.8 **ve** K_d > 0. Formülle birebir aynı ✓
- B noktası tam beyaz sınırda (σ = 0).

**"En kötü mod" kuralı:** sistem en yavaş, en kötü özdeğeri kadar iyidir → `max(Re λ)`.

**K_d'yi büyütmek bir noktadan sonra kötüleştirir.** K_p = 20 sütunu:
| K_d | en kötü σ |
|---|---|
| 3 | −3.0 |
| 4.3 (kritik sönüm) | **−4.3** (en iyi) |
| 8 | −1.25 |

Kritik değeri (K_d² = 2K_p − 21.58) geçince iki özdeğer gerçel olur ve **ayrılır**: biri hızlanır, öteki sıfıra
yaklaşır. Kapı amortisörü benzetmesi: çok güçlüyse kapı balın içindeymiş gibi kapanır. Pratikte iki bedeli daha var:
D terimi **gürültüyü büyütür** ve **kuvvet** ister.

**Haritanın tuzağı:** renk skalası `vmin=-5` ile kesildi. −5'ten küçük her değer aynı koyu renkte göründü. Skalayı
kesmek bilgiyi saklar.

---

## 5. Ayrık PID: türev tekmesi, türev filtresi, anti-windup

### 5.1 Neden sınıf?
PD'nin hafızası yok, fonksiyon yeter. I terimi eklenince **geçmiş** önemli hâle gelir: integral ve önceki hata.
Hafıza nesnenin içinde, `reset()` ile her bölüm başında sıfırlanır.

### 5.2 Bilgisayardaki PID
```
e_k   = referans − ölçüm
I_k   = I_{k-1} + Ki·e_k·dt                 (Ki ile çarpılmış sakla → kuvvet biriminde)
D_ham = (e_k − e_{k-1})/dt   veya ölçülen türev
D_k   = α·D_{k-1} + (1−α)·D_ham,  α = τ/(τ+dt)
u_k   = kırp(Kp·e_k + I_k + Kd·D_k,  ±u_max)
```

### 5.3 Dört gerçek dünya ayrıntısı
| ayrıntı | problem | çözüm |
|---|---|---|
| **Ölçülen türev** | Farkla türev gürültüyü büyütür: 0.01 / 0.02 = 0.5 | θ̇ durumda varsa `error_dot` olarak ver |
| **Türev tekmesi** | İlk çağrıda e_{k−1} yok; sıfır sayarsan K_d·e/dt'lik dev bir sıçrama olur | İlk çağrıda türev = 0 |
| **Türev filtresi** | Gürültülü türev motoru titretir | Birinci dereceden alçak geçiren filtre. τ saniye sonra %63 (1 − e⁻¹). τ, en hızlı moddan çok küçük olmalı (gecikme kararlılığı bozar). Bizde 0.01–0.02 s |
| **Anti-windup** | Doymuşken integral şişer, hata dönünce çıktı uzun süre yanlış yönde takılı kalır | Kıstırma: doymuşken **ve** hata aynı yönde zorluyorsa integrali güncelleme |

### 5.4 Anti-windup sayılarla
K_i = 1, u_max = 1, dt = 0.1. Hata 10 s boyunca +1, sonra −1.
- **Anti-windup yokken:** integral 10'a şişer. Hata dönünce önce 1'in altına inmesi gerekir:
  (10 − 1)/0.1 = **90 adım = 9 s** boyunca motor sonuna kadar **yanlış yöne** iter. Ölçülen 89 adım; fark, 0.1'in yüz
  kez toplanınca 9.99999999999998 olmasından.
- **Anti-windup varken:** integral 1'de kalır. Hata dönünce **anında** düşer (0 adım).

**Kodun anahtar fikri, "aday" değer:** integralin yeni değeri önce geçici bir değişkende hesaplanır, çıktı onunla
bulunur, doyma ve yön kontrolünden sonra kabul ya da reddedilir. Üç durum:

| durum | integral |
|---|---|
| doyma yok | kabul |
| doymuş, hata **aynı** yönde | **red** |
| doymuş, hata **ters** yönde | kabul (zaten küçültüyor) |

---

## 6. Doyma ve fiziksel sınır (43°)

Deney: AnglePD(20, 3), F_max = 10 N. Sınır 0.70 ile 0.75 rad arasında.

**Fizik:** sarkacı θ açısında **tutmak** (θ̈ = 0) için θ denklemi:

$$ m l\cos\theta\,\ddot x = m g l\sin\theta \;\Rightarrow\; \ddot x = g\tan\theta,\qquad F = (M+m)\,g\tan\theta $$

$$ F \le 10 \;\Rightarrow\; \theta_{\max} = \arctan\frac{10}{(M+m)g} = 0.747\ \text{rad} \approx 43° $$

- Bu **motorun** sınırı, kontrolcünün değil. Durağan başlayan ve 43°'den fazla yatık bir sarkacı 10 N ile **hiçbir**
  kontrolcü kurtaramaz.
- 0.5–0.7 rad arasında kontrolcü **doymuş ama kurtarıyor**. Doyma tek başına başarısızlık değil.
- Kontrolcüleri karşılaştırırken önce fiziksel sınırı hesapla: "40°'de düşüyor" ifadesi 43°'lük sınırın yanında
  anlam kazanır.

**Aynı denklemin ikinci hayatı:** ẍ = g·tanθ ≈ gθ → **yatıklık bir gaz pedalıdır** (kaskadın temeli, Bölüm 7).

---

## 7. Kaskad PID

### 7.1 Yapı
```
x_ref →(Σ)→ DIŞ DÖNGÜ (yavaş, PD) → θ_ref → İÇ DÖNGÜ (hızlı, AnglePD) → F → CARTPOLE
        ↑ x, ẋ                                    ↑ θ, θ̇
```
- **İç:** "sarkacı θ_ref'e götür, orada tut."
- **Dış:** "araba hedefe göre burada → sarkaç ne kadar yatmalı?"

### 7.2 Anahtar fikir: yatıklık = gaz pedalı
Sarkaç θ'da tutuluyorsa ẍ ≈ gθ. Dış döngünün gördüğü sistem **ẍ ≈ g·θ_ref**: gaz pedallı bir araba.

$$ \theta_{ref} = k_{p,x}\,(x_{ref} - x) - k_{d,x}\,\dot x,\qquad |\theta_{ref}| \le 0.2 $$

- **İşaretler sezgisel:** hedefe doğru yat, hıza karşı fren yap. Dış döngü **doğrudan etkili** (standart PID). Ters
  çevirme işini iç döngü yapıyor.
- **Eşdeğer K:** açılınca F = K_p k_px x + K_p k_dx ẋ + K_p θ + K_d θ̇. Bizim kazançlarla F = 4x + 6ẋ + 40θ + 5.35θ̇,
  yani K = [−4, −6, −40, −5.35]. **x ve ẋ katsayıları pozitif**: K_x deneyindeki "ters işaret" kaskadın içinde
  kendiliğinden çıkıyor.

### 7.3 Ters tepki kendiliğinden doğuyor
Hedef sağda → θ_ref > 0 → sarkaç dik iken iç döngü F = K_p(0 − θ_ref) < 0 → araba **sola** gider → sarkaç sağa yatar
→ araba sağa ivmelenir. **Ölçüm:** dik ve durağan başlangıç, hedef +1 m → araba önce **−6.3 cm**'ye gidiyor
(t ≈ 0.3 s), sonra 1 m'ye varıyor. Aşama 2'deki RHP sıfırının zaman içindeki görüntüsü.

### 7.4 Bant genişliği ayrımı
Dış döngü "θ_ref istersem sarkaç **anında** gelir" varsayıyor. Bu ancak iç döngü çok daha hızlıysa doğru.
**Kural: iç döngü dış döngüden 3–10 kat hızlı olmalı.**

| iç döngü | iç ω_n | kararlı dış kombinasyon (1500 içinde) | aynı dış (0.1, 0.15) ile |
|---|---|---|---|
| K_p = 20, K_d = 3 | 4.3 rad/s | 139 (%9) | **kararsız** (λ = +0.33 ± 4.05i; 10 s sonra x = 0.56, θ 0.49'a kadar sallanıyor) |
| K_p = 40, K_d = 5.35 (ζ = 0.7) | 7.6 rad/s | 482 (%32) | **kararlı** |

Dış ω_n ≈ √(g·k_px) = √0.98 ≈ 1 rad/s → iç/dış ≈ 7.6 ✓. Dikkat: hangi iç döngüyü seçersen seç, ulaşılabilen en
hızlı sönüm ~−1.3'te tıkanıyor. Bu, RHP sıfırının (z = 4.43) koyduğu temel sınır.

### 7.5 Doğrulama (x₀ = 0.5, θ₀ = 0.1, F_max = 10)
- x önce **0.65'e** çıkıyor (iki sebep üst üste: başlangıçtaki sağa yatıklığı yakalamak + ters tepki), ~3.4 s'de
  ±2 cm içinde, 10 s'de sıfır. Max |F| = 6 N.
- θ, θ_ref'i ~0.15–0.2 s gecikmeyle ve hafif aşmayla takip ediyor. Kapalı döngüde zayıf sönümlü bir mod var:
  λ = −1.14 ± 6.78i (ζ ≈ 0.17, periyot 0.93 s). Grafikteki çınlama ve x'teki basamaklar bu.
- 3 m uzaktan başlangıç: θ_ref kırpıldığı için max |θ| = 0.30 rad, araba yine hedefe varıyor.

### 7.6 θ_ref neden ±0.2 rad'da kırpılıyor?
Sarkaç doğrusal bölgede kalsın, 43° sınırından uzak durulsun, uzak hedefte dış döngü "45° yat" demesin. Mevcut `PID`
sınıfının `u_max`'ı dış döngüde açı sınırı olarak işe yarıyor. ⚠ Buraya motor sınırını (`u_max`) değil, açı sınırını
(`theta_ref_max`) vermek gerekiyor.

---

## 8. Yazılım tasarımı ve test stratejisi

### 8.1 Dört başlangıç kararı
1. **Ortak arayüz** `reset()` + `act(s) -> F`. `evaluate.py` karşısındakinin PID mi LQR mi RL mi olduğunu bilmeyecek.
2. **Entegrasyon tek yerde:** `run_closed_loop` sadece kontrolcüyü sarmalayıp `cp.simulate`'i çağırıyor.
3. **Motor sınırı sistem tarafında** (`run_closed_loop`, sonra ortam). Kontrolcünün kendi `u_max`'ı sadece anti-windup
   için. Aksi hâlde sınırlamayı unutan bir kontrolcü (RL) haksız avantaj kazanır.
4. **Paylaşılan kod `.py`'ye, deneyler deftere.**

Ek kurallar:
- Kontrolcü kendi periyodunu (`dt`) bilir.
- `reset()` başlangıç değerlerinin tek kaynağı (`__init__` onu çağırır).
- `if self.u_max is not None` (`if self.u_max:` değil; 0 değeri yanlışlıkla kapatır).
- `return float(u)`.

### 8.2 Test stratejisi
| test türü | örnek | neyi yakalar |
|---|---|---|
| **Birim testi** | `test_pid_integral`, `test_cascade_theta_ref_is_clipped` | tek bir parçanın hatası, nokta atışı |
| **Bilinen doğruyla karşılaştırma (regresyon)** | `test_anglepd_matches_function` | yeniden yazarken bozma. `s[0]/s[1]` hatasını **yakalayan tek test** buydu |
| **Sistem testi** | `test_cascade_regulates_x_and_theta`, `inverse_response`, `needs_fast_inner_loop` | "gerçekte çalışıyor mu?" |
| **Testin kendisini test et** | anti-windup testinde `anti_windup=False` karşılaştırması; reset testinde reset'i atlayan koşu; F_max testinde `max == 10` | testin bir şey kanıtlayabildiğini gösterir. **Özellik bozukken test patlamalı** |

**İlkeler:**
- Ondalık sayıyı `==` ile değil toleransla karşılaştır (0.1 × 10 = 0.9999999999999999).
- Test adı test ettiği şeyi söylesin.
- Assert'ü yüksek sesle oku.
- Test edilen durum gerçekten oluşmalı (sınıra dayanmayan bir doyma testi hiçbir şey kanıtlamaz).
- Reset testi sadece τ > 0 iken anlamlı. AnglePD'nin hafızasında kalan tek şey filtre.
- Hatalı kodla diğer iki test **geçti**: her test farklı bir soruyu cevaplar; "doğru mu?" sorusunu sadece bağımsız bir
  referans cevaplar.

---

## 9. Mülakat soruları ve örnek cevaplar

**S1. P, I ve D ne işe yarar?**
P bugünkü hataya (yay), D hatanın değişim hızına (amortisör, sönüm), I geçmiş hataların toplamına (kalıcı hatayı,
örneğin sabit rüzgârı siler) tepki verir. Cartpole dengesinde I genelde gerekmez: dengede tutmak için sabit bir
kuvvet gerekmiyor.

**S2. Kapalı döngü kararlılığını nasıl gösterdin?**
Kontrolcüyü F = −Ks biçiminde yazdım, kapalı döngü matrisi A − BK'nın özdeğerlerine baktım. Sadece açıya PD için
karakteristik denklem ikinci dereceden: iki katsayının pozitifliği K_p > (M+m)g ve K_d > 0 veriyor. Bunu K_p × K_d
kararlılık haritasıyla sayısal olarak doğruladım ve simülasyonla eşleştirdim.

**S3. Sadece açıya PD kararlı mı?**
Sarkaç modu kararlı, sistem değil: x ve ẋ'ye ait iki özdeğer sıfırda kalıyor (çift integratör). Araba sabit hızla
sürükleniyor; ölçümde 0.3 m/s.

**S4. Kazancı artırmak her zaman iyi mi?**
Hayır. K_d kritik sönümü geçince bir özdeğer sıfıra yaklaşıyor ve sistem yavaşlıyor. Ayrıca D gürültüyü büyütüyor
ve kuvvet istiyor; büyük kazanç doymaya ve gecikmeye karşı payı azaltıyor. Sınırda kararlı bir tasarım dijital
uygulamada ZOH gecikmesi yüzünden kararsızlaştı.

**S5. Anti-windup nedir, neden gerekir?**
Motor doymuşken integral hatayı biriktirmeye devam eder. Hata dönünce bu birikim çıktıyı uzun süre yanlış yönde
tutar: örneğimizde 9 s. Kıstırma yöntemiyle doymuşken ve hata aynı yönde zorluyorken integrali güncellemiyorum;
çıktı hata dönünce anında geri geliyor.

**S6. Türev terimini nasıl uyguladın?**
Durum içinde ölçülen θ̇ olduğu için farkla türev almıyorum, gürültüyü büyütmemek için. Farkla hesaplandığında ilk
çağrıda türev tekmesini önlüyorum ve birinci dereceden bir filtre (τ) uyguluyorum. τ en hızlı moddan çok küçük
seçilmeli, yoksa filtrenin gecikmesi kararlılığı bozar.

**S7. PID cartpole için neden zor, kaskad neden gerekli?**
Tek girdi iki çıktıyı (θ ve x) yönetmeli ve F→x'te bir RHP sıfırı var: x'e doğrudan bakan sezgisel geri besleme
("geri it") sistemi kararsızlaştırıyor. Kaskadda dış döngü x'ten bir açı referansı üretiyor (yatıklık = gaz pedalı,
ẍ ≈ gθ), hızlı iç döngü açıyı takip ediyor. Dış döngünün işaretleri sezgisel kalıyor, ters tepki kendiliğinden
çıkıyor.

**S8. İç ve dış döngü arasındaki hız ilişkisi?**
İç döngü 3–10 kat hızlı olmalı, çünkü dış döngü θ_ref'in anında takip edildiğini varsayıyor. İç döngüm ω_n ≈ 7.6,
dışı ≈ 1 rad/s. Aynı dış döngü ω_n ≈ 4.3'lük iç döngüyle kararsız çıktı; bunu bir testle belgeledim.

**S9. Arabayı sağa götürmek için neden önce sola gidiyor?**
Sağa ivmelenmek için sarkacın sağa yatık olması lazım (ẍ ≈ gθ). Dik bir sarkacı sağa yatırmanın tek yolu arabayı
önce sola çekmek. Transfer fonksiyonundaki +√(g/l) sıfırının zaman içindeki görüntüsü. Ölçümde 1 m'lik hedefte
−6 cm'lik ters sapma.

**S10. Doyma performansı nasıl etkiliyor?**
Çekim bölgesini daraltıyor. 10 N ile durağan başlangıçta tutulabilecek en büyük açı arctan(F_max/((M+m)g)) = 43°.
Deneyde PD 40°'yi kurtardı, 43°'ü kurtaramadı. Bu kontrolcüden bağımsız bir fiziksel sınır.

**S11. Kontrolcü kodunu nasıl test ettin?**
Birim testleri (integral, doyma, türev tekmesi, anti-windup, filtre zaman sabiti), fonksiyonel referansla regresyon
testi, sistem testleri (yol haritası senaryosu, ters tepki, bant genişliği) ve testlerin kendisini doğrulayan
karşılaştırmalar: özelliği kapatınca test patlıyor mu?

---

## 10. Tuzaklar (bu aşamada gerçekten yaşananlar)

| tuzak | belirti | önlem |
|---|---|---|
| Durum indisi karışıklığı (`s[0]` / `s[2]`, `s[1]` / `s[3]`): **3 kez** | Grafik θ₀ = 0.1 yerine 0'dan başlıyor; regresyon testi kırmızı | İlk kontrol: "eğri başlattığım yerden mi başlıyor?" İleride isimli indisler (`X, XD, TH, THD`) |
| Grafiği düzeltirken çalışan kodu bozmak | Önce bir sonra iki hata | Bir seferde tek şey değiştir |
| A + BK ile eksili K | Bütün harita kırmızı, kod hata vermiyor | Bilinen noktayla (C = −3 ± 3.07i) doğrula |
| K'da eksik eleman / yanlış sıra | `(4,4) (4,3)` boyut hatası | Durum sırasıyla aynı, 4 eleman, boş yere 0 |
| `0.+4.292j`'yi "0.4" okumak | Yanlış kararlılık gerekçesi | `j` sanal birim; gerçel kısım 0 |
| Sıfır özdeğerleri yok saymak | "Kararlı" denip araba kayıyor | Bütün özdeğerler σ < 0 olmalı |
| Kısa simülasyon | Yavaş kararsız mod 5 s'de görünmüyor | Süre ≥ 2–3 periyot |
| Döngü değişkeninin döngüden sonra kullanılması | "A" noktası (40, 8)'de | Farklı isimler (`kp` / `Kp`) |
| Hücre hata verdi ama grafik çıktı | Lejant ve eksen etiketleri yok | Her çalıştırmadan sonra hücrenin altına bak |
| Renk skalasının kesilmesi | Sağ üst "en iyi" sanıldı | Skalanın doyduğu yeri kontrol et |
| Dosyayı kaydetmemek / `reload` unutmak | Düzeltme var ama test hâlâ kırmızı | `Ctrl+S`, `importlib.reload` |
| Dış döngüye motor sınırı vermek | θ_ref kırpılmıyor | `theta_ref_max`; birim testi |
| `abs(u − 1.0) == 1.0` | Testin tersini sınamak | Assert'ü yüksek sesle oku |

---

## 11. Kaynak haritası

| konu | kaynak |
|---|---|
| PID yapısı, P / I / D etkileri, türev filtresi, anti-windup | Åström & Murray, *Feedback Systems*, **"PID Control"** bölümü; `pid-kontrol-ders-notlari.pdf` |
| Kaskad kontrol, iç / dış döngü bant genişliği | Åström & Murray, PID bölümü (cascade) |
| RHP sıfırları, ters tepki, temel sınırlar | Åström & Murray, **"Frequency Domain Design"** / **"Fundamental Limits"** |
| Durum geri beslemesi, A − BK, kutup yerleştirme | Brunton: "Pole Placement for the Inverted Pendulum on a Cart"; Åström & Murray, "State Feedback" |
| Ayrık zaman, ZOH, gecikmenin kararlılığa etkisi | Åström & Murray, "Linear Systems" (örneklenmiş sistemler) |

---

## 12. Kendini sına

1. K_p = 15, K_d = 2 için ω_n ve ζ'yı hesapla. Kararlı mı? Salınımlı mı?
2. m iki katına çıkarsa sadece açıya PD için gereken en küçük K_p nasıl değişir? Fiziksel sebebi?
3. τ = 0.5 s seçilseydi (türev filtresi) ne olurdu? Neden?
4. Anti-windup olmadan, K_i = 2 ve aynı senaryoda takılı kalma süresi kaç saniye olurdu?
5. F_max = 20 N olsaydı fiziksel tutma sınırı kaç derece olurdu?
6. Kaskadda dış döngünün işaretini ters çevirirsen (θ_ref = −k_px(x_ref − x)) ne olur? Önce tahmin et, sonra koş.
7. Kaskadın eşdeğer K'sı neden x ve ẋ için pozitif katsayı veriyor? K_x deneyiyle ilişkisini açıkla.
8. Hangi testin hangi hatayı yakaladığını eşleştir: indis hatası, `u_max` / `theta_ref_max` karışıklığı, reset unutma.
9. Sınırda kararlı (σ = 0) bir tasarımı neden kabul etmeyiz? Simülasyondaki kanıtın ne?
10. Kaskadda ulaşılabilen en hızlı sönüm neden iç döngüyü ne kadar hızlandırırsan hızlandır ~−1.3'te tıkanıyor?
