# Aşama 1–2 Ders Notları: Cartpole Modelleme, Doğrusallaştırma ve Sistem Analizi

> **Nasıl okunur:** Her bölüm aynı sırayı izler: **Amaç → Sezgi → Matematik (terim terim) → Kodda nerede → Mülakatta
> ne derim.** Formüllerin hepsi `notebooks/01_derive_cartpole.ipynb` ve `02_linearize.ipynb` içinde çalışır hâlde;
> okurken defteri yanında aç. Sayılar bizim parametrelerimizle: M = 1 kg, m = 0.1 kg, l = 0.5 m, g = 9.81 m/s².
>
> VS Code'da `Ctrl+Shift+V` ile önizleme açarsan formüller düzgün görünür.

---

## İçindekiler

**Aşama 1 — Modelleme**
1. [Büyük resim: neden model?](#1-büyük-resim-neden-model)
2. [Durum (state) ve varsayımlar](#2-durum-state-ve-varsayımlar)
3. [Lagrange ile hareket denklemleri](#3-lagrange-ile-hareket-denklemleri)
4. [Sayısal entegrasyon: Euler, RK4, dt](#4-sayısal-entegrasyon-euler-rk4-dt)
5. [Modelin doğru olduğunu nasıl kanıtlarım?](#5-modelin-doğru-olduğunu-nasıl-kanıtlarım)

**Aşama 2 — Analiz**
6. [Doğrusallaştırma](#6-doğrusallaştırma)
7. [Özdeğerler ve kararlılık](#7-özdeğerler-ve-kararlılık)
8. [Kontrol edilebilirlik](#8-kontrol-edilebilirlik)
9. [Gözlemlenebilirlik](#9-gözlemlenebilirlik)
10. [Transfer fonksiyonu, sıfırlar ve ters tepki](#10-transfer-fonksiyonu-sıfırlar-ve-ters-tepki)
11. [Ayrıklaştırma](#11-ayrıklaştırma)
12. [Doğrusal model nerede bozuluyor?](#12-doğrusal-model-nerede-bozuluyor)

**Ekler**
13. [Mülakat soruları ve örnek cevaplar](#13-mülakat-soruları-ve-örnek-cevaplar)
14. [Tuzaklar](#14-tuzaklar)
15. [Kaynak haritası](#15-kaynak-haritası)
16. [Kendini sına](#16-kendini-sına)

---

# AŞAMA 1 — MODELLEME

## 1. Büyük resim: neden model?

Projede üç şey yapacaksın: **kontrolcü tasarlamak** (PID, LQR, MPC), **durum tahmin etmek** (Kalman) ve **RL ajanı
eğitmek**. Üçünün de ortak ihtiyacı aynı: bir **simülatör**. Simülatörün tek işi şu soruyu cevaplamak:

> "Sistem şu an bu durumda, şu kuvveti uyguluyorum. Bir sonraki an ne olur?"

Bu iki parçadan oluşur:

```
           ┌─────────────────────┐      ṡ      ┌──────────────────┐
 s, F ───► │  f(s, F)  (FİZİK)   │ ──────────► │  ENTEGRATÖR (RK4) │ ───► s(t + dt)
           │  Lagrange ile bulunur│            │  zamanda ilerler  │
           └─────────────────────┘             └──────────────────┘
```

- **f** "durumun değişim hızını" verir (Bölüm 3).
- **Entegratör**, bu hızı küçük adımlarla toplayıp zamanda ilerler (Bölüm 4).

Aşama 2'de ise aynı f'yi **analiz** için kullanıyoruz: dik duruşun yakınında f'yi doğrusal bir modelle değiştirip
"sistem doğası gereği nasıl davranıyor, ne kadar kontrol edilebilir, nerede zorlanacağız?" sorularını cevaplıyoruz.

**Mülakatta:** *"Kontrolcüyü ve RL ajanını aynı simülatörde test ettim. Simülatörün dinamiğini Lagrange yöntemiyle
kendim türettim, sympy ile sembolik olarak, enerji korunumu ve denge noktası testleriyle de sayısal olarak doğruladım."*

---

## 2. Durum (state) ve varsayımlar

### 2.1 Durum nedir?
**Tanım:** Sistemin geleceğini (girdileri de bilerek) hesaplamak için *şu an* bilinmesi gereken **en küçük** bilgi kümesi.

$$ s = \begin{bmatrix} x \\ \dot x \\ \theta \\ \dot\theta \end{bmatrix}
\quad\text{(araba konumu, araba hızı, açı, açısal hız)} $$

**Neden hızlar da var?** Mekanik sistemler **ikinci dereceden**dir: Newton F = m·a **ivmeyi** verir. İkinci dereceden
bir diferansiyel denklemin tek çözümü için iki başlangıç koşulu gerekir: konum ve hız. Aynı açıda duran ama farklı
hızlarla dönen iki sarkacın gelecekleri farklıdır. Sistemde 2 serbestlik derecesi (araba, sarkaç) var, her biri için
konum ve hız gerekiyor, bu da 4 durum eder.

**Serbestlik derecesi (DOF)** ile **durum boyutu** farklı şeyler: 2 DOF → 4 durum.

### 2.2 Durum denklemi

$$ \dot s = f(s, F) = \begin{bmatrix} \dot x \\ \ddot x(s,F) \\ \dot\theta \\ \ddot\theta(s,F) \end{bmatrix} $$

1. ve 3. satırlar tanım gereği zaten durumun içinde. Bütün fizik ẍ ve θ̈'nın içinde. Kodda:

```python
def f(s, F, p):
    xdd, thdd = accelerations(s, F, p)        # fizik
    return np.array([s[1], xdd, s[3], thdd])  # [ẋ, ẍ, θ̇, θ̈]
```

### 2.3 İşaret kuralı (projenin değişmezi)
θ = 0 **yukarı**, θ > 0 **sağa yatık**, x > 0 sağ, F > 0 sağa itiş.
⚠ Tedrake θ = 0'ı **aşağı** alır, Brunton'un kodunda başka bir kural ve sönüm var. Kaynaklar arasında geçerken
dönüşüm θ_bizim = θ_Tedrake − π. Bu dönüşümle sin → −sin, cos → −cos olur.

### 2.4 Varsayımlar (mülakatta mutlaka sorulur)
| varsayım | etkisi | gerçekte |
|---|---|---|
| Noktasal sarkaç kütlesi | Atalet momenti m l² | Dağılmış çubukta I = m l²/3 (uçtan) → Gymnasium'daki model budur |
| Kütlesiz çubuk | Çubuğun kendi ataleti yok | Hafif çubuk için iyi yaklaşım |
| Sürtünme yok | Enerji korunur → **test edilebilirlik** | Ray sürtünmesi b_x·ẋ, mafsal sönümü b_θ·θ̇ (sonra eklenecek) |
| Kuvvet doğrudan arabaya | Motor dinamiği yok | Gerçekte motor tork → kuvvet, gecikme, doyma |
| Sonsuz ray | x sınırı yok | Ortamda |x| < 2.4 m sınırı olacak |

Sürtünmesiz seçmenin **bilinçli bir faydası** var: F = 0 iken toplam enerji korunmalı. Bu da modeli ve entegratörü
test etmek için elimize hazır bir **doğru cevap** veriyor (Bölüm 5).

---

## 3. Lagrange ile hareket denklemleri

### 3.1 Neden Newton değil?
Newton'la serbest cisim diyagramı çizersin: arabaya etki eden kuvvetler ve sarkaca etki eden kuvvetler. İkisi
arasında **çubuk kuvveti** var. Bu bilinmeyen bir **iç kuvvet**; iki denklemde de geçer ve onu yok etmek için cebir
yaparsın. Sistem büyüdükçe (çift sarkaç, robot kol) bu iş içinden çıkılmaz hâle gelir.

**Lagrange** iç (kısıt) kuvvetlerini hiç görmez. Sadece **enerjileri** yazarsın, gerisi mekanik türevdir ve sympy'ye
bırakılabilir. Çok gövdeli sistemlerde (robotik, uçak, füze) standart yöntem bu.

### 3.2 Tarif (5 adım)
1. **Genelleştirilmiş koordinatlar** seç: q = [x, θ]. Sistemi tek türlü tarif eden, birbirinden bağımsız
   değişkenler. Kısıtlar (çubuk boyu sabit) koordinat seçimine gömülü.
2. Her kütlenin **konumunu** q cinsinden yaz.
3. **Hızları** türevle bul. **Kinetik enerji T** ve **potansiyel enerji V**.
4. **L = T − V** (Lagrangian).
5. Her koordinat için **Euler–Lagrange denklemi**:
$$ \frac{d}{dt}\left(\frac{\partial L}{\partial \dot q_i}\right) - \frac{\partial L}{\partial q_i} = Q_i $$

### 3.3 Adım 2: konum
θ dikeyden ölçüldüğü için çubuğun yatay bileşeni l sinθ, düşey bileşeni l cosθ:

$$ x_p = x + l\sin\theta, \qquad y_p = l\cos\theta $$

**Kontrol:** θ = 0 → (x, l) tepede ✓. θ = π/2 → (x + l, 0), sağda yatay ✓. θ = π → (x, −l) altta ✓.

### 3.4 Adım 3: hız (zincir kuralı)
θ zamana bağlı olduğu için d/dt sinθ = cosθ · θ̇:

$$ \dot x_p = \dot x + l\cos\theta\,\dot\theta, \qquad \dot y_p = -l\sin\theta\,\dot\theta $$

**Sezgi kontrolü** (araba duruyor, sarkaç θ̇ > 0 ile dönüyor):
- θ = 0'da uç tamamen **yatay** hareket eder (tepedeki nokta dönünce önce yana gider).
- θ = 90°'de uç tamamen **aşağı** hareket eder.

### 3.5 Kinetik enerji

$$ T = \tfrac12 M\dot x^2 + \tfrac12 m(\dot x_p^2 + \dot y_p^2) $$

Açınca (sin² + cos² = 1):

$$ T = \underbrace{\tfrac12 (M+m)\dot x^2}_{(a)} + \underbrace{m l\,\dot x\,\dot\theta\cos\theta}_{(b)} + \underbrace{\tfrac12 m l^2\dot\theta^2}_{(c)} $$

| terim | anlamı |
|---|---|
| (a) | İkisi birlikte kayıyormuş gibi öteleme enerjisi (toplam kütle) |
| (b) | **Bağlaşım (coupling).** Hem ẋ hem θ̇ içeriyor. Arabayı itince sarkacın yatmasının, sarkaç düşünce arabanın kaymasının matematiksel kaynağı |
| (c) | Sarkacın mafsal etrafında dönme enerjisi; m l² noktasal kütlenin atalet momenti |

(b) olmasaydı L ikiye ayrılırdı: bir parçada sadece x, diğerinde sadece θ olurdu. Araba ile sarkaç birbirini hiç
etkilemezdi ve **kontrol imkânsız** olurdu, çünkü F sadece arabaya etki ediyor. Sarkacı kontrol edebilmemizin tek
sebebi bu terim.

### 3.6 Potansiyel enerji

$$ V = m g y_p = m g l\cos\theta $$

Araba hep aynı yükseklikte, katkısı yok. V tepede (θ = 0) en büyük. Sistem V'yi azaltmaya meyilli olduğu için
sarkaç düşer.

### 3.7 Euler–Lagrange'ın anlamı: kılık değiştirmiş Newton
En basit örnek: sarkaçsız araba. L = ½Mẋ².
- ∂L/∂ẋ = Mẋ → **momentum**
- d/dt(Mẋ) = Mẍ → momentumun değişim hızı
- ∂L/∂x = 0
- Q_x = F

Sonuç M ẍ = F, yani Newton. Euler–Lagrange denklemi şunu söyler: **"genelleştirilmiş momentumun değişimi =
genelleştirilmiş kuvvet"**, ve bunu iç kuvvetlerle uğraşmadan, her koordinat için yapar.

| parça | anlamı |
|---|---|
| ∂L/∂q̇ | o koordinatın genelleştirilmiş momentumu (x için doğrusal momentum, θ için açısal momentum) |
| d/dt(∂L/∂q̇) | momentum nasıl değişiyor |
| −∂L/∂q | konuma bağlı kuvvetler: yerçekimi (∂V/∂q) ve koordinata bağlı kinetik enerjiden gelen "sahte" kuvvetler (merkezkaç vb.) |
| Q | dışarıdan, korunumlu olmayan kuvvetler. **Sanal iş** ile bulunur: δW = F·δx + 0·δθ → Q_x = F, Q_θ = 0 |

**Bizim sistemde momentum:**
$$ p_x = \frac{\partial L}{\partial \dot x} = (M+m)\dot x + m l\dot\theta\cos\theta = M\dot x + m\dot x_p $$
Bu, **arabanın ve sarkacın toplam yatay momentumu**. F = 0 iken L, x'e bağlı olmadığı için (∂L/∂x = 0) p_x
**korunur**. Sarkaç sağa düşerken ucu sağa hız kazanır; toplam momentum sıfır kalsın diye araba sola kayar.
Animasyonda gördüğün davranışın açıklaması bu. Genel ilke: **bir koordinat L'de görünmüyorsa (döngüsel koordinat),
onun momentumu korunur** (Noether).

### 3.8 İki denklem, terim terim

**x denklemi** (Q_x = F):
$$ (M+m)\ddot x + m l\cos\theta\,\ddot\theta - m l\sin\theta\,\dot\theta^2 = F $$

| terim | fizik |
|---|---|
| (M+m)ẍ | bütün sistemi tek parça gibi ivmelendirmek |
| m l cosθ θ̈ | sarkaç açısal ivmelenince ucu yana ivmelenir; bunun tepkisi arabaya geçer |
| −m l sinθ θ̇² | dönen kütlenin çubuğa uyguladığı merkezkaç çekişinin yatay bileşeni |

**θ denklemi** (Q_θ = 0):
$$ m l\cos\theta\,\ddot x + m l^2\ddot\theta - m g l\sin\theta = 0 $$

| terim | fizik |
|---|---|
| m l² θ̈ | sarkacın dönme ataleti × açısal ivme |
| −m g l sinθ | yerçekimi torku: dikten uzaklaştıkça artar ve **daha da** uzaklaştırır |
| m l cosθ ẍ | araba ivmelenince sarkaç bunu tork olarak hisseder (otobüs kalkınca geriye savrulmak) |

### 3.9 Kütle matrisi ve çözüm
Her iki denklem de ẍ ve θ̈'da **doğrusal**. Matris biçiminde:

$$ \underbrace{\begin{bmatrix} M+m & m l\cos\theta \\ m l\cos\theta & m l^2 \end{bmatrix}}_{\mathbf M(q)}
\begin{bmatrix}\ddot x\\ \ddot\theta\end{bmatrix} =
\begin{bmatrix} F + m l\dot\theta^2\sin\theta \\ m g l\sin\theta\end{bmatrix} $$

- **M(q)** kütle (atalet) matrisi: simetrik ve pozitif tanımlı. Her mekanik sistemde böyledir, çünkü T = ½ q̇ᵀ M(q) q̇ > 0.
- det M(q) = m l²(M + m sin²θ) > 0 → **her θ'da tersinir** → ẍ, θ̈ her zaman tek türlü bulunur.
  Formüllerdeki paydanın (M + m sin²θ) kaynağı bu determinant.
- Genel robotik biçimi: **M(q) q̈ + C(q, q̇) q̇ + G(q) = B u**. Burada C merkezkaç/Coriolis, G yerçekimi. Bu biçimi
  tanıman mülakatta artı puan.

Çözüm (Ek A.1, `cartpole.accelerations`):

$$ \ddot x = \frac{F + m\sin\theta\,(l\dot\theta^2 - g\cos\theta)}{M + m\sin^2\theta} $$

$$ \ddot\theta = \frac{-F\cos\theta - m l\dot\theta^2\sin\theta\cos\theta + (M+m) g\sin\theta}{l\,(M + m\sin^2\theta)} $$

### 3.10 Toplam enerji

$$ E = T + V = \tfrac12 (M+m)\dot x^2 + m l\dot x\dot\theta\cos\theta + \tfrac12 m l^2\dot\theta^2 + m g l\cos\theta $$

F = 0 ve sürtünme yokken dE/dt = 0. Genel olarak dE/dt = F·ẋ, yani kuvvetin yaptığı güç.

### 3.11 sympy'de nasıl yapılıyor (defter 01, bölüm A)
| adım | kod | neden |
|---|---|---|
| zamana bağlı değişken | `x = sp.Function('x')(t)` | `symbols` deseydik türevi 0 olurdu |
| hız | `xp.diff(t)` | zincir kuralını sympy uygular; çıktıdaki `Derivative(theta(t), t)` = θ̇ |
| ½ | `sp.Rational(1, 2)` | tam kesir; `0.5` ondalık yaklaşım getirir, sadeleştirmeyi bozar |
| ∂L/∂ẋ | `L.diff(x.diff(t))` | sympy türev nesnesine göre de türev alabilir |
| sembole çevirme | `subs` ile önce 2. türevler | sırayı tersine yaparsan ẍ'nin içindeki ẋ önce değişir, ẍ kaybolur |
| kütle matrisi | `sp.linear_eq_to_matrix` | denklemlerin ẍ, θ̈'da doğrusal olduğunu gösterir |
| çöz | `sp.solve(eqs, [xdd, thdd])` | 2×2 doğrusal sistem |
| numpy'a | `sp.lambdify` | sembolik ifadeyi hızlı bir Python fonksiyonuna çevirir |

**Gymnasium farkı:** `CartPole-v1` dağılmış kütleli çubuk kullanır (yarım uzunluk, I = m l²/3), aksiyonları
ayrıktır (±10 N) ve Euler entegrasyonu yapar. Denklemleri bu yüzden farklı görünür; biz kendi ortamımızı yazıyoruz.

---

## 4. Sayısal entegrasyon: Euler, RK4, dt

### 4.1 Problem
ṡ = f(s, F) diferansiyel denkleminin kapalı biçimde (formülle) çözümü yok. Bilgisayar çözümü **küçük adımlarla**
yaklaşık olarak kurar.

### 4.2 Euler
Taylor açılımından: s(t+h) = s(t) + h·ṡ(t) + ½h²s̈(t) + …  Euler ilk iki terimi alır, gerisini atar:

$$ s_{k+1} = s_k + h\, f(s_k, F_k) $$

"Şimdiki eğimle düz git." Attığı terim h² mertebesinde: **yerel hata O(h²)**. T/h adım boyunca bu hatalar birikir:
**global hata O(h)**. dt'yi yarıya indirirsen hata da yarıya iner.

**Euler neden enerji pompalıyor?** Sönümsüz bir salınıcıyı en basit hâline indirelim: ż = iωz (dönen bir vektör,
genliği sabit). Bir Euler adımı z'yi (1 + iωh) ile çarpar. Bunun büyüklüğü:

$$ |1 + i\omega h| = \sqrt{1 + \omega^2 h^2} > 1 $$

Her adımda genlik biraz **büyür**, yani enerji artar. Nedeni şu: dairesel yörüngede teğet doğrultuda düz gidince
her seferinde biraz **dışarı** çıkarsın. Bizim ölçümümüz: dt = 0.02'de 10 s sonunda enerji %220 artmış.

### 4.3 RK4
Adımın içinde eğime 4 kez bakıp ağırlıklı ortalama alır:

```
k1 = f(s)                  adımın başındaki eğim
k2 = f(s + h/2·k1)         k1 ile yarıya gidip oradaki eğim
k3 = f(s + h/2·k2)         k2 ile yarıya gidip oradaki eğim (daha iyi orta tahmini)
k4 = f(s + h·k3)           sona gidip oradaki eğim
s_yeni = s + h/6·(k1 + 2k2 + 2k3 + k4)
```

Ağırlıklar (1, 2, 2, 1)/6, Simpson integral kuralıyla aynı: uçlar 1, orta 4 (= 2 + 2). Taylor açılımını h⁴'e kadar
doğru yakalar: **global hata O(h⁴)**. dt'yi yarıya indirince hata 16'da birine iner. Bedeli adım başına 4 f
çağrısı. Ama aynı doğruluk için Euler'den yüzlerce kat az adım gerektiği için çok daha ucuza gelir.

RK4'ün ż = iωz için büyütme katsayısı |R(iωh)| ≈ 1 − (ωh)⁶/144 < 1. Euler'in tersine çok hafif **sönümler**,
ama küçük h'de ihmal edilebilir. dt = 0.02 ve ω = 4.65 için katsayı 0.9999999955.

### 4.4 Ölçümlerimiz (defter 01, D1)
F = 0, θ₀ = 0.3, 10 s; en büyük göreli enerji kayması:

| yöntem | max \|ΔE/E₀\| |
|---|---|
| Euler, dt = 0.02 | 2.2 (%220!) |
| Euler, dt = 0.001 | 7.8e-2 |
| RK4, dt = 0.02 | 7.4e-6 |
| RK4, dt = 0.02, 20 alt adım | 1.8e-11 |

**RK4, dt = 0.02'deki sıçramalar nereden?** Sarkaç alttan geçerken |θ̇| ≈ 9.2 rad/s'ye çıkıyor. Bir adımda
θ'nın değişimi θ̇·dt ≈ 0.18 rad, yani hareket adıma göre hızlı ve yerel hata büyüyor. Dik denge civarında, kontrolün
çalıştığı bölgede, hareket yavaş olduğu için bu sorun yok.

### 4.5 Kontrol periyodu ≠ entegrasyon adımı (önemli)
İkisi farklı kavramlar:
- **Kontrol periyodu (dt = 0.02 s, 50 Hz):** kontrolcünün (ya da RL ajanının) ölçüm alıp yeni bir kuvvet hesapladığı
  aralık. Kuvvet bu süre boyunca **sabit tutulur**: sıfırıncı derece tutma (ZOH), gerçek dijital kontrolcülerin
  çalışma biçimi.
- **Entegrasyon adımı (h):** fiziğin hesaplandığı aralık. Bir kontrol periyodu içinde birden fazla olabilir.

Kodda: `simulate(..., dt=0.02, substeps=20)` → kuvvet 0.02 s sabit, fizik 0.001 s adımlarla.
Karar (KARARLAR.md): **dt = 0.02, substeps = 20.**

### 4.6 dt'yi nasıl gerekçelendirirsin?
1. **Fiziğin zaman ölçeği:** kararsız modun ikiye katlanma süresi ln2/λ ≈ 0.15 s (Bölüm 7). dt = 0.02 bu sürede
   ~7 müdahale demek. Kaba kural: örnekleme frekansı en hızlı ilgili dinamiğin 10–20 katı olmalı.
2. **Yakınsama testi:** dt = 0.02 ve dt = 0.001 ile koş, karşılaştır. 5 s'de θ farkı 2e-4 rad: yeterli.
3. **Enerji testi:** yukarıdaki tablo.

### 4.7 Neden scipy'nin `solve_ivp`'si değil?
`solve_ivp` adaptif adımlı ve çok hassas. Ama kontrol ve RL döngüsünde **her 0.02 s'de dışarıdan yeni bir girdi**
geliyor, ve adım sayısının **sabit ve deterministik** olması gerekiyor (tekrarlanabilir deneyler, hız). Sabit
adımlı RK4 bu yüzden standart seçim. `solve_ivp`'yi testlerde **referans** olarak kullanıyoruz.

*(İleri seviye)* Uzun süreli enerji davranışı önemliyse **simplektik** entegratörler (yarı-örtük Euler, Verlet)
enerjiyi sürüklemeden salınımlı tutar. Bizim gibi kısa bölümlü ve kontrollü simülasyonda RK4 yeterli.

---

## 5. Modelin doğru olduğunu nasıl kanıtlarım?

Mülakatta en sık gelen soru: *"Simülatörünün doğru olduğunu nereden biliyorsun?"* Tek bir test yetmez; her test
farklı bir hata türünü yakalar (`test_all.py`):

| test | neyi yakalar | neyi **kaçırır** |
|---|---|---|
| **Enerji korunumu** (F = 0) | entegratör hataları, T ile f arasında tutarsızlık | F teriminin işaret hatası (F = 0'da görünmez!) |
| **Denge noktaları** (θ = 0, π'de ṡ = 0) | sabit terimlerdeki hatalar (yanlış g işareti vb.) | dinamiğin geri kalanı |
| **Fiziksel sezgi** (θ > 0 sağa düşer; F > 0 → araba sağa, sarkaç sola) | **işaret kuralı** hataları, F'nin işareti | büyüklük hataları |
| **sympy çapraz kontrolü** (100 rastgele durum) | elle yazımdaki yazım hataları (`l*M` yerine `M` gibi) | türetmenin kendisindeki kavramsal hata (ikisi aynı yanlış varsayımı yaparsa) |
| **dt yakınsaması** | adım boyunun yetersizliği | model hataları |

**İlke:** Bağımsız yollardan doğrula. Elle yazılmış formül + sympy türetmesi + enerji gibi fiziksel bir değişmez,
üçü birden aynı hatayı yapamaz.

*Gerçek örnek:* İlk denemedeki `Pendilum - Inverse Pendilum/dynamics.py` dosyasında paydada `l*M + m sin²θ` yazıyordu
(doğrusu M + m sin²θ) ve yerçekimi terimlerinin işareti θ = 0 **aşağı** kuralına göreydi. Denge testi ve sympy
kontrolü ikisini de anında yakalardı.

---

# AŞAMA 2 — ANALİZ

## 6. Doğrusallaştırma

### 6.1 Amaç
Kontrol teorisinin güçlü araçları (özdeğer analizi, kontrol edilebilirlik, LQR, Kalman, kutup yerleştirme) şu
biçimdeki sistemler için kurulmuştur:

$$ \dot s = A s + B F $$

Bizim f'mizde sin, cos ve θ̇² var; doğrusal değil. Çözüm: **çalışmak istediğimiz noktanın yakınında** f'yi doğrusal
bir modelle değiştirmek.

### 6.2 Sezgi
"Dünya yuvarlak ama oda düz görünür." Pürüzsüz her eğri yeterince yakından bakınca düzdür. Matematikteki karşılığı
Taylor açılımının ilk mertebesi:

$$ f(s^*+\delta s,\, F^*+\delta F) \approx \underbrace{f(s^*, F^*)}_{=0\ \text{(denge)}} + \underbrace{\frac{\partial f}{\partial s}\Big|_*}_{A}\delta s + \underbrace{\frac{\partial f}{\partial F}\Big|_*}_{B}\delta F $$

- **Denge noktası** (s\*, F\*): f(s\*, F\*) = 0 olan nokta, yani sistemin orada durabildiği yer. Sabit terim bu yüzden düşer.
- **δs = s − s\***: dengeden **sapma**. Yukarı dengede s\* = 0 olduğu için δs = s.
- **A = ∂f/∂s (Jacobian, 4×4):** satır i, sütun j = "j. durum biraz değişince i. türev ne kadar değişir?"
- **B = ∂f/∂F (4×1):** "kuvvet biraz değişince her türev ne kadar değişir?"

### 6.3 Elle doğrusallaştırma (küçük açı)
θ ≈ 0 ve θ̇ ≈ 0 iken: sinθ ≈ θ, cosθ ≈ 1, sin²θ ≈ 0, θ̇²·(…) ≈ 0 (küçük × küçük).

$$ \ddot x \approx \frac{F - m g\theta}{M} = -\frac{m g}{M}\theta + \frac{1}{M}F $$

$$ \ddot\theta \approx \frac{-F + (M+m) g\theta}{l M} = \frac{(M+m) g}{M l}\theta - \frac{1}{M l}F $$

$$ A = \begin{bmatrix} 0 & 1 & 0 & 0 \\ 0 & 0 & -\frac{m g}{M} & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & \frac{(M+m)g}{M l} & 0 \end{bmatrix}
= \begin{bmatrix} 0 & 1 & 0 & 0 \\ 0 & 0 & -0.981 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 21.58 & 0 \end{bmatrix},
\qquad B = \begin{bmatrix} 0 \\ \frac1M \\ 0 \\ -\frac{1}{M l}\end{bmatrix} = \begin{bmatrix} 0 \\ 1 \\ 0 \\ -2\end{bmatrix} $$

### 6.4 A ve B'yi okumak (en önemli beceri)
| eleman | değer | fiziksel anlamı |
|---|---|---|
| A[0,1] = 1, A[2,3] = 1 | 1 | tanım: konumun türevi hız |
| A[1,2] = −mg/M | −0.981 | sarkaç sağa yatınca araba **sola** itilir. m/M küçük → etkisi zayıf |
| A[3,2] = (M+m)g/(Ml) | **+21.58** | yatma **kendini büyütür**. Artı işaret kararsızlığın kaynağı. l küçüldükçe büyür (kısa sarkaç daha hızlı düşer: kalem vs süpürge) |
| B[1] = 1/M | 1 | itince araba ivmelenir |
| B[3] = −1/(Ml) | **−2** | sağa itince sarkaç **sola** döner. Elinde süpürge dengelerken eli sağa çekince süpürge sola yatar |
| A'nın 1. sütunu sıfır | 0 | dinamikler **x'e bağlı değil**: rayın neresinde olduğun fiziği değiştirmez (öteleme simetrisi) |

**Parametre sezgisi (mülakatta sorulur):** "Daha uzun sarkacı dengelemek kolay mı zor mu?" Daha **kolay**:
A[3,2] ∝ 1/l küçülür, sarkaç daha yavaş düşer. Kalemi parmağında dengeleyemezsin, süpürgeyi dengelersin.

### 6.5 Sistematik yol: Jacobian (sympy)
```python
A = f.jacobian(s).subs(denge)      # ∂f/∂s, denge noktasında
B = f.jacobian([F]).subs(denge)    # ∂f/∂F
```
Küçük açı kuralları ile Jacobian aynı sonucu verir. Jacobian, kuralları sistematik ve hatasız uygulamanın yolu,
özellikle aşağı denge gibi küçük açının "θ ≈ 0" olmadığı noktalarda.

**Aşağı denge (θ\* = π):** cosθ\* = −1 olduğu için A[3,2] = −21.58 ve B[3] = +2. `cartpole.analytic_AB(P, np.pi)`
bunu c = cosθ\* ile genel olarak yazıyor.

### 6.6 Sayısal Jacobian (sonlu farklar), neden ve nasıl
Model karmaşıksa (sürtünme, aerodinamik, bir simülatörün içi) sembolik türev alamayabilirsin. **Merkezi fark**:

$$ \frac{\partial f}{\partial s_i} \approx \frac{f(s^* + \varepsilon e_i) - f(s^* - \varepsilon e_i)}{2\varepsilon} $$

- Hata O(ε²). Tek yönlü farkta O(ε) olurdu.
- **ε seçimi bir takas:** çok büyükse kesme hatası (Taylor'un attığı terimler), çok küçükse yuvarlama hatası
  (iki yakın sayının farkı). Merkezi fark için optimum yaklaşık ε ≈ (makine ε)^{1/3} ≈ 6e-6. Biz 1e-6 kullanıyoruz.
- Sonucumuz: analitik ve sayısal arasındaki fark yukarı dengede 6e-12, aşağı dengede 3e-9.

Bu iki yöntemi karşılaştırmak, sembolik sonucun ve kodun birbirini doğrulaması demek. Bölüm 5'teki "bağımsız yollar" ilkesi.

---

## 7. Özdeğerler ve kararlılık

### 7.1 Tek boyutlu sezgi
ż = a·z → z(t) = z₀·e^{at}. a > 0 ise büyür, a < 0 ise söner, a = 0 ise sabit kalır.

### 7.2 Çok boyutlu: modlar
ṡ = A s'nin çözümü s(t) = e^{At} s₀. A köşegenleştirilebilirse:

$$ s(t) = \sum_i c_i\, e^{\lambda_i t}\, v_i $$

Her **özdeğer λᵢ** bir **mod**un büyüme hızını, her **özvektör vᵢ** o modun **şeklini** (hangi durumların hangi
oranla birlikte hareket ettiğini) verir. Herhangi bir başlangıç bu modların karışımıdır.

| λ | davranış |
|---|---|
| gerçel, > 0 | üstel büyüme (kararsız) |
| gerçel, < 0 | üstel sönüm |
| sanal ±iω | sönümsüz salınım |
| karmaşık σ ± iω | salınım; σ < 0 sönümlü, σ > 0 büyüyen |
| 0 | sabit kalır (ama katlıysa dikkat, 7.4) |

**Kararlılık kuralı (sürekli zaman):** tüm özdeğerlerin gerçel kısmı < 0 ise asimptotik olarak kararlı.
**Tek bir tanesi bile** > 0 ise kararsız.

### 7.3 Yukarı dengede
A'nın θ bloğu (x'ten bağımsız): θ̈ = 21.58 θ. e^{λt} dene:

$$ \lambda^2 = \frac{(M+m) g}{M l} = 21.58 \;\Rightarrow\; \lambda = \pm 4.646\ \text{s}^{-1} $$

**Tüm özdeğerler: 0, 0, +4.646, −4.646 → kararsız.**

- **Zaman sabiti** τ = 1/λ = 0.215 s; **ikiye katlanma** ln2/λ = 0.149 s. Kontrolcünün bu hızla yarışması gerekiyor.
- Sıfır hızla bırakılan sarkaç: θ(t) = θ₀ cosh(λt) = θ₀(e^{λt} + e^{−λt})/2. Sönen mod hemen ölür, büyüyen mod baskın çıkar.
- **Simülasyondan ölçüm** (defter 02): θ₀ = 0.01, log θ'nın eğimi 4.57 (λ = 4.65). Fark nereden? Ölçüm
  aralığının başında cosh henüz saf üstel değil, sonunda ise sinθ < θ olduğu için gerçek sistem biraz daha yavaş.

### 7.4 İki sıfır özdeğer: "çift integratör"
x bloğu (θ = 0, F = 0): ẍ = 0. Bu bir **çift integratör**. λ = 0 iki kez tekrarlanıyor ama A'nın bu bloğu
[[0, 1], [0, 0]] köşegenleştirilemez (Jordan bloğu). Çözüm:

$$ x(t) = x_0 + \dot x_0\, t $$

Yani başlangıç hızı varsa araba **sınırsızca** uzaklaşır: polinom büyüme, üstel değil. Bu yüzden "λ = 0, sorun yok"
demek yanlış. **Tekrarlanan sıfır özdeğer, Lyapunov anlamında kararsızdır.** Doğru anlamı: "rayda arabayı merkeze
geri çeken hiçbir şey yok". Kontrolcünün x'i de dengelemesi gerekiyor.

### 7.5 Aşağı denge
θ̈ = −21.58 θ → λ = ±4.646 **i** → sönümsüz salınım. Frekans ω = 4.646 rad/s, periyot 2π/ω = 1.35 s.

**Neden sabit pivotlu sarkaçtan (√(g/l) = 4.43 rad/s, periyot 1.42 s) hızlı?** Araba serbest olduğu için sarkaç
salınırken araba da ters yöne kayıyor. Mafsal sabit olmadığından etkin atalet azalıyor. M → ∞ olunca (araba
kıpırdamaz) √((M+m)g/(Ml)) → √(g/l) olur ve sabit pivota döner. Güzel bir limit kontrolü.

### 7.6 Mülakatta
*"Yukarı dengede A'nın özdeğerleri 0, 0, ±4.65. Pozitif olan sarkacın düşme modu: açı yaklaşık 0.15 s'de ikiye
katlanıyor, bu da kontrol bant genişliği ve örnekleme hızı için alt sınır koyuyor. İki sıfır özdeğer arabanın çift
integratör olmasından geliyor: geri getirici kuvvet olmadığı için başlangıç hızı varsa araba sınırsız kayar. Bunu
simülasyonda log-ölçekli θ(t)'nin eğiminden de ölçtüm, 4.57 çıktı."*

---

## 8. Kontrol edilebilirlik

### 8.1 Soru
"Elimde **tek bir kuvvet** var, ama **dört durum** var (ve **iki** serbestlik derecesi, yani sistem *az eyleyicili*,
underactuated). Bu tek kuvvetle dört durumu da istediğim yere götürebilir miyim?"

**Tanım:** Herhangi bir başlangıç durumundan herhangi bir son duruma sonlu sürede götüren bir F(t) varsa sistem
kontrol edilebilirdir.

### 8.2 Test: 𝒞 = [B, AB, A²B, A³B], rank = n mi?

**Sezgi: kuvvetin etkisi zincir boyunca yayılıyor.**
- **B:** F'nin *doğrudan* etkilediği yönler → ẍ ve θ̈ (hızlar).
- **AB:** bir türev adımı sonra → hızlar konumlara aktı (x, θ).
- **A²B, A³B:** konumlar hızları, hızlar konumları etkiliyor (bağlaşım üzerinden).

Bu dört vektör 4 boyutlu uzayın tamamını kaplıyorsa (rank 4), kuvvet her yöne "ulaşabiliyor" demektir.
Neden Aⁿ'de durulduğunu Cayley–Hamilton söylüyor: A⁴ zaten A⁰…A³'ün bir kombinasyonu, yeni bir yön eklemez.

$$ \mathcal C = \begin{bmatrix} 0 & \frac1M & 0 & \frac{m g}{M^2 l} \\ \frac1M & 0 & \frac{m g}{M^2 l} & 0 \\ 0 & -\frac{1}{M l} & 0 & -\frac{(M+m)g}{M^2 l^2} \\ -\frac{1}{M l} & 0 & -\frac{(M+m)g}{M^2 l^2} & 0 \end{bmatrix},
\qquad \det\mathcal C = \frac{g^2}{M^4 l^4} \neq 0 $$

**Sonuç:** rank 4 → kontrol edilebilir. Determinant g ≠ 0 olduğu sürece hiçbir M, m, l değerinde sıfır olmuyor.

**İlginç sınır durumu, g = 0 (uzay):** det = 0, yani kontrol edilebilirlik kaybolur. Neden? Yerçekimi yoksa ẍ = F/M
ve θ̈ = −F/(Ml), dolayısıyla ẍ + l θ̈ = 0: sarkaç ucunun yatay konumu (x + lθ) kuvvetten hiç etkilenmiyor. Kütlesiz
çubuk dikken kuvveti sadece kendi doğrultusunda (dikey) iletebilir, ucu yana itemez. Yerçekimi varken uç, açıya
bağlı bir yatay kuvvet hissediyor ve bağlaşım bu kanaldan kuruluyor.

### 8.3 Rank ikilidir, "ne kadar zor" ise koşul sayısı
Rank sadece evet/hayır der. **Koşul sayısı** cond(𝒞) = 48: bazı yönlere ulaşmak diğerlerinden çok daha fazla kuvvet
gerektiriyor ama makul. Koşul sayısı 1e10 olsaydı kağıt üstünde kontrol edilebilir, pratikte değil derdik.
Daha sağlam ölçüt: kontrol edilebilirlik **Gramian**'ı (enerji maliyeti). LQR'de dolaylı olarak karşına çıkacak.

### 8.4 Alternatif test: PBH
Her özdeğer λ için rank[A − λI, B] = n mi? Kontrol edilemeyen bir mod varsa o modun sol özvektörü B'ye diktir:
kuvvet o modu hiç "görmez". Özellikle **hangi modun** kontrol edilemediğini bulmak için yararlı.

### 8.5 Neden önemli?
- Kontrol edilebilir ise **durum geri beslemesi u = −Ks ile kapalı çevrim özdeğerlerini istediğin yere koyabilirsin**
  (kutup yerleştirme teoremi). LQR de bunun "optimal" versiyonu.
- Az eyleyicili (1 girdi, 2 DOF) ama **yerel olarak** tam kontrol edilebilir. Bunu mümkün kılan, Bölüm 3.5'teki
  bağlaşım terimi.
- ⚠ Bu **doğrusal** bir sonuç: dengenin yakınında geçerli. Kuvvet sınırı (|F| ≤ 10 N) ve büyük açılar ayrı konu
  (Aşama 4: çekim bölgesi).

---

## 9. Gözlemlenebilirlik

### 9.1 Soru
"Sadece bazı durumları **ölçebiliyorsam** (y = C s), ölçümlerin zaman içindeki geçmişinden bütün durumu
çıkarabilir miyim?" Kontrol edilebilirliğin **ikizi (dual)**.

### 9.2 Test: 𝒪 = [C; CA; CA²; CA³], rank = n mi?
Sezgi: y'yi ve türevlerini (ẏ = CAs, ÿ = CA²s, …) biliyorsan, bunlar s üzerine kaç bağımsız denklem verir?

### 9.3 Sonuçlarımız
| ölçülen | rank 𝒪 | yorum |
|---|---|---|
| sadece x | **4** | x'in hareketinden sarkacın açısı da çıkarılabilir (A[1,2] ≠ 0: θ, ẍ'yi etkiliyor) |
| sadece θ | **2** | x ve ẋ **görünmez**: A'nın θ satırları x'e bağlı değil. Sabit hızla kayan araba θ'da hiç iz bırakmaz |
| x ve θ | 4 | (gerçek kurulumlar: enkoder + açı sensörü) |

**Aşama 5'e etkisi:** Kalman filtresi ancak gözlemlenebilir durumları kestirebilir. Sadece açı sensörüyle arabanın
konumunu kestiremezsin. En az bir konum ölçümü gerekiyor.

**Bölüm 10'la bağlantı:** G_θ(s)'de x modları (s² kutbu) sadeleşip gidiyor. Gözlemlenemeyen (ya da kontrol
edilemeyen) mod transfer fonksiyonunda **kutup-sıfır sadeleşmesi** olarak görünür.

---

## 10. Transfer fonksiyonu, sıfırlar ve ters tepki

### 10.1 Transfer fonksiyonu
Laplace dönüşümüyle (sıfır başlangıç koşulu) türev s ile çarpmaya dönüşür. ṡ = As + BF, y = Cs ise:

$$ G(s) = \frac{Y(s)}{F(s)} = C(sI - A)^{-1}B $$

- **Kutuplar:** paydanın kökleri = A'nın özdeğerleri (sadeleşme yoksa).
- **Sıfırlar:** payın kökleri. G(z) = 0 olan s değerleri: o "frekansta" giriş çıkışa geçmiyor.

### 10.2 Bizim sistem
$$ G_x(s) = \frac{X(s)}{F(s)} = \frac{l s^2 - g}{s^2\,\big(M l s^2 - (M+m) g\big)} \qquad
G_\theta(s) = \frac{\Theta(s)}{F(s)} = \frac{-1}{M l s^2 - (M+m) g} $$

- G_x'in **kutupları:** 0, 0, ±4.646. **Sıfırları:** ls² − g = 0 → **s = ±√(g/l) = ±4.43**.
- G_θ'nın sonlu sıfırı yok; x modları (s²) sadeleşmiş (Bölüm 9).

**s = +4.43 sağ yarı düzlemde (RHP) bir sıfır. Böyle bir sisteme "minimum fazlı olmayan" (non-minimum phase) denir.**

**Sıfırın fiziksel anlamı:** √(g/l), mafsalı **sabit** bir sarkacın doğal frekansı. Arabayı "x hiç hareket etmeyecek"
şekilde zorlarsan geriye mafsalı sabit bir sarkaç kalır. Genel kural: sıfırlar, çıkışı sıfırda **tutmaya
zorladığında** geriye kalan iç dinamiğin (sıfır dinamiği) kutuplarıdır. Bizde bu iç dinamik **kararsız** (tepede
duran sabit pivotlu sarkaç), dolayısıyla sıfır RHP'de.

### 10.3 Ters tepki (inverse response)
RHP sıfırı olan sistemin basamak cevabı **önce ters yöne** gider.

**Fiziksel hikâye:** Dengedeki arabayı 1 m sağa götürmek istiyorsun. Arabayı doğrudan sağa itersen sarkaç sola
düşer (B[3] < 0). Doğru sıra şöyle:
1. Önce arabayı **sola** çek → sarkaç **sağa** yatar.
2. Sağa yatık sarkacı "yakalamak" için arabayı **sağa** sür → araba sağa gider, sarkaç dikleşir.

Süpürgeyi elinde dengelerken sağa yürümek istediğinde elini bir an sola çekersin. Aynı şey.

**Deney (defter 02, bölüm 5):** kutup yerleştirmeyle dengelenmiş sisteme x_ref = 1 m:

| kontrolcü | en küçük x | max \|F\| | max \|θ\| |
|---|---|---|---|
| yavaş (kutuplar −2…−3.5) | −0.03 m | 2.7 N | 0.09 rad |
| hızlı (kutuplar −4…−7) | **−0.16 m** | **42.8 N** (F_max = 10'u aşar!) | 0.38 rad |

**Hızlandıkça ters sapma ve kuvvet büyüyor.** RHP sıfırı hiçbir kontrolcünün yok edemeyeceği **temel bir sınır**.

### 10.4 Neden kontrolü zorlaştırır? (derin cevap)
1. **Tersine çevrilemez:** RHP sıfırını bir kontrolcü kutbuyla sadeleştirmeye kalkarsan RHP'de bir kontrolcü kutbu
   olur, yani iç kararsızlık. Bu yasak.
2. **Bant genişliği sınırı:** kaba kural, RHP sıfırı z kapalı çevrim bant genişliğini ≲ z/2 ile sınırlar. Kararsız
   kutup p ise bant genişliğinin ≳ 2p olmasını **gerektirir**.
3. **Bizde ikisi çakışıyor:** z/p = √(M/(M+m)) = 0.95 < 1, yani sıfır kutuptan **önce** geliyor. "Hızlı ol" ve "yavaş
   ol" talepleri çelişiyor. Teorik sonuç: **sadece x ölçülüp tek döngüyle** kontrol edilseydi, *her* kararlı kılan
   kontrolcü için duyarlılık tepesi ‖S‖∞ ≥ |z+p|/|z−p| ≈ **42** olurdu (Skogestad & Postlethwaite), yani berbat sağlamlık.
4. **Pratik sonuç:** θ'yı da ölçmek gerekiyor. G_θ'da RHP sıfırı yok. Hızlı bir **iç döngü θ'yı**, yavaş bir **dış
   döngü x'i** kontrol eder (Aşama 3, kaskad PID). Ya da tam durum geri beslemesi kullanılır (LQR). Ters tepki yine de
   kalır ama sağlamlık kurtarılır.

**Mülakatta:** *"F'den x'e transfer fonksiyonunda +√(g/l)'de bir sıfır var. Fiziksel olarak, mafsalı sabitlenmiş
sarkacın kararsız dinamiği. Sonucu ters tepki: arabayı sağa götürmek için önce sola gitmek gerekiyor. Bu, ulaşılabilir
bant genişliğini sınırlıyor ve bu sistemde sıfır kararsız kutuptan bile yavaş, o yüzden x'i tek başına ölçüp
kontrol etmek pratikte imkânsız. Bu yüzden açıyı iç döngüde kontrol eden kaskad yapı ya da tam durum geri beslemesi
kullanılıyor."*

### 10.5 Sayısal tuzak
`control.ss2tf`, payın başına ~1e-16 büyüklüğünde bir yuvarlama artığı bırakıyor: s³ katsayısı 2e-15. `control.zeros`
bunu gerçek sanıp **−4.7e14'te sahte bir sıfır** döndürüyor. `cartpole.transfer_zeros` göreli olarak ihmal edilebilir
baş katsayıları atıyor. Ders: **sayısal sonucu her zaman sembolik ya da fiziksel beklentiyle karşılaştır.**

---

## 11. Ayrıklaştırma

### 11.1 Neden?
Kontrolcün bilgisayarda çalışıyor: her dt = 0.02 s'de bir ölçüm alıyor, bir kuvvet hesaplıyor, kuvveti dt boyunca
**sabit tutuyor** (ZOH). Kontrolcünün "gördüğü" sistem bu yüzden ayrık zamanlı:

$$ s_{k+1} = A_d\, s_k + B_d\, F_k $$

LQR (dlqr), Kalman ve MPC'yi bu biçimde kuracağız.

### 11.2 Tam çözüm (ZOH)
Bir aralıkta F sabitken ṡ = As + BF'nin kesin çözümü:

$$ s(t_k + dt) = e^{A\,dt}\, s(t_k) + \left(\int_0^{dt} e^{A\tau}\,d\tau\right) B\, F_k $$

$$ A_d = e^{A\,dt}, \qquad B_d = \int_0^{dt} e^{A\tau}d\tau\; B $$

- **e^{A dt} (matris üsteli):** ż = az'nin çözümü e^{at} ise, sistemin çözümü e^{At}. "Kimse dokunmazsa dt sonra durum ne olur?"
- **B_d:** dt boyunca sabit kuvvetin birikmiş etkisi.

**Hesaplama hilesi (Van Loan):** İntegrali ayrıca hesaplamadan tek bir matris üsteliyle:

$$ \exp\!\left(\begin{bmatrix} A & B \\ 0 & 0 \end{bmatrix} dt\right) = \begin{bmatrix} A_d & B_d \\ 0 & I \end{bmatrix} $$

`cartpole.discretize` bunu yapıyor; `control.c2d(..., 'zoh')` ile farkı tam 0.

**Bu ayrıklaştırma bir yaklaşım değil.** Doğrusal sistem ve ZOH girişi için **kesin**: ayrık model, sürekli
modelin t = k·dt anlarındaki çözümüyle aynı (testte fark 5e-11).

### 11.3 Euler ayrıklaştırması ile farkı
Euler: A_d ≈ I + A dt, B_d ≈ B dt. Bu, e^{A dt}'nin Taylor açılımının ilk iki terimi. Bizim değerler:

$$ A_d = \begin{bmatrix} 1 & 0.02 & -0.0002 & 0 \\ 0 & 1 & -0.01965 & -0.0002 \\ 0 & 0 & 1.00432 & 0.02003 \\ 0 & 0 & 0.4323 & 1.00432 \end{bmatrix},
\quad B_d = \begin{bmatrix} 0.0002 \\ 0.02 \\ -0.0004 \\ -0.0401 \end{bmatrix} $$

Euler'de A_d[2,2] = 1, oysa gerçekte 1.00432; A_d[3,2] = 0.4316, gerçekte 0.4323. Küçük farklar gibi görünüyor ama
kararsız bir sistemde birikiyor. Euler ayrıklaştırması ayrıca B_d[0] ve B_d[2]'yi **sıfır** verir: kuvvetin bir adım
içinde konuma etkisini kaçırır.

### 11.4 Özdeğer eşlemesi: z = e^{λ dt}
Sürekli zamandaki λ, ayrık zamanda z = e^{λ dt}'ye gider:

| sürekli | ayrık | kararlılık |
|---|---|---|
| Re λ < 0 (sol yarı düzlem) | \|z\| < 1 (birim çemberin içi) | kararlı |
| Re λ = 0 | \|z\| = 1 | sınırda |
| Re λ > 0 | \|z\| > 1 | kararsız |

Bizde: λ = +4.646 → z = e^{4.646·0.02} = **1.0974**. "Her adımda sapma %9.7 büyüyor." λ = 0 → z = 1.
Ölçülen |z| değerleri ile e^{λ dt} birebir aynı.

---

## 12. Doğrusal model nerede bozuluyor?

Deney (defter 02, bölüm 9): F = 0, aynı θ₀'dan doğrusal ve doğrusal olmayan model, 1 s. "Ayrışma" θ'da %10 göreli hata.

| θ₀ | ayrışma anı | o anki gerçek θ | θ(1 s) gerçek | θ(1 s) doğrusal |
|---|---|---|---|---|
| 0.05 rad (3°) | 0.91 s | 1.56 rad (90°) | 2.20 | 2.60 |
| 0.20 rad (11°) | 0.60 s | 1.50 rad (86°) | 4.67 | 10.41 |
| 0.50 rad (29°) | 0.38 s | 1.34 rad (77°) | 5.46 | 26.03 |

**Yorum:**
1. Belirleyici olan başlangıç açısı değil, **o anki açı**. Kararsız sistem her başlangıcı büyük açılara taşıyor ve
   doğrusal model ~75–90°'de kopuyor.
2. Doğrusal modelde θ **sınırsız** büyüyor (26 rad = 4 tur!). Gerçek sarkaç ise aşağı düşüp salınıyor. Doğrusal
   model "düşmenin başlangıcını" iyi anlatıyor, "düştükten sonrasını" hiç anlatamıyor.
3. **x çok daha erken ayrışıyor:** ẍ'deki sinθ cosθ ve θ̇² terimleri doğrusal modelde yok.
4. **Kontrol açısından:** dengeleyen bir kontrolcü sarkacı zaten küçük açılarda tutuyor, orada doğrusal model çok iyi
   (sinθ ≈ θ hatası 10°'de %0.5). Sarkaç aşağıdan başlarsa doğrusal kontrolcü yetmez → Aşama 11 (enerji ile kaldırma)
   ve RL burada devreye giriyor. **Projenin ana sorusunun matematiksel temeli bu bölüm.**

---

# EKLER

## 13. Mülakat soruları ve örnek cevaplar

**S1. Hareket denklemlerini nasıl türettin, neden Lagrange?**
Genelleştirilmiş koordinatlar q = [x, θ] seçtim, kinetik ve potansiyel enerjiyi yazdım, L = T − V üzerinden
Euler–Lagrange denklemlerini sympy ile kurdum. Lagrange'ın avantajı, araba ile sarkaç arasındaki çubuk kuvveti gibi
kısıt kuvvetlerini hiç hesaba katmaması. Sonuç M(q)q̈ = b biçiminde; kütle matrisi her açıda tersinir olduğu için
ivmeler tek türlü bulunuyor.

**S2. Modelinin doğruluğunu nasıl kanıtladın?**
Bağımsız yollardan: (1) sympy türetmesini elle yazdığım kodla 100 rastgele durumda karşılaştırdım (fark 1e-14);
(2) sürtünmesiz sistemde enerji korunmalı, RK4 ile kayma 1e-11; (3) denge noktalarında türev sıfır; (4) işaret
testleri: F > 0 → araba sağa, sarkaç sola; (5) dt yakınsaması. Her test farklı bir hata türünü yakalıyor. Örneğin
enerji testi F'nin işaret hatasını göremez, onu işaret testi yakalar.

**S3. Neden RK4? dt'yi nasıl seçtin?**
Euler sönümsüz sistemde enerji pompalıyor (dt = 0.02'de 10 s'de %220). RK4'ün global hatası O(dt⁴). Kontrol
periyodu 0.02 s: kararsız mod 0.15 s'de ikiye katlanıyor, bu süre içinde ~7 müdahale ediyoruz. Fiziği bu periyodun
içinde 20 alt adımla entegre ediyorum, çünkü sarkaç alttan hızlı geçerken tek adımlık RK4 hatası 1e-6 hedefini
aşıyordu. Kontrol periyodu ile entegrasyon adımı ayrı kavramlar.

**S4. Doğrusallaştırma nedir, ne zaman geçerli?**
Denge noktası etrafında birinci mertebe Taylor açılımı: A = ∂f/∂s, B = ∂f/∂F. Geçerliliği yaklaşım hatasına
bağlı: sinθ ≈ θ 10°'de %0.5, 30°'de %5 hata. Deneyde doğrusal model θ ≈ 75–90°'de ayrıştı; x ise daha erken.
Dengeleme için yeterli, kaldırma (swing-up) için değil.

**S5. A matrisinin pozitif özdeğeri fiziksel olarak ne demek?**
Sarkacın düşme modu: λ = √((M+m)g/(Ml)) = 4.65 s⁻¹. Açı ~0.15 s'de ikiye katlanıyor. Kontrolcünün tepki hızı ve
örnekleme frekansı için alt sınır. Simülasyonda log θ'nın eğiminden ölçtüm: 4.57.

**S6. Sıfır özdeğerler sorun mu?**
İki tane var ve Jordan bloğu oluşturuyorlar: araba bir çift integratör. Başlangıç hızı varsa x zamanla doğrusal
büyür; Lyapunov anlamında kararsız. Rayda geri getirici kuvvet olmadığı için kontrolcü x'i de aktif olarak tutmalı.

**S7. Sistem kontrol edilebilir mi, nasıl bildin? Az eyleyicili olmasına rağmen nasıl?**
𝒞 = [B, AB, A²B, A³B] rank 4, determinantı g²/(M⁴l⁴) ≠ 0. Tek girdi iki serbestlik derecesini kontrol edebiliyor,
çünkü kinetik enerjideki m l ẋ θ̇ cosθ bağlaşım terimi kuvvetin etkisini sarkaca taşıyor. Bu doğrusal ve yerel bir
sonuç; kuvvet sınırı ve büyük açılar ayrıca incelenmeli.

**S8. Minimum fazlı olmayan sistem nedir? Bu sistemde neden var, ne yapıyorsun?**
Transfer fonksiyonunda sağ yarı düzlemde sıfır olan sistem; basamak cevabı önce ters yöne gider. Bizde F→x'te
+√(g/l)'de: mafsalı sabitlenmiş sarkacın kararsız dinamiği. Arabayı sağa götürmek için önce sola çekip sarkacı sağa
yatırmak gerekiyor. Ulaşılabilir hızı temelden sınırlıyor ve burada sıfır kararsız kutuptan yavaş olduğu için sadece
x ölçen tek döngülü bir kontrolcü çok kötü sağlamlığa mahkûm. Çözüm: θ'yı da ölçmek, kaskad ya da tam durum geri beslemesi.

**S9. Gözlemlenebilirlik sonucun ne, Kalman'ı nasıl etkiler?**
Sadece x ölçülürse rank 4, sadece θ ölçülürse rank 2: θ dinamiği x'e bağlı olmadığı için arabanın konumu ve hızı
açıda iz bırakmıyor. Kalman filtresi için en az bir konum ölçümü gerekli.

**S10. Neden ayrıklaştırıyorsun, Euler ayrıklaştırmasından farkı ne?**
Kontrolcü dijital: kuvveti dt boyunca sabit tutuyor (ZOH). Bu varsayım altında A_d = e^{A dt}, B_d = ∫e^{Aτ}dτ B
kesin sonuç; Van Loan'ın blok matris üsteli hilesiyle hesaplıyorum. Euler (I + A dt) birinci mertebe yaklaşım, kararsız
modu az tahmin ediyor ve kuvvetin bir adımdaki konum etkisini kaçırıyor. Ayrık özdeğerler z = e^{λ dt}: kararsız mod
|z| = 1.097.

**S11. Gymnasium'daki CartPole'dan farkın ne?**
Gymnasium dağılmış kütleli çubuk (I = ml²/3), ayrık aksiyon (±10 N) ve Euler entegrasyonu kullanıyor. Benimki
noktasal kütle, sürekli kuvvet, RK4 ve kendi türettiğim, test edilmiş bir model. Sürekli aksiyon, klasik
kontrolcülerle (LQR, MPC) adil karşılaştırma için gerekli.

**S12. Modeli gerçeğe yaklaştırmak için ne eklersin?**
Ray ve mafsal sürtünmesi (Q_x = F − b_x ẋ, Q_θ = −b_θ θ̇), motor dinamiği ve gecikme, kuvvet doyması, sensör
gürültüsü ve nicemleme, dağılmış kütleli çubuk. Ve parametre belirsizliği: Aşama 12'deki domain randomization tam
olarak bunu ele alıyor.

---

## 14. Tuzaklar

| tuzak | belirtisi | önlem |
|---|---|---|
| Açı kuralı karışıklığı (yukarı / aşağı) | Kontrolcü sarkacı daha hızlı düşürüyor | Tek kural; işaret testleri |
| `sp.symbols('x')` ile zamana bağlı değişken | Türevler 0 çıkıyor | `sp.Function('x')(t)` |
| `subs` sırası | ẍ yerine ẋ'nin türevi kalıyor | önce 2. türevler |
| `0.5` yerine `sp.Rational(1,2)` kullanmamak | Sadeleşme bozuk, ondalıklı ifadeler | `Rational` |
| Euler ile uzun simülasyon | Sürtünmesiz sarkaç kendiliğinden hızlanıyor | RK4 / enerji testi |
| Kontrol periyodu = entegrasyon adımı sanmak | Hızlı hareketlerde fizik hatası | `substeps` |
| `ss2tf` + `zeros` sonucuna körü körüne güvenmek | 1e14'te sahte sıfır | sembolik sonuçla karşılaştır |
| "λ = 0 kararlıdır" | Araba sınırsız kayıyor | tekrarlanan 0 = Jordan bloğu |
| Doğrusal modeli büyük açıda kullanmak | Tahmin 4 tur dönüyor | geçerlilik deneyi |
| Kuvvet sınırını unutmak | Kâğıtta hızlı kontrolcü 43 N istiyor | tüm kontrolcülerde F_max = 10 N |

---

## 15. Kaynak haritası

| konu | kaynak | not |
|---|---|---|
| Lagrange tarifi | Tedrake, *Underactuated Robotics*, Appendix "Multi-Body Dynamics" → "Deriving the equations of motion" | M(q)q̈ + C q̇ + G = Bu biçimi de orada |
| Cartpole denklemleri | Tedrake, "Acrobots, Cart-Poles, and Quadrotors" → "The Cart-Pole System" | ⚠ θ = 0 aşağı |
| Lagrange (fizik açısı) | Herhangi bir analitik mekanik kitabı (Goldstein ya da Taylor, *Classical Mechanics*), Lagrange bölümleri | Noether, döngüsel koordinatlar |
| RK4, hata mertebeleri | Wikipedia "Runge–Kutta methods" → RK4; "Euler method" → stability | |
| Durum uzayı, özdeğerler | Brunton, *Control Bootcamp*: "Linear Systems", "Stability and Eigenvalues" | |
| Doğrusallaştırma | Brunton: "Linearizing Around a Fixed Point"; Åström & Murray, *Feedback Systems*, "Linear Systems" | |
| Kontrol edilebilirlik / gözlemlenebilirlik | Brunton: "Controllability", "Observability"; Åström & Murray, "State Feedback" başı | PBH, Gramian |
| Transfer fonksiyonu, RHP sıfırı | Åström & Murray, "Transfer Functions" ve "Fundamental Limits" bölümleri | z, p sınırları |
| RHP sıfır/kutup sınırları (ileri) | Skogestad & Postlethwaite, *Multivariable Feedback Control*, Bölüm 5 (performans sınırları) | ‖S‖∞ ≥ \|z+p\|/\|z−p\| |
| Ayrıklaştırma | Åström & Murray "Linear Systems"; python-control `c2d` dokümanı | Van Loan |
| Kendi ders notların | Modern Kontrol, ilgili haftalar | |

---

## 16. Kendini sına

Cevapları yukarıda var ama **önce kendin cevapla**, sonra kontrol et. (Aşamanın "🎯 Anladım ölçütü"nü sen yazacaksın;
bunlar sadece alıştırma.)

1. Sarkaç kütlesi m iki katına çıkarsa A[1,2] ve A[3,2] nasıl değişir? Sezgisel olarak açıkla.
2. Neden ∂L/∂x = 0 yatay momentumun korunduğunu söylüyor? F ≠ 0 iken ne olur?
3. Euler'in enerji pompaladığını tek satırlık bir hesapla göster.
4. dt = 0.02'yi 0.1'e çıkarsak ne bozulur? İki ayrı sebep söyle (fizik doğruluğu ve kontrol).
5. Sadece θ ölçülürse hangi durumlar kestirilemez, neden? Bunun G_θ(s)'deki karşılığı ne?
6. RHP sıfırının yerini değiştirmek için hangi parametreyi değiştirirsin? z/p oranı neden her zaman < 1?
7. Aşağı dengede özdeğerler neden sanal? Sürtünme eklesek nasıl değişirdi?
8. Ayrık özdeğer z = 1.097 ne demek? 10 adım (0.2 s) sonra küçük bir sapma kaç katına çıkar?
9. Bir arkadaşın "λ = 0 olduğu için araba konumu kararlı" diyor. Neden yanlış?
10. Doğrusal ve doğrusal olmayan model θ₀ = 0.05'te bile 1 s sonra ayrışıyor. Bu, LQR'nin işe yaramayacağı anlamına mı gelir? Neden değil?
