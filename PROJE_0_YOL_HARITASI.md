# Proje 0 — Ters Sarkaç (Cartpole) + Sarkaç Kaldırma: Klasik Kontrol ile RL

> Bu belge Proje 0'ı baştan sona yürütmek için sıralı bir yol haritasıdır. Her aşamada şu blokları bulacaksın:
> **💡 Sade dille** (kavram + benzetme) → **👀 Önce simüle et** (formülden önce gözle gör) → **🎤 Mülakatta nasıl sorulur**
> → **📚 Öğren** (oku / izle) → **🧮 Bilgisayarla çöz** → **💻 Kodla** → **✅ Doğrula** → **📦 Çıktı** → **🎯 Anladım ölçütü** (sen dolduracaksın).
> Aşamaları sırayla yap; bir aşamanın doğrulaması geçmeden sonrakine geçme.
>
> **Okuma sırası:** Önce "Sade dille"yi oku ve "Önce simüle et"i yap. Kavram oturmadan formüle ve koda geçme.
> Formül, simülasyonda gördüğün şeyin tarifidir. "Mülakatta" bölümündeki soruları kendi cümlelerinle cevaplayabiliyorsan
> aşamanın özünü almışsın demektir. *(isteğe bağlı)* işaretli maddeler mekanik ayrıntıdır; atlayabilirsin.
>
> **Çalışma ilkesi:** Türetmeler ve analizler kâğıtta değil **bilgisayarda** yapılır: sembolik işler `sympy` ile
> (`notebooks/` altında), sayısal analiz `numpy` / `scipy` / `python-control` ile, teori soruları küçük sayısal
> deneylerle. En sondaki **Ek A**'daki referans denklemler, kendi sympy sonucunu kontrol etmek içindir.
>
> **Hedef süre: ~2,5 hafta (12–13 iş günü).** Gün planı en altta; kendi hızına göre kaydır.

---

## İçindekiler

- [Genel bakış](#genel-bakış)
- [Aşama 0 — Kurulum ve kurallar](#aşama-0--kurulum-ve-kurallar)
- [Aşama 1 — Cartpole modelleme (Lagrange + RK4)](#aşama-1--cartpole-modelleme-lagrange--rk4)
- [Aşama 2 — Doğrusallaştırma ve sistem analizi](#aşama-2--doğrusallaştırma-ve-sistem-analizi)
- [Aşama 3 — PID](#aşama-3--pid)
- [Aşama 4 — LQR](#aşama-4--lqr)
- [Aşama 4b — MPC (model öngörülü kontrol)](#aşama-4b--mpc-model-öngörülü-kontrol)
- [Aşama 5 — Kalman filtresi ve LQG](#aşama-5--kalman-filtresi-ve-lqg)
- [Aşama 6 — Ortak değerlendirme düzeneği](#aşama-6--ortak-değerlendirme-düzeneği)
- [Aşama 7 — Gymnasium ortamı](#aşama-7--gymnasium-ortamı)
- [Aşama 8 — RL teorisi: policy gradient, PPO, SAC](#aşama-8--rl-teorisi-policy-gradient-ppo-sac)
- [Aşama 9 — RL eğitimi: PPO ve SAC ile cartpole](#aşama-9--rl-eğitimi-ppo-ve-sac-ile-cartpole)
- [Aşama 10 — (İsteğe bağlı) PPO'yu elle yaz](#aşama-10--isteğe-bağlı-ppoyu-elle-yaz)
- [Aşama 11 — Sarkaç kaldırma: enerji + LQR hibrit, SAC](#aşama-11--sarkaç-kaldırma-enerji--lqr-hibrit-sac)
- [Aşama 12 — Sağlamlık ve domain randomization](#aşama-12--sağlamlık-ve-domain-randomization)
- [Aşama 13 — Sonuç raporu](#aşama-13--sonuç-raporu)
- [Gün planı](#gün-planı)
- [Ek A — Referans denklemler (sympy sonucunu kontrol için)](#ek-a--referans-denklemler-sympy-sonucunu-kontrol-için)
- [Ek B — Parametreler ve kurallar](#ek-b--parametreler-ve-kurallar)
- [Ek C — Kaynak listesi](#ek-c--kaynak-listesi)
- [Ek D — Bilinen tuzaklar](#ek-d--bilinen-tuzaklar)

---

## Genel bakış

### Proje hangi soruları cevaplıyor?
1. **Klasik kontrol nerede iyi?** Denge noktası yakınında PID ve LQR.
2. **Nerede yetersiz?** LQR doğrusallaştırılmış modele dayanır → yalnızca denge yakınında geçerli. Doyma (kuvvet sınırı)
   çekim bölgesini daraltır. Sarkaç aşağıdan başlarsa LQR tek başına kaldıramaz.
3. **RL bu boşluğu doldurur mu, bedeli nedir?** Örnek verimliliği (kaç adım), kararlılık, kontrol eforu.
4. **Model yanlışken kim daha sağlam?** LQR mi, nominal RL mi, domain randomization ile eğitilmiş RL mi?

### İki ortam
| ortam | görev | klasik çözüm | RL |
|---|---|---|---|
| **A. Cartpole (sürekli kuvvet)** | Yukarıdaki sarkacı dengede tut, arabayı merkezde tut | PID (kaskad), LQR, LQG | PPO, SAC |
| **B. Basit sarkaç kaldırma** | Aşağıda asılı sarkacı yukarı kaldır ve dengele (sınırlı tork) | Enerji tabanlı kaldırma + LQR'ye geçiş | SAC |

> ⚠ Gymnasium'daki `CartPole-v1` **ayrık aksiyonlu** ve modeli farklı (dağılmış kütleli çubuk). Ortamı kendin yazacaksın.

### Proje sonunda elinde olacaklar
- Kendi türettiğin dinamik modeli + testleri
- PID, LQR, MPC, LQG, enerji tabanlı kaldırma denetleyicileri
- İki Gymnasium ortamı
- PPO ve SAC eğitimleri (çok tohumlu)
- Ortak değerlendirme düzeneği ve sağlamlık tablosu
- Sonuç raporu (İngilizce README)

---

## Aşama 0 — Kurulum ve kurallar

**Süre:** 1–2 saat (Gün 1)

### 💻 Yap
1. Python 3.10–3.12 ile sanal ortam:
   ```bash
   python -m venv .venv
   ```
   ```bash
   .venv\Scripts\activate
   ```
2. Paketler:
   ```bash
   pip install numpy scipy sympy matplotlib control cvxpy gymnasium stable-baselines3 tensorboard pytest
   ```
3. `git init`, `.gitignore` (`.venv/`, `runs/`, `*.zip`).
4. **Sade yapı — proje boyunca yalnız bunlar:**
   ```
   Proje 0 - Pendilums/
     KARARLAR.md       kararlar ve tahminler
     cartpole.py       parametreler, dinamik f, enerji, RK4, doğrusallaştırma (A, B), ayrıklaştırma, animasyon
     pendulum.py       basit sarkaç dinamiği (Aşama 11; RK4'ü cartpole.py'den import eder)
     controllers.py    PID, LQR, Kalman/LQG, MPC, enerji ile kaldırma — hepsi aynı arayüzle
     envs.py           iki Gymnasium ortamı
     evaluate.py       senaryolar + ölçütler + karşılaştırma tablosu
     train.py          PPO / SAC eğitimi
     test_all.py       bütün doğrulama testleri (pytest)
     notebooks/        her aşamaya bir defter: sympy, analiz, grafikler ve yorumların
     runs/             yalnız RL modelleri ve logları (otomatik oluşur, git'e girmez)
   ```
   **Tek kural:** başka bir aşamanın da kullanacağı kod `.py` dosyasına gider; bir kerelik hesap, grafik ve yorum
   o aşamanın defterine. Grafikler defterin içinde kalır, ayrı bir `results/` klasörü yok.

   > Mevcut dosyalarını taşımak için: `dynamics/cartpole.py` + `dynamics/integrators.py` + `analysis/linearize.py`
   > → `cartpole.py`; `dynamics/cartpole_sym.py` → `notebooks/01_...` defterine (sembolik türetme bir kerelik iş);
   > `analysis/stage1_plots.py`, `stage2_plots.py` → ilgili defterlere; `analysis/animate.py` → `cartpole.py` içinde
   > bir fonksiyon; `tests/test_dynamics.py` + `tests/test_linearize.py` → `test_all.py`.
5. Bir `KARARLAR.md` aç: her önemli karar (açı kuralı, parametreler, ödül, ölçütler) tarihiyle buraya yazılır.
   BVR'daki `REQUIREMENTS.md` alışkanlığı.

### 📌 Projenin değişmez kuralları (Ek B'de ayrıntı)
- **Açı kuralı:** θ = 0 **yukarı** (dik), θ > 0 sağa yatık. x > 0 sağ, F > 0 sağa itiş.
- **Tüm denetleyiciler aynı kuvvet sınırını** (|F| ≤ F_max) ve aynı kontrol periyodunu (dt) kullanır.
- **Tahmin önce, koşu sonra:** her deneyden önce `KARARLAR.md`'ye ne beklediğini yaz.

---

## Aşama 1 — Cartpole modelleme (Lagrange + RK4)

**Süre:** 1 gün (Gün 1)

### 💡 Sade dille
Bu aşama simülasyonun **motorunu** kurar. Bir sistemin nasıl hareket ettiğini anlatan denklemlere hareket denklemi denir.
**Lagrange yöntemi** bunları bulmanın düzenli bir yolu: kuvvetlerle tek tek uğraşmak yerine iki enerjiyi yazarsın,
hareket enerjisi (T) ve konum enerjisi (V). Gerisi mekanik bir türev işlemidir ve onu sympy yapar.

Sonuçta elinde şunu söyleyen bir fonksiyon olur: *"Şu an araba konumu, araba hızı, açı ve açısal hız bu; uygulanan
kuvvet bu → ivmeler şunlar."* Buna **f** diyoruz.

Bilgisayar bu ivmeleri küçük zaman adımlarıyla üst üste ekleyerek hareketi üretir; buna **entegrasyon** denir.
- **Euler:** "şimdiki eğimle düz git". Kaba bir yöntem ve zamanla enerji sızdırır, sarkaç kendiliğinden hızlanır.
- **RK4:** adımın içinde eğime 4 kez bakıp ağırlıklı ortalama alır. Çok daha doğru.
- Benzetme: Euler bir virajı düz çizgilerle almak, RK4 virajın kavisini birkaç noktadan tahmin edip ona göre dönmek.

### 👀 Önce simüle et
F = 0 ile iki deneme yap ve animasyonu izle:
- θ₀ = 0.1 rad (yukarıya yakın): sarkaç düşer, araba da ters yöne kayar. Neden?
- θ₀ = π + 0.1 (aşağıya yakın): sarkaç aşağıda sallanır.

Sonra aynı denemeyi Euler ile tekrarla ve sallanmanın her turda büyüdüğünü gör. Enerji testi bu gözlemin sayısal hâli.

### 🎤 Mülakatta nasıl sorulur
- "Simülasyon modelinin doğru olduğunu nasıl doğruladın?" → enerji korunumu, denge noktaları, sympy çapraz kontrolü.
- "Neden RK4? Zaman adımını (dt) nasıl seçtin?"
- "Modelinin varsayımları neler?" → noktasal kütle, kütlesiz çubuk, sürtünme yok.

### 📚 Öğren
| konu | kaynak |
|---|---|
| Lagrange mekaniği (genelleştirilmiş koordinatlar, L = T − V, Euler–Lagrange denklemi) | Tedrake, *Underactuated Robotics*, **Appendix: Multi-Body Dynamics** → "Deriving the equations of motion" ✅ okundu |
| Cartpole hareket denklemleri | Tedrake, **"Acrobots, Cart-Poles, and Quadrotors"** → "The Cart-Pole System" ✅ okundu. ⚠ Tedrake θ=0'ı **aşağı** alır; sen yukarı alıyorsun |
| sympy ile Lagrange türetmesi | sympy dokümanı → **"Classical Mechanics" / `sympy.physics.mechanics`** → `LagrangesMethod` öğreticisi (ya da düz sympy ile `diff` kullanarak elle kurmak) |
| Aynı sistemin kontrol kitabındaki hâli (isteğe bağlı) | Åström & Murray, *Feedback Systems*, **"System Modeling"** → "balance systems" örneği |
| Video anlatım (isteğe bağlı) | Brunton, *Control Bootcamp* → **"Inverted Pendulum on a Cart"**. ⚠ Brunton'un modelinde sönüm terimi ve farklı açı kuralı var |
| Sayısal entegrasyon: Euler, RK4 | Wikipedia "Runge–Kutta methods" → RK4 bölümü |

### 🧮 Bilgisayarla çöz (`notebooks/01_derive_cartpole.ipynb`)
1. Semboller: M, m, l, g, F ve zamana bağlı x(t), θ(t). Genelleştirilmiş koordinatlar **q = [x, θ]**.
2. Sarkaç kütlesinin konumunu x ve θ cinsinden tanımla, sympy ile türevini al.
3. **T**, **V**, **L = T − V**; Euler–Lagrange denklemlerini sympy ile kur (x için genelleştirilmiş kuvvet F, θ için 0).
   İki yol: düz sympy (`diff`) ya da `sympy.physics.mechanics.LagrangesMethod`. Önerim düz sympy — her adımı görürsün.
4. `sympy.solve` ile **ẍ ve θ̈**'yı çöz → durum denklemi ṡ = f(s, F), **s = [x, ẋ, θ, θ̇]**.
5. Toplam enerji E(s).
6. `sympy.lambdify` ile f ve E'yi numpy fonksiyonuna çevir; sonucu `cartpole.py`'ye elle aktar ve ikisinin aynı
   olduğunu test et.

> 🔍 Ek A.1 ile karşılaştır (`sympy.simplify(senin − referans) == 0`).

### 💻 Kodla
`cartpole.py`:
- `params` (dataclass): M, m, l, g, (sonra eklenecek: sürtünme b_x, b_θ).
- `f(s, F, params) -> ds/dt`
- `energy(s, params) -> E`
- `euler_step`, `rk4_step(f, s, u, dt)`. Kontrol girdisi adım boyunca sabit (sıfırıncı derece tutma).
- Basit bir animasyon fonksiyonu (`matplotlib.animation`) → davranışı gözle görmek hata ayıklamada çok işe yarar.

### ✅ Doğrula (`test_all.py`)
- **Enerji korunumu:** F = 0, sürtünme yok, θ₀ = 0.3 rad → 10 s simülasyon; RK4'te göreli enerji kayması çok küçük
  (ör. < 1e-6) olmalı. Euler'de enerjinin büyüdüğünü gör ve not al.
- **Denge noktaları:** s = [0,0,0,0] ve s = [0,0,π,0] F = 0'da türev sıfır.
- **Fiziksel sezgi:** θ₀ küçük pozitif → sarkaç sağa düşer; F > 0 sabit → araba sağa ivmelenir ve sarkaç sola yatar.
- **`cartpole.py` ile sympy'nin `lambdify` sonucu aynı türevi veriyor** (rastgele 100 durumda).
- **Adım boyu:** dt = 0.02 ve dt = 0.001 ile koşup sonuçların yakın olduğunu göster (dt seçimini gerekçelendirir).

### 📦 Çıktı
Test edilmiş `cartpole.py`, sarkacın serbest düşüşünün animasyonu, Euler ve RK4 enerji grafiği.

### 🎯 Anladım ölçütü
_(sen yaz)_

---

## Aşama 2 — Doğrusallaştırma ve sistem analizi

**Süre:** ½ gün (Gün 2)

### 💡 Sade dille
Hareket denklemlerinde sin ve cos var, yani **doğrusal değiller**. Böyle denklemleri analiz etmek zor. Ama sarkaç dik
konuma çok yakınken sin θ ≈ θ ve cos θ ≈ 1 olur ve sistem basit bir doğrusal denkleme dönüşür:

  **ṡ = A·s + B·F**

- **A:** sistem kendi hâline bırakılınca durum nasıl değişir.
- **B:** uygulanan kuvvet durumu nasıl değiştirir.
- Benzetme: Dünya yuvarlak ama oturduğun oda düz görünür. Eğri şeyler yakından bakınca düzdür. Doğrusallaştırma
  "yakından bakmak"tır ve yalnız o noktanın yakınında geçerlidir.

**Özdeğerler** sistemin doğal hareket biçimleridir. Her biri e^(λt) gibi davranan bir parçaya karşılık gelir:
- λ > 0: büyür → sarkacın düşmesi (bulduğun ≈ +4.6 bu).
- λ < 0: söner → geri dönüş.
- λ = 0: olduğu gibi kalır → sürtünmesiz araba, ittiğin yerde durur ya da sabit hızla kayar.

**Kontrol edilebilirlik:** "Elimdeki tek kuvvetle dört durumun hepsini istediğim yere götürebilir miyim?" sorusunun
cevabı. Rank 4 ise evet.

**Sağ yarı düzlem sıfırı (ters tepki):** Arabayı sağa götürmek istiyorsan önce **sola** gitmen gerekir, çünkü sarkacı
önce sağa yatırmalısın ki sonra arabayı sağa sürükleyebilesin. Benzetme: elinde süpürge dengelerken sağa yürümek için
elini önce bir anlığına sola çekersin.

### 👀 Önce simüle et
- Doğrusal ve doğrusal olmayan modeli aynı θ₀'dan başlat (0.05, 0.2, 0.5 rad) ve üst üste çiz. Küçük açıda örtüşüyor,
  büyükte ayrılıyor. "Yakından bakınca düz" sözünün sayısal hâli bu.
- F = 0, θ₀ = 0.01 ile θ(t)'yi **logaritmik eksende** çiz. Düz bir çizgi görmelisin; eğimi ≈ 4.6 olmalı. Özdeğeri
  formülden değil, simülasyondan okumuş olursun.

### 🎤 Mülakatta nasıl sorulur
- "Doğrusallaştırma ne zaman geçerli, ne zaman bozulur?"
- "Pozitif özdeğer fiziksel olarak ne demek?"
- "Sistemin kontrol edilebilir olduğunu nasıl bildin?"
- "Minimum fazlı olmayan sistem nedir, kontrolü neden zorlaştırır?"

### 📚 Öğren
| konu | kaynak |
|---|---|
| Durum uzayı, özdeğerler ve kararlılık | Brunton *Control Bootcamp*: **"Linear Systems"**, **"Stability and Eigenvalues"** |
| Denge noktası etrafında doğrusallaştırma (Jacobian) | Brunton: **"Linearizing Around a Fixed Point"**; Åström & Murray, "Linear Systems" bölümündeki doğrusallaştırma kısmı |
| Kontrol edilebilirlik | Brunton: **"Controllability"** (ve devamındaki kontrol edilebilirlik videoları); Åström & Murray, **"State Feedback"** bölümünün başı |
| Ayrıklaştırma (sıfırıncı derece tutma, matris üsteli) | Åström & Murray, "Linear Systems" bölümü; `python-control` dokümanı `c2d` / `sample_system` |
| Gözlemlenebilirlik (Aşama 5 için şimdiden) | Brunton: **"Observability"** |
| Kendi ders notların | Modern Kontrol, ilgili haftalar |

### 🧮 Bilgisayarla çöz (`notebooks/02_linearize.ipynb`)
1. Aşama 1'deki sympy f(s, F) üzerinden **A = ∂f/∂s**, **B = ∂f/∂F**: `sympy.Matrix.jacobian`, sonra
   s* = 0, F* = 0 yerine koy. Sembolik A, B'yi gör (hangi parametreye nasıl bağlı?).
2. A'nın özdeğerleri (sembolik ve sayısal): kaç tane sıfır, kaç tane pozitif? Pozitif özdeğerin fiziksel anlamı
   (sarkacın düşme hızı).
3. Kontrol edilebilirlik matrisi **𝒞 = [B, AB, A²B, A³B]** ve rankı (`control.ctrb`, `numpy.linalg.matrix_rank`).
4. *(isteğe bağlı)* Aşağı denge noktası (θ = π) için de A ve özdeğerler.
5. **F → x transfer fonksiyonu ve sıfırları** (sympy ile C(sI − A)⁻¹B ya da `control.ss2tf` + `control.zeros`).
   Sağ yarı düzlemde sıfır var mı? (Aşama 3'te önemli olacak.)

> 🔍 Ek A.2 ile karşılaştır.

### 💻 Kodla
`cartpole.py`'ye ekle:
- `analytic_AB(params)` → sympy'den çıkan sembolik sonucun sayısallaştırılmış hâli
- `numeric_AB(f, s_star, u_star, eps)` → sonlu farklarla Jacobian
- `discretize(A, B, dt)` → (A_d, B_d)
- Özdeğerler, kontrol edilebilirlik rankı, transfer fonksiyonu sıfırları (`control.ss2tf`, `control.zeros`)

### ✅ Doğrula
- Analitik ve sayısal Jacobian eşit (tolerans içinde).
- **Doğrusal model ile doğrusal olmayan modeli karşılaştır:** θ₀ = 0.05, 0.2, 0.5 rad; F = 0; 1 s.
  Hangi açıda ayrışmaya başlıyor? Grafiği kaydet.
- Ayrık model, sürekli modelin dt ile örneklenmiş çözümüyle aynı.

### 📦 Çıktı
A, B, A_d, B_d; özdeğer ve kontrol edilebilirlik sonuçları; doğrusal / doğrusal olmayan karşılaştırma grafiği.

### 🎯 Anladım ölçütü
_(sen yaz)_

---

## Aşama 3 — PID

**Süre:** ½ gün (Gün 2)

### 💡 Sade dille
**Açık döngü ve kapalı döngü:**
- Açık döngüde kuvveti önceden belirlersin (ör. F = 0); sistem ne yaparsa yapar ve sarkaç düşer.
- Kapalı döngüde kuvveti **her an ölçtüğün duruma göre** hesaplarsın: ölç → hesapla → uygula → sistem değişir → tekrar ölç.
- Benzetme: elinin üstünde süpürge dengelemek. Süpürge sağa yatarsa elini sağa götürürsün. Yani θ > 0 ise F > 0.

**PID'nin üç terimi:**
- **P (oransal):** hata ne kadar büyükse o kadar it. Yay gibi davranır.
- **D (türev):** hata ne kadar hızlı değişiyorsa ona karşı koy. Amortisör gibi davranır, salınımı söndürür.
- **I (integral):** biriken hatayı siler (ör. sürekli bir rüzgâr varsa). Bu projede çoğu zaman gerekmez.

**Kontrolcü sistemi yeni bir sisteme dönüştürür:** F = −K·s'yi ṡ = A·s + B·F'ye koyunca ṡ = **(A − BK)·s** çıkar.
Bu "kapalı döngü matrisi"dir. Kontrolcünün tek işi, A'daki pozitif özdeğeri (düşme) **negatif tarafa çekmek**.

**Yalnız açıya PD (F = K_p·θ + K_d·θ̇)** için θ denklemi şuna dönüşür:

  θ̈ + b·K_d·θ̇ + (b·K_p − a)·θ = 0, burada a = (M+m)g/(Ml), b = 1/(Ml)

İkinci dereceden bir denklemde kararlılık şartı **"bütün katsayılar pozitif"**. Routh tablosu gerekmez. Buradan:
- **K_d > 0:** sönüm olmalı, yoksa sarkaç sonsuza kadar salınır.
- **K_p > (M+m)·g ≈ 10.8 N/rad:** itme, yerçekiminin sarkacı devirme etkisinden güçlü olmalı.

**Neden araba sürüklenir?** Bu PD yalnız açıya bakıyor; arabanın konumu ve hızı hesapta yok. A − BK'nın iki
özdeğeri hep sıfırda kalıyor. Hangi durumlara ait olduklarını bulmak senin sorun.

**Kaskad PID:** iki katlı bir karar yapısı.
- Dış döngü: "arabanın olması gereken yer burası → sarkaç şu kadar yatmalı" (açı referansı θ_ref).
- İç döngü: "sarkacı θ_ref'e yatır".
- İç döngü dış döngüden belirgin şekilde hızlı olmalı.

### 👀 Önce simüle et
Formüle girmeden önce birkaç (K_p, K_d) çiftini deneyip animasyonu izle:
- K_p = 5, K_d = 1: sarkaç düşer (K_p yetersiz).
- K_p = 20, K_d = 0: sarkaç durmadan salınır (sönüm yok).
- K_p = 20, K_d = 3: sarkaç durur ama araba kayıp gider.

Sonra ısı haritasını çiz: bu üç gözlemin sınırları yukarıdaki iki şartla örtüşmeli.

### 🎤 Mülakatta nasıl sorulur
- "P, I ve D terimleri ne işe yarar?"
- "Kapalı döngü kararlılığını nasıl gösterdin?"
- "PID cartpole için neden zor? Kaskad yapıya neden ihtiyaç var?"
- "Arabayı sağa götürmek için neden önce sola gitmesi gerekiyor?"

### 📚 Öğren
| konu | kaynak |
|---|---|
| PID yapısı, P / I / D terimlerinin etkisi, türev filtresi, integral sarması (anti-windup) | Åström & Murray, **"PID Control"** bölümü; bilgisayarındaki `pid-kontrol-ders-notlari.pdf` |
| Kaskad (iç içe) kontrol | Åström & Murray, PID bölümü + iç döngü / dış döngü bant genişliği ayrımı |
| Minimum fazlı olmayan sistemler (sağ yarı düzlem sıfırı → ters yönlü ilk tepki) | Åström & Murray, "Frequency Domain Design" / "Fundamental Limits" bölümleri (sağ yarı düzlem sıfırları) |

### 🧮 Bilgisayarla çöz (`notebooks/03_pid_analysis.ipynb`)
1. **Yalnız açıya PD** (F = K_p θ + K_d θ̇): A − B·K_PD kapalı döngü matrisini sympy ile kur, karakteristik polinomu
   çıkar (`charpoly`). θ'yu kararlı yapan K_p, K_d aralığını bul (ikinci dereceden polinomda tüm katsayılar pozitif olmalı; Routh tablosu
   gerekmez). Sayısal olarak
   doğrula: K_p × K_d ızgarasında θ alt sisteminin özdeğerlerinin gerçel kısımlarını ısı haritası olarak çiz.
   (**İşarete dikkat**: sarkaç sağa yatıyorsa araba hangi yöne itilmeli?)
2. Aynı kapalı döngüde x'e ait özdeğerler nerede? Neden araba sürüklenir?
3. **Kaskad yapı:** dış döngü x hatasından bir **açı referansı θ_ref** üretir, iç döngü θ'yu θ_ref'e götürür.
   Simülasyonda arabayı sağa götürürken θ(t) ve x(t)'nin ilk anını çiz: sarkaç önce hangi yöne yatıyor, araba önce hangi
   yöne gidiyor? Bunu Aşama 2'deki sağ yarı düzlem sıfırıyla ilişkilendir.

### 💻 Kodla
`controllers.py`:
- Ayrık PID sınıfı: `reset()`, `__call__(error, dt)`; türev için düşük geçiren filtre, integral için anti-windup, çıkış doyması.
- `AnglePD` (yalnız açı) ve `CascadePID` (konum → açı referansı → kuvvet).
- Kapalı döngü simülasyon fonksiyonu: `simulate(controller, s0, T, dt, params)` → durum ve kuvvet geçmişi.

### ✅ Doğrula
- Yalnız açı PD: θ dengede, x sürükleniyor → grafikle göster.
- Kaskad PID: θ₀ = 0.1 rad, x₀ = 0.5 m → hem θ hem x sıfıra gidiyor.
- Kuvvet doyması açıkken de çalışıyor.
- Ayar sürecini kısa bir notla yaz: kaç deneme sürdü, ne zor oldu?

### 📦 Çıktı
İki PID denetleyicisi, cevap grafikleri, ayar notu.

### 🎯 Anladım ölçütü
_(sen yaz)_

---

## Aşama 4 — LQR

**Süre:** 1 gün (Gün 3)

### 💡 Sade dille
PID'de kazançları deneme-yanılmayla ayarladın ve iki ayrı döngü kurman gerekti. LQR'de ise kazançları sen seçmezsin;
**neyi ne kadar önemsediğini** söylersin, matematik en iyi kazancı hesaplar:
- **Q:** hangi hataya ne kadar kızarım? (Açı hatası mı daha kötü, konum hatası mı?)
- **R:** kuvvet harcamaya ne kadar kızarım? (Büyük R → tutumlu ama yavaş kontrolcü.)

Çıkan **K** dört durumun hepsine birden bakar. Hem açıyı hem konumu tek bir formülle kontrol eder: F = −K·s.

**P matrisi ve RL bağlantısı:** P, "bu durumdan başlarsan en iyi kontrolle toplam ne kadar maliyet ödersin" sorusunun
cevabını verir: V(s) = sᵀ·P·s. Bu, RL'deki **değer fonksiyonunun** ta kendisi. Fark şu: model bilindiği ve doğrusal
olduğu için kapalı formda hesaplanabiliyor. Riccati denklemini döngüyle yinelemek de "son adımdan geriye doğru en iyi
kararı hesapla" demek; RL'deki **değer iterasyonunun** doğrusal hâli.

**Çekim bölgesi:** kontrolcünün kurtarabildiği başlangıç durumlarının haritası.

**Doyma:** LQR "10 N'den fazla itemem" bilgisini bilmez; büyük açıda istediği kuvvet kırpılır ve kurtaramaz.

### 👀 Önce simüle et
- Bryson kuralıyla bir K al. Aynı başlangıçtan PID ve LQR animasyonunu yan yana izle.
- θ₀'ı adım adım büyüt (10°, 20°, 30°…). LQR'nin hangi açıda kaybettiğini gör, sonra F(t) grafiğine bak: kuvvet
  ne zamandan beri sınırda?

Çekim bölgesi haritası bu gözlemin sistematik hâli.

### 🎤 Mülakatta nasıl sorulur
- "Q ve R'yi nasıl seçtin, değiştirince ne oldu?"
- "LQR hangi varsayımlarla optimal? Gerçekte hangileri bozuluyor?" → doğrusal model, kısıt yok, tam durum bilgisi.
- "LQR ile RL arasındaki ilişki ne?" → değer fonksiyonu, dinamik programlama.
- "LQR'nin çekim bölgesini neler sınırlıyor?"

### 📚 Öğren
| konu | kaynak |
|---|---|
| Kutup yerleştirme (LQR'ye geçiş için) | Brunton: **"Pole Placement for the Inverted Pendulum on a Cart"** |
| LQR: maliyet fonksiyonu, Riccati denklemi, K = R⁻¹BᵀP | Brunton: **"LQR Control for an Inverted Pendulum on a Cart"**; Åström & Murray, **"State Feedback"** bölümü → "Linear Quadratic Regulators" |
| LQR'nin optimal kontrol kökeni (dinamik programlama / HJB) — RL ile bağlantı | Tedrake, **"Linear Quadratic Regulators"** bölümü. ⭐ Değer fonksiyonu V(x) = xᵀPx: RL'deki değer fonksiyonunun kapalı form hâli |
| Ayrık zamanlı LQR | Tedrake LQR bölümü (ayrık zaman kısmı); `scipy.linalg.solve_discrete_are` dokümanı |
| Çekim bölgesi (region of attraction) kavramı | Tedrake, "Lyapunov Analysis" bölümünün başı (yalnız kavram, ayrıntı gerekmez) |

### 🧮 Bilgisayarla çöz (`notebooks/04_lqr.ipynb`)
1. **Riccati'yi elle iterasyonla çöz:** ayrık Riccati denklemini (Ek A.3) P₀ = Q'dan başlayıp yakınsayana kadar döngüyle
   yinele (bu, sonlu ufuklu dinamik programlamanın ta kendisi — RL'deki değer iterasyonunun doğrusal-karesel hâli).
   Sonucu `scipy.linalg.solve_discrete_are` ve `control.dlqr` ile karşılaştır. P simetrik ve pozitif tanımlı mı
   (özdeğerleri)?
2. **Değer fonksiyonunu doğrula:** rastgele bir s₀'dan LQR ile simüle edip gerçekleşen toplam maliyeti hesapla;
   s₀ᵀ P s₀ ile aynı mı?
3. **Q ve R'nin fiziksel anlamı:** Q'daki θ ağırlığını 10 kat artırırsan ne beklersin? R'yi artırırsan?
   Önce `KARARLAR.md`'ye tahmin yaz, sonra koşuda kontrol et.
4. **Bryson kuralı:** Q_ii = 1 / (izin verilen en büyük durum_i)², R = 1 / F_max² → başlangıç ayarın.

> 🔍 Ek A.3 ile karşılaştır.

### 💻 Kodla
`controllers.py`'ye ekle:
- `lqr_continuous(A, B, Q, R)` ve `lqr_discrete(A_d, B_d, Q, R)` → K. `scipy.linalg.solve_continuous_are` /
  `solve_discrete_are` ile; sonucu `control.lqr` / `control.dlqr` ile çapraz kontrol et.
- `LQRController(K, F_max)`: u = −K s, doymalı.

### ✅ Doğrula
- Kapalı döngü özdeğerleri: A − BK (sürekli) sol yarı düzlemde; A_d − B_d K birim çember içinde.
- θ₀ = 0.1 rad, x₀ = 0.5 m cevabı; PID ile aynı grafikte karşılaştır.
- **Q/R taraması:** 3–4 farklı ayarın oturma süresi ve ∫F² değerleri → takas tablosu. Tahminin doğru çıktı mı?
- **Çekim bölgesi taraması:** θ₀ ∈ [0°, 90°] (1° adım), ẋ₀ = θ̇₀ = 0; doymalı ve doymasız LQR için başarı/başarısızlık.
  Doyma sınır açısını ne kadar düşürüyor? 2B tarama (θ₀ × θ̇₀) ile çekim bölgesinin haritasını çiz.

### 📦 Çıktı
LQR denetleyicisi, Q/R takas tablosu, **çekim bölgesi haritası** (projenin en önemli grafiklerinden biri).

### 🎯 Anladım ölçütü
_(sen yaz)_

---

## Aşama 4b — MPC (model öngörülü kontrol)

**Süre:** 1 gün (Gün 4)

### 💡 Sade dille
LQR, kapalı formda tek bir K kazancı verir ve **kısıtları bilmez**: kuvvet sınırı (F_max) ve ray sınırı (|x| ≤ 2.4 m)
hesabın içinde yoktur, sonradan kırpılır. Aşama 4'teki çekim bölgesi haritasında bunun bedelini gördün.
MPC her kontrol adımında şunu yapar:
1. Mevcut durumdan başlayarak **önümüzdeki N adımı** modelle öngör.
2. Bu N adım için maliyeti en aza indiren kuvvet dizisini, **kısıtlara uyarak** bir optimizasyon problemiyle bul.
3. Bu dizinin **yalnız ilk** kuvvetini uygula; bir sonraki adımda yeni ölçümle her şeyi baştan yap
   ("kayan ufuk", receding horizon).

Bedeli: her adımda bir optimizasyon çözmek → hesap süresi. RL ile en sık karşılaştırılan yöntem budur:
**MPC çevrimiçi optimizasyon** yapar ve model ister; **RL çevrimdışı öğrenir**, çalışırken yalnız bir ağ çalıştırır.

Benzetme: satranç oyuncusu her hamlede birkaç hamle ilerisini düşünür, en iyi planı seçer, yalnız ilk hamleyi oynar;
rakip oynayınca yeniden düşünür.

### 👀 Önce simüle et
- cvxpy ile **tek bir** N adımlık problemi çöz. Planlanan kuvvet dizisini (u₀ … u_{N−1}) ve öngörülen θ'yı çiz.
- İlk kuvveti uygula, simülasyonu bir adım ilerlet, yeniden çöz. Yeni planı eskisinin üstüne çiz: plan her adımda
  biraz değişiyor. "Kayan ufuk" bu.

### 🎤 Mülakatta nasıl sorulur
- "MPC ile LQR arasındaki fark ne, ne zaman hangisi?"
- "MPC ile RL'yi nasıl karşılaştırırsın?" → model gereksinimi, kısıtlar, çalışma anı hesap süresi.
- "MPC'de kararlılığı nasıl sağlarsın?" → terminal maliyet (LQR'nin P'si).
- "Gerçek zamanda çalışır mı? Çözüm süresini nasıl ölçtün?"

### 📚 Öğren
| konu | kaynak |
|---|---|
| MPC'nin fikri, kayan ufuk | Brunton: **"Model Predictive Control"** videosu (YouTube) |
| Doğrusal MPC'yi karesel program (QP) olarak kurmak | **cvxpy** dokümanı → örnekler → **"Control"** örneği (doğrusal sistem, kısıtlı kontrol) |
| Terminal maliyet ve kararlılık (neden son adıma LQR'nin P'si konur) | Borrelli, Bemporad, Morari, *Predictive Control for Linear and Hybrid Systems* (yazarların sitesinde ücretsiz PDF) → doğrusal MPC ve kararlılık bölümleri (yalnız ana fikir) |

### 🧮 Bilgisayarla çöz (`notebooks/04b_mpc.ipynb`)
1. **Problemi cvxpy ile kur:** değişkenler s₀…s_N ve u₀…u_{N−1}.
   - Maliyet: Σ (s_kᵀ Q s_k + u_kᵀ R u_k) + **s_Nᵀ P s_N** (P = Aşama 4'teki LQR Riccati çözümü).
   - Kısıtlar: s_{k+1} = A_d s_k + B_d u_k, |u_k| ≤ F_max, |x_k| ≤ x_max, s₀ = mevcut durum.
2. **MPC ile LQR'nin aynılığı:** kısıtlar aktif değilken (küçük açı) MPC'nin ilk kuvveti LQR'nin −K s'si ile aynı mı?
   (Terminal maliyet P ise birebir aynı olmalı. Neden? Bu, iki yöntemin bağlantısını gösteren en önemli test.)
3. **Ufuk etkisi:** N = 5, 10, 30 için aynı başlangıçtan cevap ve her adımın çözüm süresi.
4. **Süre bütçesi:** dt = 20 ms. Her adımın çözümü bu bütçeye sığıyor mu?

### 💻 Kodla
`controllers.py`'ye ekle:
- `MPCController(A_d, B_d, Q, R, P, N, F_max, x_max)`: problemi **bir kez** kur; başlangıç durumunu `cvxpy.Parameter`
  olarak ver ve her adımda yalnız onu güncelle (her adımda problemi yeniden kurmak çok yavaştır).
- Diğer denetleyicilerle aynı arayüz: `reset()`, `act(s) -> F`.

### ✅ Doğrula
- Kısıtsız bölgede MPC ≈ LQR (madde 2).
- **Çekim bölgesi:** Aşama 4'teki haritayı MPC için de çiz. Doymalı LQR'ye göre sınır açısı genişliyor mu?
  (Dürüst beklenti: MPC de **doğrusal model** kullanıyor; dik konumdan uzakta model yanlışlaştığı için kazanç sınırlı
  olabilir. Ne kadar kazandırdığını ölç ve yaz.)
- **Ray kısıtı:** büyük x₀ ile başlayan bir senaryoda LQR rayın dışına çıkarken MPC içeride kalıyor mu?

### 📦 Çıktı
MPC denetleyicisi, LQR / MPC çekim bölgesi karşılaştırması, ufuk ve çözüm süresi tablosu.

### 🎯 Anladım ölçütü
_(sen yaz)_

---

## Aşama 5 — Kalman filtresi ve LQG

**Süre:** 1 gün (Gün 5). **Derin eksenin (kestirim) ilk adımı.**

### 💡 Sade dille
Şimdiye kadar kontrolcüye dört durumu da tam ve gürültüsüz verdin. Gerçekte sensörler her şeyi ölçmez; ölçtüklerini de
gürültülü ölçer. Burada yalnız konum (x) ve açı (θ) ölçülüyor, hızlar ölçülmüyor.

**Kalman filtresi iki bilgi kaynağını birleştirir:**
- **Model:** "Fizik kurallarına göre şimdi burada olmalıyım."
- **Ölçüm:** "Sensör burada olduğumu söylüyor."

Hangisine ne kadar güveneceğine belirsizliklerine göre karar verir:
- **Q_kf:** model ne kadar yanılabilir?
- **R_kf:** sensör ne kadar gürültülü?
- **P:** filtrenin kendi tahminine ne kadar emin olduğu.

Benzetme: sisli bir havada gemi kullanmak. Hızından ve rotandan nerede olduğunu hesaplarsın (model); ara sıra bir
fenerin görünür (ölçüm). Fener netse ona güvenirsin, puslu görünüyorsa hesabına.

**Neden hızları sonlu farkla hesaplamıyoruz?** (θ_k − θ_{k−1}) / dt işleminde dt küçük olduğu için gürültü de dt'ye
bölünüp büyür.

**LQG:** Kalman'ın kestirimini LQR'ye vermek. **Ayrılma ilkesi:** filtreyi ve kontrolcüyü ayrı ayrı tasarlayıp
birleştirebilirsin (doğrusal sistem ve Gauss gürültü için).

**Gözlemlenebilirlik:** "Ölçtüklerimden ölçmediklerimi çıkarabilir miyim?" sorusunun cevabı.

### 👀 Önce simüle et
- θ'yu gürültüyle ölç, θ̇'yı sonlu farkla hesapla ve gerçek θ̇ ile üst üste çiz. Ne kadar gürültülü olduğunu gör.
- Bu gürültülü hızı LQR'ye ver ve F(t)'ye bak: kuvvet titriyor mu?
- Sonra Kalman'ın kestirdiği θ̇'yı aynı grafiğe ekle.

### 🎤 Mülakatta nasıl sorulur
- "Kalman filtresinde Q ve R ne anlama geliyor, nasıl ayarladın?"
- "Filtrenin doğru çalıştığını nasıl anlarsın?" → kestirim hatası, NEES tutarlılık testi.
- "Ayrılma ilkesi nedir? LQG'nin bilinen zayıflığı ne?" → kararlılık payı garantisi yok (Doyle).
- "Ölçmediğin bir durumu nasıl kestirirsin?"

### 📚 Öğren
| konu | kaynak |
|---|---|
| Neden durum kestirimi gerekir, gözlemlenebilirlik | Brunton: **"Motivation for Full-State Estimation"**, **"Observability"**, **"Full-State Estimation"** |
| Kalman filtresi | Brunton: **"The Kalman Filter"**; ⭐ **Roger Labbe, *Kalman and Bayesian Filters in Python*** (ücretsiz, GitHub) → "g-h Filter", "Discrete Bayes", "One Dimensional Kalman Filters", **"Multivariate Kalman Filters"** bölümleri |
| Gaussian dağılım, kovaryans, Bayes kuralı (temel olasılık) | Labbe, **"Probabilities, Gaussians, and Bayes' Theorem"** ve **"Multivariate Gaussians"** bölümleri |
| LQG ve ayrılma ilkesi (separation principle) | Brunton: **"LQG"** videoları; Åström & Murray, **"Output Feedback"** bölümü |

### 🧮 Bilgisayarla çöz (`notebooks/05_kalman.ipynb`)
1. Ölçüm modeli: yalnız x ve θ ölçülüyor → **H**. Gözlemlenebilirlik matrisi ve rankı (`control.obsv`).
   Yalnız θ ölçülseydi gözlemlenebilir olur muydu? Yalnız x? (İkisini de hesapla — sonuç sezgine uyuyor mu?)
2. *(isteğe bağlı)* **Durağan Kalman kazancı:** filtrenin Riccati denklemini `solve_discrete_are` ile çöz (LQR'nin "dual"ı:
   A → Aᵀ, B → Hᵀ). Kodundaki yinelemeli filtrenin P'si bu değere yakınsıyor mu?
3. **R_kf deneyi:** R_kf'yi 0.01×, 1×, 100× yapıp Kalman kazancının büyüklüğünü çiz. Filtre ölçüme mi modele mi
   güveniyor? (⚠ LQR'nin Q/R'si ile karıştırma — kodda `Q_lqr`, `Q_kf` gibi farklı isimler kullan.)
4. **Ayrılma ilkesi:** LQG kapalı döngü özdeğerleri = {A − BK'nin özdeğerleri} ∪ {A − LH'nin özdeğerleri} olmalı.
   Kapalı döngü matrisini kurup sayısal olarak kontrol et.

> 🔍 Ek A.4 ile karşılaştır.

### 💻 Kodla
`controllers.py`'ye ekle:
- `KalmanFilter(A_d, B_d, H, Q_kf, R_kf)`: `predict(u)`, `update(z)`, durum ve kovaryans geçmişi.
- Simülasyona ölçüm gürültüsü ekle: z = H s + v, v ~ N(0, R_kf).
- `LQGController`: Kalman kestirimi → LQR.
- Karşılaştırma için bir de "sonlu farkla hız" denetleyicisi: hızları (θ_k − θ_{k−1}) / dt ile hesaplayıp LQR'ye ver.

### ✅ Doğrula
- **Kestirim hatası:** gerçek ẋ, θ̇ ile kestirilen değerleri üst üste çiz.
- **Tutarlılık testi (NEES):** kestirim hatası, filtrenin kendi kovaryansıyla uyumlu mu? (Labbe'de "Evaluating Filter
  Performance" kısmı.) Bu test filtrenin "kendine fazla güvenip güvenmediğini" gösterir.
- Üç kontrolcü aynı gürültüde: tam durumlu LQR (ideal) vs sonlu farklı LQR vs LQG. ∫F² ve θ RMS tablosu.
- Gürültüyü 3 seviyede artır: hangisi önce bozuluyor?
- R_kf'yi bilerek yanlış ayarla (10 kat küçük / büyük): ne oluyor?

### 📦 Çıktı
Kalman filtresi, LQG denetleyicisi, kestirim grafikleri, gürültü seviyesi tablosu.

### 🎯 Anladım ölçütü
_(sen yaz)_

---

## Aşama 6 — Ortak değerlendirme düzeneği

**Süre:** ½ gün (Gün 6). **RL'den ÖNCE yazılır**, böylece RL'yi klasik denetleyicilerle aynı ölçütte değerlendirirsin.

### 💡 Sade dille
Adil bir karşılaştırma için her denetleyici **aynı başlangıç durumlarıyla, aynı sınırlarla ve aynı ölçütlerle** sınanır.
- **Tohum:** senaryoları rastgele ama tekrarlanabilir şekilde üretmenin yolu. Aynı tohum her zaman aynı senaryoları verir.
- **Geliştirme bloğu ve nihai blok:** ayar yaparken geliştirme bloğunu kullanırsın; nihai bloğa en sonda bir kez
  bakarsın. Aksi halde farkında olmadan teste göre ayar yaparsın. Sınav sorularını önceden görmek gibi.
- **Güven aralığı:** 50 denemede 45 başarı "%90" eder, ama gerçek oran kabaca %78–96 arasında olabilir. Az denemeyle
  "A, B'den iyi" demeden önce aralıklara bak.

### 👀 Önce elle yap
5 senaryoyu PID ve LQR ile tek tek koştur, sonuçları elle bir tabloya yaz. Ne ölçtüğünü netleştirdikten sonra
otomatikleştir.

### 🎤 Mülakatta nasıl sorulur
- "Sonuçlarının tesadüf olmadığını nasıl gösterdin?"
- "Karşılaştırmaların adil miydi? Aynı koşulları nasıl sağladın?"
- "Deneylerin tekrar üretilebilir mi?"

### 📚 Öğren
| konu | kaynak |
|---|---|
| Ortak rastgele sayılar (CRN), eşleştirilmiş karşılaştırma | BVR'daki `scripts/eval_commander.py` ve `REQUIREMENTS.md` EVAL bölümleri — kendi işin |
| Başarı oranı için güven aralığı (Wilson) | Wikipedia "Binomial proportion confidence interval" → Wilson score interval |
| Az tohumla RL sonuçlarını raporlamak | Agarwal ve ark. (2021) "Deep RL at the Edge of the Statistical Precipice" — özet bölümü + `rliable` README |

### 📝 Karar ver (`KARARLAR.md`'ye yaz)
1. **Başarı tanımı:** ör. bölüm sonuna kadar |θ| < 12° ve |x| < 2.4 m, ve son 2 saniyede |θ| < 2°.
2. **Ölçütler:** başarı oranı, oturma süresi, ∫F² dt (efor), θ RMS, en büyük |x|.
3. **Senaryo dağılımı:** başlangıç durumları (ör. θ₀ ∼ U(−0.2, 0.2) rad, x₀ ∼ U(−0.5, 0.5) m, hızlar küçük),
   **sabit tohum blokları**: geliştirme bloğu ve nihai test bloğu (nihai bloğa en sonda bir kez bakılır).
4. Bölüm süresi (ör. 10 s = 500 adım, dt = 0.02).

### 💻 Kodla
`evaluate.py` (senaryo üretimi, ölçütler ve karşılaştırma tek dosyada):
- Girdi: bir denetleyici (ortak arayüz: `reset()`, `act(obs) -> F`), senaryo bloğu, parametreler (Aşama 12'de değişecek).
- Çıktı: senaryo başına satır (CSV) + özet tablo (ortalama, Wilson aralığı).
- Denetleyicileri eşleştirilmiş karşılaştıran bir fonksiyon (aynı senaryolarda fark).

### ✅ Doğrula
- Aynı tohum → aynı senaryo, aynı sonuç (tekrar üretilebilirlik testi).
- PID, LQR, MPC, LQG'yi düzenekten geçir; Aşama 3–5'teki sonuçlarla tutarlı.

### 📦 Çıktı
Klasik denetleyicilerin ilk ortak tablosu (tam durum ve gürültülü ölçüm altında).

### 🎯 Anladım ölçütü
_(sen yaz)_

---

## Aşama 7 — Gymnasium ortamı

**Süre:** ½ gün (Gün 6)

### 💡 Sade dille
Gymnasium ortamı, simülasyonunu RL kütüphanelerinin anlayacağı standart bir **oyuna** çeviren sarmalayıcıdır:
- `reset()`: oyunu baştan başlat, ilk gözlemi ver.
- `step(aksiyon)`: bir adım oyna; yeni gözlem, ödül ve "oyun bitti mi" bilgisini döndür.

**Ödül**, RL'ye ne istediğini söylemenin **tek** yolu. LQR'deki maliyetin tersi gibi düşünebilirsin. Ödül yanlış
tasarlanırsa ajan kuralın boşluğunu bulur (ödül sömürüsü): ör. hayatta kalmaya ödül verip konumu cezalandırmazsan
ajan arabayı rayın kenarına götürüp orada bekleyebilir.

**İki farklı "bitti":**
- `terminated`: oyun kaybedildi (sarkaç düştü). Gelecek yok.
- `truncated`: süre doldu. Aslında gelecek vardı, sadece kestik.

Ajan bu ikisini karıştırırsa süre sonunu "ölüm" sanar ve yanlış öğrenir.

### 👀 Önce simüle et
Ortamı üç politikayla oynat ve her birinin toplam ödülünü ve bölüm uzunluğunu yaz: rastgele aksiyon, sıfır kuvvet ve
LQR. LQR açık ara en yüksek ödülü almalı. Almıyorsa ödülün ya da ortamın hatalıdır.

### 🎤 Mülakatta nasıl sorulur
- "Ödül fonksiyonunu nasıl tasarladın, neden böyle?"
- "Ajanın ödülü sömürdüğü bir durum oldu mu?"
- "Gözlem ve aksiyon uzayını nasıl seçtin, neden aksiyonu [−1, 1]'e ölçekledin?"

### 📚 Öğren
| konu | kaynak |
|---|---|
| Gymnasium arayüzü: `reset`, `step`, `observation_space`, `action_space`, `terminated` / `truncated` farkı | Gymnasium dokümanı → **"Create a Custom Environment"** öğreticisi; "Basic Usage" sayfası |
| SB3 için ortam tasarımı ipuçları (aksiyonu [−1, 1]'e ölçekleme, gözlem normalizasyonu) | SB3 dokümanı → **"Reinforcement Learning Tips and Tricks"** ve **"Using Custom Environments"** |
| Ödül tasarımı | Sutton & Barto, bölüm 3 (MDP, ödül hipotezi); Spinning Up "Key Concepts" |

### 📝 Karar ver (`KARARLAR.md`)
1. **Gözlem:** tam durum [x, ẋ, θ, θ̇] (temel), ve gürültülü sürüm (ölçüm gürültüsü bayrağıyla).
2. **Aksiyon:** `Box(-1, 1)`; ortam içinde F = a · F_max.
3. **Ödül:** LQR maliyetinin negatifine benzer bir şey, ör. r = −(sᵀ Q s + R F²)·dt, ya da 1 − normalize maliyet.
   Düşünmen gereken: düşüşte bölüm bitince ajan "erken bitirip cezadan kaçmayı" öğrenebilir mi?
   Pozitif sabit + maliyet mi, yalnız negatif maliyet mi? Seçimini ve gerekçeni yaz.
4. **Bitiş:** `terminated` (sarkaç düştü / araba sınır dışı), `truncated` (süre doldu). Bu ayrım bootstrapping için önemli.
5. **Başlangıç dağılımı:** eğitimde Aşama 6'daki dağılım ya da biraz daha geniş.

### 💻 Kodla
`envs.py`: `CartPoleContinuousEnv(params, obs_noise=0, ...)`; dinamik için Aşama 1'deki RK4 kullanılır
(**yeniden yazılmaz**). `render()` için Aşama 1'deki animasyon.

### ✅ Doğrula
- `gymnasium.utils.env_checker.check_env(env)` hatasız.
- **LQR'yi ortamın içinden çalıştır:** Aşama 4 sonuçlarıyla birebir aynı mı? (Ortamın dinamik veya ölçeklemede hata
  yapmadığının kanıtı.)
- Rastgele politika ve sıfır politika: ortalama ödül ve bölüm uzunluğu → RL için alt referans.

### 📦 Çıktı
Test edilmiş ortam, alt referans değerleri.

### 🎯 Anladım ölçütü
_(sen yaz)_

---

## Aşama 8 — RL teorisi: policy gradient, PPO, SAC

**Süre:** 1 gün (Gün 7). Okumaların bir kısmı Aşama 9'daki eğitimler koşarken de yapılabilir.

### 💡 Sade dille
LQR modeli biliyordu ve en iyi kontrolü hesapladı. RL modeli bilmez; **deneme-yanılmayla** öğrenir.
- **Politika:** durumu alıp aksiyon üreten sinir ağı. Eğitimin amacı bu ağı iyileştirmek.
- **Policy gradient:** "iyi sonuç veren aksiyonların olasılığını artır, kötülerinkini azalt."
- **Avantaj:** bu aksiyon, o durumda ortalama bir aksiyondan ne kadar iyiydi?
- **Değer fonksiyonu:** "bu durumdan sonra ne kadar ödül toplarım?" LQR'deki sᵀ·P·s'nin örneklerden öğrenilen hâli.

**İki algoritma:**
- **PPO (on-policy):** yalnız şu anki politikanın topladığı veriyle öğrenir, kullandıktan sonra veriyi atar.
  Güncellemeleri küçük tutar (kırpma), çünkü politika bir anda çok değişirse performans çöker. Kararlı ama veri israf eder.
- **SAC (off-policy):** geçmiş deneyimleri bir havuzda saklar ve tekrar tekrar kullanır, bu yüzden daha az veriyle
  öğrenir. Ayrıca "entropi" terimiyle rastgele davranmayı ödüllendirir; bu keşfi canlı tutar.

Benzetme: PPO her antrenmandan sonra notlarını atan ama dikkatli ilerleyen bir öğrenci; SAC eski sınavlarını saklayıp
tekrar tekrar çalışan bir öğrenci.

### 👀 Önce simüle et
Bu aşamadaki "Bilgisayarla çöz" maddeleri zaten küçük simülasyonlar. Önce grafiklere bak, sonra formülü oku:
ör. PPO kırpma grafiğinde düz kalan bölgeyi gördükten sonra makaledeki formülü okumak çok daha kolay.

### 🎤 Mülakatta nasıl sorulur
- "PPO ile SAC'ın farkı ne? Hangisini ne zaman seçersin?"
- "PPO'daki kırpma ne işe yarıyor?"
- "On-policy ve off-policy ne demek?"
- "Keşfi nasıl sağlıyorsun?"
- "Değer fonksiyonu ile LQR'nin P matrisi arasında nasıl bir ilişki var?"

### 📚 Öğren
| sıra | konu | kaynak |
|---|---|---|
| 1 | MDP, getiri, değer fonksiyonları, Bellman denklemi | Sutton & Barto, bölüm 3–4 (bilgisayarında); Spinning Up → **"Part 1: Key Concepts in RL"** |
| 2 | Policy gradient teoremi, REINFORCE, taban çizgisi | Sutton & Barto, **bölüm 13**; Spinning Up → **"Part 3: Intro to Policy Optimization"**. Kendi `RL Algorithm Designs/REINFORCE` kodunu tekrar oku |
| 3 | Actor-critic, avantaj, **GAE** | Spinning Up "Part 3" sonu; Schulman ve ark. (2016) "High-Dimensional Continuous Control Using Generalized Advantage Estimation" — yalnız GAE formülü kısmı. Kendi `A2C` kodun |
| 4 | **PPO:** oran (ratio), kırpılmış amaç, değer kaybı, entropi | Spinning Up → **"Proximal Policy Optimization"** sayfası; Schulman ve ark. (2017) PPO makalesi |
| 5 | PPO'nun pratik ayrıntıları | ⭐ Huang ve ark., **"The 37 Implementation Details of Proximal Policy Optimization"** (ICLR 2022 Blog Track) — en azından "core" ve "continuous action" kısımları |
| 6 | Off-policy, Q-öğrenme, replay buffer, DDPG/TD3 | Spinning Up → **"DDPG"**, **"TD3"** sayfaları. Kendi `Continuous Action Space` kodun |
| 7 | **SAC:** maksimum entropi amacı, iki Q ağı, sıcaklık (α) ve otomatik ayarı, reparametrizasyon | Spinning Up → **"Soft Actor-Critic"** sayfası; Haarnoja ve ark. (2018) SAC makaleleri (ilki ve "Algorithms and Applications") |
| 8 | ⭐ **LQR ile RL'nin bağlantısı** | Tedrake LQR bölümü (değer fonksiyonu xᵀPx) + Sutton & Barto bölüm 4 (dinamik programlama). Bilinen modelde DP → LQR; bilinmeyen modelde örneklerden → RL |
| (isteğe bağlı) | Video ders | David Silver RL dersleri (UCL), 1–7. dersler; ya da Hugging Face Deep RL Course |

### 🧮 Bilgisayarla çöz (`notebooks/08_rl_theory.ipynb`) — teoriyi küçük sayısal deneylerle doğrula
1. **Log-türev hilesi (policy gradient):** 1 boyutlu Gauss politika π(a) = N(μ, σ²) ve basit bir ödül, ör.
   r(a) = −(a − 2)². ∇_μ E[r] için (i) analitik gradyan, (ii) Monte Carlo ile E[r(a)·∇_μ log π(a)] tahmini,
   (iii) taban çizgisi (baseline) çıkarılmış tahmin. Örnek sayısına göre varyansı çiz → taban çizgisinin işe yaradığını gör.
2. **PPO kırpılmış amacı:** L(r) = min(r·Â, clip(r, 1−ε, 1+ε)·Â) fonksiyonunu oran r'ye göre Â > 0 ve Â < 0 için çiz.
   Gradyanın nerede sıfırlandığını grafikten oku → "çok büyük güncellemeyi engelleme" mekanizması.
3. **GAE:** sahte bir ödül ve değer dizisi üzerinde Â'yı λ ∈ {0, 0.5, 0.95, 1} için hesapla. λ = 0'da tek adımlı TD
   hatasına, λ = 1'de Monte Carlo getiri − V'ye eşit olduğunu sayısal olarak kontrol et.
4. *(isteğe bağlı)* **Aşırı tahmin yanlılığı (SAC/TD3'te neden iki Q ağı):** gürültülü Q tahminlerinin maksimumunu al →
   E[max] > max E. Tek tahmin ile "iki tahminden küçüğü" yanlılığını karşılaştır.
5. On-policy ve off-policy farkı: hangisinin cartpole'da daha az örnekle öğrenmesini beklersin?
   **Tahminini `KARARLAR.md`'ye yaz.**

### 📦 Çıktı
Teori defteri (4 küçük deney + grafikler), tahminler.

### 🎯 Anladım ölçütü
_(sen yaz)_

---

## Aşama 9 — RL eğitimi: PPO ve SAC ile cartpole

**Süre:** 1–1½ gün (Gün 7–8). Eğitimler arka planda koşarken Aşama 8'in kalanını ve Aşama 11'in okumalarını yap.

### 💡 Sade dille
Eğitimde ajan yüz binlerce adım oynar. Her N adımda bir **sınav** yapılır: geliştirme bloğundaki senaryolar,
rastgelelik kapalı (deterministik) oynatılır. Böylece "ne kadar öğrendi" ölçülür.
- **Öğrenme eğrisi:** kaç adım deneyimle ne kadar iyi olduğunu gösterir. Örnek verimliliği karşılaştırmasının temeli.
- **Tohum:** RL sonuçları rastgele başlangıca çok bağlıdır. Tek tohumla "SAC daha iyi" demek yazı-tura atmaya yakın;
  bu yüzden 5 tohum.
- **Sonuca bakmak yetmez, davranışa bak:** iki politika aynı başarıyı alıp çok farklı davranabilir (biri titreyerek,
  biri yumuşak).

### 👀 Önce simüle et
- Eğitimden **önce** rastgele politikanın animasyonunu kaydet.
- Eğitim sırasında birkaç checkpoint'in animasyonunu izle (ör. %10, %50, %100). Ajanın neyi hangi sırayla öğrendiğini
  gör: önce düşmemeyi mi, sonra arabayı ortalamayı mı?

### 🎤 Mülakatta nasıl sorulur
- "RL politikan LQR'yi geçti mi? Neden ya da neden değil?"
- "Kaç tohum kullandın, tohumlar arası fark ne kadardı?"
- "Politikan neden titriyor, nasıl düzeltirsin?" → efor cezası, aksiyon yumuşatma.
- "Hiperparametreleri nasıl seçtin?"

### 📚 Öğren
| konu | kaynak |
|---|---|
| SB3 PPO ve SAC kullanımı, `VecNormalize`, callback'ler, TensorBoard | SB3 dokümanı → PPO, SAC, "Vectorized Environments", "Callbacks", "Tensorboard Integration" |
| Hiperparametre başlangıç değerleri | **RL Baselines3 Zoo** deposundaki `hyperparams/ppo.yml` ve `sac.yml` (sürekli kontrol ortamları, ör. Pendulum) |
| Deneylerin tekrar üretilebilirliği ve tohum varyansı | Henderson ve ark. (2018) "Deep Reinforcement Learning that Matters" |

### 💻 Yap
1. **Önce tahmin yaz:** PPO ve SAC kaç adımda LQR seviyesine yaklaşır? Hangisi daha çok efor (∫F²) harcar?
2. `train.py`: algoritma, tohum, adım sayısı, ortam ayarları komut satırından; her koşu kendi klasörüne
   (config + tohum + git hash + model + log). BVR'daki MLOps-lite alışkanlığı.
3. **Başlangıç bütçeleri** (değiştirebilirsin): PPO ~300k–1M adım, SAC ~50k–200k adım.
4. **5 tohum × 2 algoritma.** Periyodik değerlendirme (her N adımda geliştirme bloğunda deterministik oynatma).
5. Eğitilen politikaları Aşama 6 düzeneğinden geçir.

### ✅ Doğrula / analiz
- **Öğrenme eğrileri:** x ekseni ortam adımı, y ekseni geliştirme bloğu başarı oranı ve maliyet; tohumlar arası aralıkla.
- **Ortak tablo:** PID, LQR, MPC, LQG, PPO, SAC — aynı senaryolar. Çalışma anında adım başına hesap süresini de ekle
  (MPC: optimizasyon; RL: ağın bir ileri geçişi).
- **Davranış karşılaştırması:** aynı başlangıçtan LQR ve RL'nin F(t) ve θ(t) grafikleri. RL "titriyor" mu (bang-bang
  benzeri kontrol)? Neden?
- **Politika ile LQR kazancını karşılaştır:** RL politikasının denge noktası yakınındaki Jacobian'ı (∂a/∂s, sonlu farkla)
  LQR'nin −K'sına benziyor mu? (Çok öğretici bir test.)
- Tahminin tuttu mu? Tutmadıysa neden?

### 📦 Çıktı
Eğitilmiş modeller, öğrenme eğrileri, ortak karşılaştırma tablosu, "RL ile LQR kazancı" analizi.

### 🎯 Anladım ölçütü
_(sen yaz)_

---

## Aşama 10 — (İsteğe bağlı) PPO'yu elle yaz

**Süre:** 1–2 gün, **ana planın dışında.** Proje bittikten sonra ya da Proje 1'e geçmeden yapılabilir.
Zaman kutusu aşılırsa bırak, "denendi" diye not et.

### 💡 Sade dille
SB3'ün PPO'su bir kara kutu. Kendin yazınca içinde ne olduğunu görürsün: veri toplama, avantaj hesabı, kırpılmış kayıp,
mini-batch güncelleme. Kütüphanedeki küçük "ayrıntıların" (ör. avantaj normalizasyonu) performansı ne kadar
değiştirdiğini de bizzat ölçersin.

### 🎤 Mülakatta nasıl sorulur
- "Bir RL algoritmasını sıfırdan yazdın mı? En çok hangi ayrıntı fark yarattı?"

### 📚 Öğren
- **CleanRL** deposundaki `ppo_continuous_action.py` (tek dosya, ~300 satır) — referans olarak oku, kopyalama.
- "37 Implementation Details" yazısı (Aşama 8).

### 💻 Yap
- Kendi PPO'nu yaz (`ppo_scratch.py`): rollout toplama, GAE, kırpılmış kayıp, değer kaybı, mini-batch güncelleme,
  avantaj normalizasyonu, gözlem normalizasyonu.
- Aynı cartpole ortamında SB3 PPO ile karşılaştır (aynı bütçe, 3+ tohum).

### ✅ Doğrula
- Öğrenme eğrisi SB3 PPO'ya yakın mı? Değilse hangi ayrıntı eksik? (Ayrıntıları tek tek açıp kapatarak — ablasyon.)

### 🎯 Anladım ölçütü
_(sen yaz)_

---

## Aşama 11 — Sarkaç kaldırma: enerji + LQR hibrit, SAC

**Süre:** 1½–2 gün (Gün 8–10)

### 💡 Sade dille
Bu kez araba yok; sarkaç sabit bir mafsala bağlı ve mafsaldaki motor **zayıf** (en fazla 2 N·m). Sarkacı aşağıdan tek
hamlede kaldıramaz.

Benzetme: **salıncaktaki çocuk.** Salıncak doğru anlarda itilince her salınımda biraz daha yükselir.

**Enerji kontrolü** tam bunu yapar: "sarkacın enerjisi yukarıdaki dik konumun enerjisinden az mı? O zaman hareket
yönünde it." Enerji yeterince toplanınca sarkaç yukarıya yaklaşır ve orada **LQR devralır**. İki farklı denetleyicinin
duruma göre devreye girmesine **hibrit (anahtarlamalı) kontrol** denir.

**Lyapunov argümanı:** "enerji hatası hiçbir zaman büyümüyor" diyebilmenin matematiksel yolu. Bir büyüklüğün zamanla
hep azaldığını gösterirsen sistemin hedefe gittiğini kanıtlamış olursun.

**SAC** ise ikisini **tek bir ağla** öğrenir: ne zaman pompalayacağını ve ne zaman dengeleyeceğini ayrı ayrı
söylemene gerek kalmaz. Asıl karşılaştırma: tasarımcı emeği (klasik) ile eğitim emeği (RL).

### 👀 Önce simüle et
- Aşağıdan başlayıp sabit u = u_max uygula: sarkaç belli bir açıda takılır, yukarı çıkamaz.
- Sonra "her zaman hareket yönünde tam güçle it" kuralını dene: u = u_max·sign(θ̇). Her salınımda biraz daha yükseldiğini
  gör. Enerji kontrolü bu fikrin düzgün hâli.
- Enerjiyi zamana göre çiz: basamak basamak artıyor mu?

### 🎤 Mülakatta nasıl sorulur
- "Klasik çözümle RL'yi karşılaştırdın; hangisi ne kadar emek istedi, hangisi daha iyi çalıştı?"
- "Anahtarlamalı kontrolün riskleri neler?" → geçiş anında sıçrama, iki denetleyici arasında gidip gelme.
- "Lyapunov fonksiyonu nedir, burada ne işe yaradı?"
- "LQR neden tek başına sarkacı kaldıramıyor?"

### Ortam B: basit sarkaç (sabit pivot)
Kütle m, uzunluk l, pivotta tork u, |u| ≤ u_max. **u_max, sarkacı doğrudan kaldıramayacak kadar küçük seçilir**
(ör. m = 1, l = 1 → statik kaldırma için ~9.8 N·m gerekir; u_max = 2 N·m). Böylece sarkacın **sallanarak enerji
toplaması** zorunlu olur.

### 📚 Öğren
| konu | kaynak |
|---|---|
| Basit sarkaç dinamiği, faz portresi, enerji | ⭐ Tedrake, **"The Simple Pendulum"** bölümü (tamamı) |
| Enerji şekillendirme (energy shaping) ile kaldırma | Tedrake, "The Simple Pendulum" → **"Energy-shaping"** kısmı; ⭐ **Åström & Furuta (2000), "Swinging up a pendulum by energy control", *Automatica*** |
| Kaldırma + dengelemeye geçiş (hibrit / anahtarlamalı kontrol) | Tedrake, cart-pole bölümündeki "swing-up + LQR" kısmı; Åström & Furuta makalesinin tartışma kısmı |
| Açının sarılması, gözlemde sin/cos kullanımı | Gymnasium `Pendulum-v1` kaynak kodu ve dokümanı (yalnız tasarım fikri için; modeli kendin yaz) |

### 🧮 Bilgisayarla çöz (`notebooks/11_swingup.ipynb`)
1. Sarkaç denklemini Aşama 1'deki sympy Lagrange kalıbıyla türet (θ = 0 yukarı, isteğe bağlı viskoz sürtünme b).
2. Enerji E(θ, θ̇) ve **yukarı denge noktasının enerjisi E_top**.
3. **Ė'yi sympy ile hesapla** (dinamiği yerine koyarak, sürtünmesiz): Ė'nin u ile ilişkisi nedir?
4. Enerji hatası Ẽ = E − E_top için **Ẽ → 0 yapan bir kontrol yasası** öner. sympy ile d/dt(Ẽ²/2) ifadesini çıkar ve
   işaretinin ≤ 0 olduğunu göster (Lyapunov argümanı). Sayısal olarak da Ẽ(t) grafiğiyle doğrula.
5. **Faz portresi:** (θ, θ̇) düzleminde serbest sarkacın vektör alanını ve sabit enerji eğrilerini çiz; kaldırma
   denetleyicisinin yörüngesini üstüne bindir.
6. Yukarıda doğrusallaştır (sympy Jacobian), LQR tasarla.
7. **Geçiş kuralı:** ne zaman enerji kontrolünden LQR'ye geçilir? Sarkaç LQR'sinin çekim bölgesini (θ × θ̇ taraması,
   Aşama 4'teki gibi) çıkar ve eşiği bu haritadan seç.

> 🔍 Ek A.5 ile karşılaştır.

### 💻 Kodla
- `pendulum.py` (+ `test_all.py`'ye enerji korunumu testi, Aşama 1'deki gibi)
- `controllers.py`'ye ekle: `EnergySwingUp(k, u_max)`, `HybridSwingUpLQR(...)`
- `envs.py`'ye ekle: `PendulumSwingUpEnv` — gözlem [cos θ, sin θ, θ̇], aksiyon [−1, 1] → u; başlangıç aşağıda (θ₀ ≈ π + küçük gürültü)
  ve bir de "her yerden" dağılımı; ödül ör. −(θ_sarılı² + 0.1 θ̇² + 0.001 u²).
- SAC eğitimi, 5 tohum (Aşama 9 altyapısıyla).

### ✅ Doğrula
- Enerji kontrolüyle Ẽ(t) grafiği sıfıra gidiyor; u doymada.
- Hibrit denetleyici her başlangıçtan kaldırıp dengeliyor; geçiş anında sıçrama/titreşim var mı?
- **Karşılaştırma tablosu** (Aşama 6 düzeneğinin sarkaç versiyonu): başarı oranı, kaldırma süresi, ∫u², sallanma sayısı.
- **Yalnız LQR'yi aşağıdan başlat:** başarısız olduğunu göster (Aşama 4'teki çekim bölgesi sınırının ikinci kanıtı).
- **Tasarım maliyeti notu:** hibrit denetleyiciyi tasarlamak kaç saat sürdü, SAC'ı eğitmek kaç saat (insan + hesap)?

### ➕ Ana planın dışında: arabalı sarkaçta kaldırma
Cartpole'da kaldırma: Tedrake'deki **kısmi geri besleme doğrusallaştırması (partial feedback linearization) + enerji
şekillendirme**, ve yine SAC. Belirgin şekilde zor; yaparsan ayrı bir zaman kutusu koy.

### 📦 Çıktı
Enerji ve hibrit denetleyici, sarkaç ortamı, SAC modelleri, karşılaştırma tablosu, kaldırma animasyonları.

### 🎯 Anladım ölçütü
_(sen yaz)_

---

## Aşama 12 — Sağlamlık ve domain randomization

**Süre:** 1½ gün (Gün 10–11). DR eğitimleri arka planda koşarken klasik denetleyicilerin taramasını yap.

### 💡 Sade dille
Simülasyondaki parametreler gerçekte **hiçbir zaman** tam doğru değildir: kütle biraz farklıdır, sürtünme vardır,
motor bir-iki adım geç tepki verir. **Sağlamlık** şu soruyu sorar: model yanlışken kontrolcü hâlâ çalışıyor mu?
- **LQR** K'yı nominal modelle hesaplar. Gerçek sistem farklıysa A − BK'nın özdeğerleri kayar, sağ tarafa geçerse
  sistem kararsızlaşır.
- **Domain randomization (DR):** eğitimde her bölümde parametreleri rastgele değiştirmek. Ajan tek bir dünyaya değil,
  bir **dünyalar ailesine** uyum sağlar. Bedeli: nominal dünyada biraz daha temkinli (daha yavaş) davranabilir.
- Benzetme: hep aynı pistte antrenman yapan sürücü ile her gün farklı pistte, farklı havada antrenman yapan sürücü.

Bu, **simülasyondan gerçeğe geçişin** (sim-to-real) temel aracı ve savunma projelerinde en çok sorulan konulardan biri.

### 👀 Önce simüle et
- LQR'yi sarkaç kütlesi 2 katına çıkmış sistemde çalıştır ve animasyonu izle.
- Aynısını 2 adımlık eyleyici gecikmesiyle yap. Hangisi daha çok bozuyor? Tahminini önce `KARARLAR.md`'ye yaz.

### 🎤 Mülakatta nasıl sorulur
- "Simülasyonla gerçek arasındaki farkı nasıl azaltırsın?"
- "Domain randomization'ın bedeli ne? Hangi parametreleri hangi aralıkta rastgeleleştirdin, neden?"
- "Gecikme bir kontrol sistemini neden bozar?"
- "Eğitim aralığının dışındaki koşullarda ne oldu?"

### 📚 Öğren
| konu | kaynak |
|---|---|
| Domain randomization | Tobin ve ark. (2017) "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World" (fikir için giriş + yöntem) |
| ⭐ Dinamik parametre rastgeleleştirme | **Peng ve ark. (2018) "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization"** (ICRA) |
| Klasik tarafta sağlamlık: kazanç / faz payı, LQR'nin garanti payları, LQG'nin garantisi olmaması | Åström & Murray, "Robust Performance" bölümünün başı; Doyle (1978) "Guaranteed Margins for LQG Regulators" (yalnız özet — 1 sayfalık ünlü not) |

### 📝 Karar ver (`KARARLAR.md`) — **koşudan ÖNCE**
1. **Test ızgarası:** sarkaç kütlesi m ×{0.5, 0.75, 1, 1.5, 2}, uzunluk l ×{0.7, 1, 1.3}, araba kütlesi M ×{0.8, 1, 1.2},
   sürtünme {0, orta}, ölçüm gürültüsü {0, orta, yüksek}, eyleyici gecikmesi {0, 1, 2, 3 adım}.
   (Hepsinin çarpımı çok büyükse: her seferinde tek parametre değiştir + birkaç birleşik "kötü durum".)
2. **DR eğitim aralığı:** ör. m ×[0.6, 1.6], l ×[0.8, 1.2] … — **test ızgarasının bir kısmı bu aralığın dışında**
   kalsın (dış değerleme / extrapolation).
3. **Tahminler:** LQR hangi parametreye en hassas? Nominal RL ile LQR'den hangisi daha önce bozulur? DR nominal
   performanstan ne kadar kaybettirir?

### 💻 Kodla
- Ortama parametre rastgeleleştirme: her `reset()`'te parametreler aralıktan çekilir. Gecikme için aksiyon kuyruğu.
- PPO-DR ve SAC-DR eğitimi (5 tohum).
- `evaluate.py`'ye parametre ızgarası desteği.

### ✅ Doğrula / analiz
- **Sağlamlık tablosu / ısı haritası:** satırlar denetleyiciler (PID, LQR, MPC, LQG, PPO, SAC, PPO-DR, SAC-DR),
  sütunlar parametre sapmaları, hücrede başarı oranı (Wilson aralığıyla).
- **Nominal maliyet:** DR'li politikalar nominal koşulda ne kadar kötüleşti?
- **Mekanizma:** LQR'nin bozulduğu yerde neden bozulduğunu kapalı döngü özdeğerleriyle açıkla
  (yanlış modelle tasarlanmış K, gerçek A, B ile birleşince kutuplar nereye gidiyor?).
- Tahminlerin tuttu mu?

### 📦 Çıktı
Sağlamlık ısı haritası, DR modelleri, mekanizma açıklaması.

### 🎯 Anladım ölçütü
_(sen yaz)_

---

## Aşama 13 — Sonuç raporu

**Süre:** 1 gün (Gün 12)

### 💡 Sade dille
Rapor, yaptığın her şeyi **bir hikâyeye** bağlar: "Klasik kontrol şurada iyi, şurada yetersiz; RL bu boşluğun şu
kısmını doldurdu, bedeli de şu." Okuyan biri ilk paragrafta ne bulduğunu, sonraki bölümlerde nasıl bulduğunu görmeli.

### 🎤 Mülakatta nasıl sorulur
- "Bu projeyi 2 dakikada anlat." → README'nin ilk paragrafı bu cevap olmalı. Sesli prova et.
- "Tekrar yapsan neyi farklı yapardın?"

### 💻 Yap
1. **README (İngilizce):** problem, yöntemler, ana sonuç tabloları, grafikler, nasıl çalıştırılır.
2. **Sonuçlar bölümü — her biri bir cümle + kanıt:**
   - Denge görevinde kim kazandı, neden?
   - Çekim bölgesi sınırı ve kaldırma görevinde RL'nin katkısı.
   - Örnek verimliliği: PPO ve SAC.
   - Sağlamlık: DR'nin kazancı ve bedeli.
3. **Sınırlamalar:** simülasyon modeli, parametre aralıkları, tohum sayısı, gerçek donanım yok.
4. **Nihai test bloğu:** geliştirme sırasında hiç bakmadığın bloğu EN SONDA bir kez koş; tabloyu ondan raporla.
5. **Proje 1'e bağlantı:** Kalman / LQG burada başladı; güdüm projesinde IMM ve POMDP'ye dönüşecek.

### 🎯 Anladım ölçütü
_(sen yaz)_

---

## Gün planı

| gün | aşamalar | günün sonunda elinde olan |
|---|---|---|
| 1 | 0, 1 | Kurulum, sympy türetmesi, test edilmiş dinamik + RK4, animasyon |
| 2 | 2, 3 | A, B, özdeğer / kontrol edilebilirlik / RHP sıfırı; açı-PD ve kaskad PID |
| 3 | 4 | LQR (elle Riccati iterasyonu dahil), Q/R takası, **çekim bölgesi haritası** |
| 4 | 4b | MPC, LQR ile aynılık testi, LQR / MPC çekim bölgesi karşılaştırması |
| 5 | 5 | Kalman filtresi, LQG, gürültü tablosu |
| 6 | 6, 7 | Ortak değerlendirme düzeneği, Gymnasium ortamı, klasik denetleyicilerin ilk ortak tablosu |
| 7 | 8, 9 (başlangıç) | RL teori defteri; PPO ve SAC eğitimleri arka planda başlatıldı |
| 8 | 9, 11 (başlangıç) | RL sonuçları ve analizi; sarkaç modeli ve enerji kontrolü |
| 9 | 11 | Hibrit kaldırma + LQR, sarkaç ortamı, SAC eğitimi |
| 10 | 11 (bitiş), 12 (başlangıç) | Kaldırma karşılaştırması; DR kararları ve DR eğitimleri başlatıldı |
| 11 | 12 | Sağlamlık ısı haritası ve mekanizma analizi |
| 12 | 13 | Nihai test bloğu, README, sonuçlar |
| (+1) | tampon | Taşan işler |

**Toplam ~2,5 hafta.** Planın dışında: Aşama 10 (PPO'yu elle yazmak) ve arabalı sarkaçta kaldırma.

**Durma kuralı:** bir aşama kendi gününü iki kez aşarsa "yapılamadı, sebebi şu" yazılır ve geçilir.
**Paralellik:** RL eğitimleri saatler sürebilir — eğitim koşarken bir sonraki aşamanın okuması / klasik kısmı yapılır.

---

## Ek A — Referans denklemler (sympy sonucunu kontrol için)

Kural: θ = 0 yukarı, θ > 0 sağa yatık; x > 0 sağ; F > 0 sağa. Noktasal sarkaç kütlesi, kütlesiz çubuk, sürtünme yok.

### A.1 Cartpole hareket denklemleri

Sarkaç kütlesinin konumu: (x + l sin θ, l cos θ)

T = ½(M + m)ẋ² + m l ẋ θ̇ cos θ + ½ m l² θ̇²
V = m g l cos θ

Euler–Lagrange:
- (M + m) ẍ + m l θ̈ cos θ − m l θ̇² sin θ = F
- m l ẍ cos θ + m l² θ̈ − m g l sin θ = 0

ẍ ve θ̈ için çözüm:

ẍ = [F + m sin θ (l θ̇² − g cos θ)] / (M + m sin² θ)

θ̈ = [−F cos θ − m l θ̇² sin θ cos θ + (M + m) g sin θ] / [l (M + m sin² θ)]

Toplam enerji: E = T + V

> Not: Gymnasium `CartPole-v1` dağılmış kütleli çubuk (yarım uzunluk, atalet momenti) kullanır; denklemleri farklı
> görünür. Kaynaklar arasında θ kuralı (yukarı / aşağı) ve sönüm terimleri de değişir.

### A.2 Doğrusallaştırma (s = [x, ẋ, θ, θ̇], yukarı denge)

```
A = [[0, 1,        0,           0],
     [0, 0,     −m g / M,       0],
     [0, 0,        0,           1],
     [0, 0, (M + m) g / (M l),  0]]

B = [0, 1/M, 0, −1/(M l)]ᵀ
```

Özdeğerler: 0, 0, ±√((M + m) g / (M l)).
F → x transfer fonksiyonunun sıfırları: ±√(g / l) → sağ yarı düzlemde bir sıfır (minimum fazlı değil).

### A.3 LQR

Sürekli: J = ∫ (sᵀ Q s + uᵀ R u) dt
AᵀP + PA − PBR⁻¹BᵀP + Q = 0, K = R⁻¹BᵀP, u = −K s

Ayrık: J = Σ (s_kᵀ Q s_k + u_kᵀ R u_k)
P = AᵀPA − AᵀPB (R + BᵀPB)⁻¹ BᵀPA + Q, K = (R + BᵀPB)⁻¹ BᵀPA, u_k = −K s_k

Optimal maliyet (değer fonksiyonu): V(s) = sᵀ P s

### A.4 Kalman filtresi (ayrık)

Tahmin: x̂⁻ = A x̂ + B u, P⁻ = A P Aᵀ + Q_kf
Kazanç: K = P⁻ Hᵀ (H P⁻ Hᵀ + R_kf)⁻¹
Güncelleme: x̂ = x̂⁻ + K (z − H x̂⁻), P = (I − K H) P⁻

Cartpole için (x ve θ ölçülüyor):
```
H = [[1, 0, 0, 0],
     [0, 0, 1, 0]]
```

### A.5 Basit sarkaç ve enerji ile kaldırma

θ = 0 yukarı, pivotta tork u, sürtünme b:
m l² θ̈ = m g l sin θ − b θ̇ + u

Enerji: E = ½ m l² θ̇² + m g l cos θ, E_top = m g l
Sürtünmesiz: Ė = θ̇ u

Enerji kontrolü: u = −k θ̇ Ẽ (Ẽ = E − E_top), doymalı → d/dt Ẽ = −k θ̇² Ẽ, Ẽ → 0 (θ̇ = 0 anlarında anlık durur)

Yukarıda doğrusal model: θ̈ = (g / l) θ − (b / (m l²)) θ̇ + u / (m l²)

---

## Ek B — Parametreler ve kurallar

| parametre | cartpole | sarkaç |
|---|---|---|
| araba kütlesi M | 1.0 kg | — |
| sarkaç kütlesi m | 0.1 kg | 1.0 kg |
| uzunluk l | 0.5 m | 1.0 m |
| g | 9.81 m/s² | 9.81 m/s² |
| girdi sınırı | F_max = 10 N | u_max = 2 N·m |
| kontrol periyodu dt | 0.02 s (50 Hz) | 0.02 s |
| entegrasyon | RK4, kontrol adımı içinde gerekirse alt adımlar | RK4 |
| bölüm süresi | 10 s (500 adım) | 10 s (500 adım) |
| başarı sınırları | |θ| < 12°, |x| < 2.4 m | son 2 s'de |θ| < 0.1 rad |

Bunlar başlangıç önerisi; değiştirirsen `KARARLAR.md`'ye yaz.

---

## Ek C — Kaynak listesi

### Kitaplar
| kaynak | erişim | kullanıldığı aşamalar |
|---|---|---|
| ⭐ R. Tedrake, *Underactuated Robotics* | ücretsiz, underactuated.mit.edu | 1, 4, 11 |
| K. J. Åström & R. M. Murray, *Feedback Systems* | bilgisayarında (`Tez-RL-Gemini/Kontrol - Claude Programı/`); ücretsiz PDF yazarların sitesinde | 1, 2, 3, 4, 5, 12 |
| ⭐ R. Labbe, *Kalman and Bayesian Filters in Python* | ücretsiz, GitHub (Jupyter defterleri) | 5 |
| R. Sutton & A. Barto, *Reinforcement Learning: An Introduction* (2. baskı) | bilgisayarında | 7, 8 |
| S. Brunton & J. N. Kutz, *Data-Driven Science and Engineering* | doğrusal kontrol bölümü | 2, 4 |
| Modern Kontrol ders notların | bilgisayarında | 2, 4 |

### Video
| kaynak | aşamalar |
|---|---|
| ⭐ Steve Brunton, *Control Bootcamp* (YouTube oynatma listesi) | 1, 2, 4, 5 |
| David Silver, UCL RL dersleri (isteğe bağlı) | 8 |

### Web
| kaynak | aşamalar |
|---|---|
| ⭐ OpenAI *Spinning Up in Deep RL* (spinningup.openai.com) | 8 |
| ⭐ "The 37 Implementation Details of PPO" (ICLR 2022 Blog Track) | 8, 10 |
| Gymnasium dokümanı: "Create a Custom Environment" | 7 |
| Stable-Baselines3 dokümanı: "RL Tips and Tricks", PPO, SAC, VecNormalize | 7, 9 |
| RL Baselines3 Zoo (hiperparametre dosyaları) | 9 |
| CleanRL (`ppo_continuous_action.py`) | 10 |
| python-control dokümanı (`lqr`, `dlqr`, `ss2tf`, `c2d` / `sample_system`) | 2, 4 |
| cvxpy dokümanı ("Control" örneği, `Parameter` kullanımı) | 4b |
| Brunton, "Model Predictive Control" videosu | 4b |
| Borrelli, Bemporad, Morari, *Predictive Control for Linear and Hybrid Systems* (ücretsiz PDF) | 4b |

### Makaleler
| makale | aşama |
|---|---|
| Schulman ve ark. (2017), *Proximal Policy Optimization Algorithms* | 8 |
| Schulman ve ark. (2016), *High-Dimensional Continuous Control Using Generalized Advantage Estimation* | 8 |
| Haarnoja ve ark. (2018), *Soft Actor-Critic* (+ "Algorithms and Applications") | 8 |
| Henderson ve ark. (2018), *Deep Reinforcement Learning that Matters* | 9 |
| Agarwal ve ark. (2021), *Deep RL at the Edge of the Statistical Precipice* | 6, 9 |
| Åström & Furuta (2000), *Swinging up a pendulum by energy control*, Automatica | 11 |
| Tobin ve ark. (2017), *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World* | 12 |
| Peng ve ark. (2018), *Sim-to-Real Transfer of Robotic Control with Dynamics Randomization* | 12 |
| Doyle (1978), *Guaranteed Margins for LQG Regulators* | 12 |

---

## Ek D — Bilinen tuzaklar

| tuzak | belirti | önlem |
|---|---|---|
| Açı kuralı / işaret karışıklığı | Denetleyici sarkacı daha hızlı düşürüyor | Ek B kuralı; Aşama 1 fiziksel sezgi testleri |
| Kaynaklar farklı kural kullanıyor (Tedrake: θ=0 aşağı; Brunton: sönüm terimi) | Kopyalanan denklem çalışmıyor | Her denklemi kendi kuralına çevir, sympy ile kontrol et |
| Euler entegrasyonu | Enerji büyüyor, sarkaç "kendiliğinden" hızlanıyor | RK4 + enerji testi |
| Sürekli K'yı ayrık sistemde kullanmak | Büyük dt'de kararsızlık | Ayrık LQR ya da dt'nin yeterince küçük olduğunu göster |
| Doymayı unutmak | LQR simülasyonda her açıdan kurtarıyor | Tüm denetleyicilerde aynı F_max |
| LQR ve Kalman'ın Q/R'sini karıştırmak | Yanlış matris yanlış yerde | `Q_lqr`, `Q_kf` gibi ayrı isimler |
| `terminated` ile `truncated` karıştırmak | Değer fonksiyonu süre sonunu "ölüm" sanıyor | Gymnasium API'sini doğru kullan |
| Ödül sömürüsü | Ajan sınırda bekliyor ya da erken bitiriyor | Ödülü maliyetten türet; davranışı mutlaka görselleştir |
| Aksiyonu ölçeklememek | SAC/PPO kötü öğreniyor | Aksiyon [−1, 1], ortam içinde ölçekleme |
| `VecNormalize` istatistiklerini kaydetmemek | Değerlendirmede model kötü görünüyor | Modelle birlikte istatistikleri kaydet; değerlendirmede güncellemeyi kapat |
| Tek tohum | "SAC PPO'dan iyi" gibi yanlış hüküm | 5 tohum, aralıkla raporla |
| Açıyı sarmamak (sarkaç) | Ajan 2π fazla dönüşü ceza sanıyor | Gözlemde sin/cos, maliyette sarılı açı |
| MPC problemini her adımda yeniden kurmak | Adım başına çözüm süresi dt'yi aşıyor | Problemi bir kez kur, başlangıç durumunu `cvxpy.Parameter` ile güncelle |
| Nihai test bloğuna erken bakmak | Farkında olmadan ona göre ayar yapmak | Nihai blok yalnız Aşama 13'te |
