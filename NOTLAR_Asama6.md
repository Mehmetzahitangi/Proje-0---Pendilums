# Aşama 6 Ders Notları: Ortak Değerlendirme Düzeneği

> Her bölüm: **Amaç → Sezgi → Yöntem → Kodda nerede → Mülakatta ne derim.**
> Kod: `evaluate.py` (`senaryolar`, `olcutler`, `wilson`, `degerlendir`, `ozet`, `csv_yaz`, `eslestirilmis`), testler:
> `test_all.py` (Aşama 6), deney: `notebooks/06_evaluate.ipynb`, kararlar: `KARARLAR.md` → Aşama 6.

---

## İçindekiler
1. [Neden ayrı bir düzenek, neden RL'den önce?](#1-neden-ayrı-bir-düzenek-neden-rlden-önce)
2. [Adil karşılaştırmanın dört ilkesi](#2-adil-karşılaştırmanın-dört-ilkesi)
3. [Kararlar](#3-kararlar)
4. [Wilson güven aralığı](#4-wilson-güven-aralığı)
5. [Eşleştirilmiş karşılaştırma ve bootstrap](#5-eşleştirilmiş-karşılaştırma-ve-bootstrap)
6. [Kod yapısı](#6-kod-yapısı)
7. [İlk ortak tablo ve okunuşu](#7-ilk-ortak-tablo-ve-okunuşu)
8. [Kaybedilen senaryolar: kararsız mod koordinatı](#8-kaybedilen-senaryolar-kararsız-mod-koordinatı)
9. [Mülakat soruları ve örnek cevaplar](#9-mülakat-soruları-ve-örnek-cevaplar)
10. [Tuzaklar](#10-tuzaklar)
11. [Kaynak haritası](#11-kaynak-haritası)
12. [Kendini sına](#12-kendini-sına)

---

## 1. Neden ayrı bir düzenek, neden RL'den önce?

Aşama 3–5'te her karşılaştırma o anki soruya göre farklı bir senaryo, süre, hatta başarı tanımıyla yapıldı. Aşama 9'un
sorusu ("RL klasiklerden iyi mi?") için hepsini **aynı terazide** tartmak gerekiyor.

**RL'den önce kurulmasının sebebi:** RL sonuçlarını gördükten sonra teraziyi ayarlamak, farkında olmadan RL'yi (ya da
klasikleri) kayıracak şekilde ayarlamak demek. Ölçüt önce, sonuç sonra.

---

## 2. Adil karşılaştırmanın dört ilkesi

| ilke | nasıl uygulandı | neden |
|---|---|---|
| **Aynı senaryolar (ortak rastgele sayılar)** | her senaryo: başlangıç durumu + kendi `sensor_seed`'i; her kontrolcü aynısını görüyor | iki öğrenciyi aynı sınavla karşılaştırmak |
| **Sabit tohumlu bloklar** | geliştirme (seed 0, 50) ayar için; **nihai (seed 1000, 200) en sonda bir kez** | teste göre ayar yapmamak (train / val / test) |
| **Aynı başarı tanımı ve ölçütler** | tanım koddan önce `KARARLAR.md`'de | Aşama 4.4: tanım değişince AnglePD 42° → 5° |
| **Belirsizliği raporla** | Wilson aralığı, eşleştirilmiş fark + bootstrap | az örnekle "A > B" dememek |

---

## 3. Kararlar

| karar | seçim |
|---|---|
| Başarı | sonlu; bütün bölüm boyunca \|x\| ≤ 2.4; sonda \|θ\| < 0.05, \|θ̇\| < 0.1 |
| Ölçütler | başarı, oturma (x ve θ ±0.02'ye girip bir daha çıkmadığı an, **saniye**; başarısızsa NaN), efor ∫F² dt, θ RMS, en büyük \|x\| (özette **sadece başarılılar**) |
| Bölüm | 10 s, dt = 0.02, substeps = 5 |
| Bloklar | **nominal**: x ±0.5, ẋ ±0.1, θ ±0.2, θ̇ ±0.1. **zor**: x ±1, ẋ ±0.5, θ ±0.6, θ̇ ±1 (düzgün dağılım) |
| Boyut | geliştirme 50 (seed 0), nihai 200 (seed 1000) |
| Sensör | tam durum ve z = [x, θ], σ = 0.005 |

**Zor blok neden eklendi?** Nominalde tam durumlu dört kontrolcünün hepsi 50/50'ydi. **Bir test, test ettiği şeyleri
ayırabilmeli.** Nominal "temel işlevsellik" testi olarak kaldı; karşılaştırmanın yükünü zor blok taşıyor (ROA kenarı).

---

## 4. Wilson güven aralığı

$$ \frac{\hat p + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}},\qquad \hat p = k/n,\ z = 1.96 $$

**Basit formül (p̂ ± 1.96√(p̂(1−p̂)/n)) neden değil?** p̂ = 1 iken [1, 1] verir: "kesinlikle %100". Wilson 50/50 için
**[0.93, 1.00]** der: "en az %93". Başarı oranları sık sık 0'a ya da 1'e yakın olduğu için önemli.

| k/n | Wilson |
|---|---|
| 50/50 | [0.93, 1.00] |
| 45/50 | [0.79, 0.96] |
| 42/50 | [0.71, 0.92] |
| 21/50 | [0.29, 0.56] |

---

## 5. Eşleştirilmiş karşılaştırma ve bootstrap

Aynı senaryolarda koşturulduğu için iki kontrolcü **senaryo senaryo** karşılaştırılabilir: d_i = B_i − A_i (ikisinin de
başarılı olduğu senaryolarda). Senaryo zorluğundan gelen değişkenlik iki tarafta aynı olduğu için çıkarmada yok oluyor.
Ayrı ayrı ortalamaları karşılaştırmaktan **çok daha hassas.** Ortak rastgele sayıların asıl ödülü.

**Bootstrap aralığı:** d'den yerine koyarak n'lik örneklemler çek, her birinin ortalamasını al, 2000 kez tekrarla; %2.5 ve
%97.5 yüzdelikleri ortalama farkın %95 aralığı. Dağılım varsayımı gerektirmiyor.

Ayrıca **başarı çapraz tablosu:** ikisi de başarılı / sadece A / sadece B. Bir kontrolcünün diğerinin kaybettiğini
kurtarıp kurtarmadığını gösteriyor.

---

## 6. Kod yapısı

| fonksiyon | görevi | dikkat |
|---|---|---|
| `senaryolar(dagilim, n, seed)` | tekrarlanabilir blok; her senaryoya `s0` ve `sensor_seed` | aynı tohum → aynı blok (test) |
| `olcutler(t, S, U)` | bir bölümün ölçütleri (dict) | **oturma = t[indis]**, başarısızsa NaN (test) |
| `wilson(k, n)` | %95 aralık | bilinen değer testi |
| `degerlendir(ad, yap, blok, sensor_sigma, params)` | blokta koşturur, satır listesi | `yap` her senaryoda taze kontrolcü; `params` Aşama 12 için hazır |
| `ozet`, `ozet_yazdir` | başarı + Wilson, başarılılarda medyanlar | `dict.fromkeys` ile sıra korunur |
| `csv_yaz` | senaryo başına satırları dosyaya | Aşama 9'da RL ile birleştirilecek |
| `eslestirilmis(satirlar, a, b, olcut)` | çapraz tablo + fark ortalaması + bootstrap aralığı | |

**Testler:** senaryo tekrarlanabilirliği, Wilson bilinen değerler, oturma saniye/NaN, `degerlendir` gürültülü sensörle
bile tekrarlanabilir.

---

## 7. İlk ortak tablo ve okunuşu

**Nominal (geliştirme, 50):**
| kontrolcü | başarı | Wilson | oturma med | efor med |
|---|---|---|---|---|
| Kaskad PID | 50/50 | [0.93, 1.00] | 3.02 s | 5.7 |
| LQR Bryson | 50/50 | [0.93, 1.00] | 2.15 s | 2.9 |
| LQR R×10 | 50/50 | [0.93, 1.00] | 2.31 s | 1.7 |
| MPC N=10 | 50/50 | [0.93, 1.00] | 2.15 s | 2.9 |
| Sonlu fark σ.005 | 21/50 | [0.29, 0.56] | 10.00 s | 567.8 |
| LQG σ.005 | 50/50 | [0.93, 1.00] | 2.85 s | 9.8 |

- Bryson = MPC birebir: kısıtlar aktif değil → MPC = LQR (Aşama 4b), 50 senaryoda.
- LQG, Bryson ile aynı başarı. Gürültünün bedeli ~0.7 s ve 3× efor (ayrılma ilkesi pratikte).
- Sonlu fark düşmüyor, **titriyor**: θ̇ hiç "sakin" olmuyor. Başarı tanımındaki θ̇ şartının sebebi.

**Zor (geliştirme, 50):**
| kontrolcü | başarı | Wilson | oturma med | efor med |
|---|---|---|---|---|
| Kaskad PID | 45/50 | [0.79, 0.96] | 3.72 s | 38.6 |
| LQR Bryson | 42/50 | [0.71, 0.92] | 2.49 s | 19.0 |
| LQR R×10 | 45/50 | [0.79, 0.96] | 2.80 s | 14.1 |
| MPC N=10 | 43/50 | [0.74, 0.93] | 2.54 s | 19.0 |
| Sonlu fark σ.005 | 15/50 | [0.19, 0.44] | 9.96 s | 577.0 |
| LQG σ.005 | 42/50 | [0.71, 0.92] | 2.91 s | 23.1 |

- **Başarı oranları ayırt edilemiyor:** Wilson aralıkları üst üste. 50 senaryoyla "45 > 42" denemez.
- **Eşleştirilmiş, R×10 − Kaskad:** oturma **−1.02 s [−1.17, −0.86], 45/45 senaryoda** daha hızlı; efor **−12.9 [−17.5, −8.2]**,
  %87'sinde daha az. İkisi de **aynı 5 senaryoyu** kaybediyor.
- **Sonuç:** başarıda fark yok; hız ve eforda R × 10 yüksek güvenle daha iyi.
- **max |x|** başlangıçta bütün bölümlerden hesaplanınca 389 m çıktı (raydan kaçan başarısızlar). Özet ölçütler
  **başarılılar** üzerinden; başarısızlık zaten başarı sütununda.

---

## 8. Kaybedilen senaryolar: kararsız mod koordinatı

| senaryo | θ₀ | θ̇₀ | \|θ + θ̇/λ\| |
|---|---|---|---|
| 0 | −31.6° | −0.97 | 0.76 |
| 4 | −32.4° | −0.75 | 0.73 |
| 21 | +32.5° | +0.78 | 0.74 |
| 39 | +29.3° | +0.50 | 0.62 |
| 48 | +28.4° | +0.53 | 0.61 |

**θ + θ̇/λ** (λ = 4.65): başlangıçta Aşama 2'deki kararsız modun (e^{+λt}) payı = 2B ROA haritasındaki şeridin merkez
çizgisinden (θ̇ = −λθ) uzaklık. **Tek eşik başarıyı tamamen ayırıyor:** başarılıların hepsi ≤ 0.587, başarısızların hepsi
≥ 0.61. ~30° yatık ve aynı yöne devrilen sarkaçlar. Aşama 9'da RL'nin bu 5 senaryodan birini kurtarıp kurtaramadığına
bakılacak.

---

## 9. Mülakat soruları ve örnek cevaplar

**S1. Sonuçlarının tesadüf olmadığını nasıl gösterdin?**
Aynı tohumlu senaryo bloklarında, aynı sensör gürültüsü dizileriyle koşturdum. Başarı oranlarını Wilson aralıklarıyla
raporladım; zor blokta 42 ile 45 ayırt edilemiyordu. Eşleştirilmiş karşılaştırma LQR'nin kaskaddan 1.0 s daha hızlı
oturduğunu gösterdi (%95 aralık [−1.17, −0.86], 45 senaryonun hepsinde).

**S2. Karşılaştırmaların adil miydi?**
Her kontrolcü aynı başlangıç durumları, aynı gürültü dizileri, aynı kuvvet sınırı, aynı süre ve koddan önce yazılmış
aynı başarı tanımıyla sınandı. Her senaryoda taze kontrolcü nesnesi (hafıza sızmıyor).

**S3. Deneylerin tekrar üretilebilir mi?**
Senaryolar ve sensör gürültüsü tohumlu; aynı tohum → birebir aynı sonuç. Bunu bir testle doğruluyorum. Farklı
oturumlarda aynı tablo çıktı.

**S4. Neden iki blok (nominal / zor)?**
Nominalde herkes %100'dü, test ayırt etmiyordu. Zor blok çekim bölgesinin kenarını yokluyor; klasik kontrolcülerin
gerçekten farklılaştığı ve RL'nin fark yaratabileceği yer orası.

**S5. Geliştirme ve nihai blok neden ayrı?**
Ayar yaparken baktığım bloğa göre farkında olmadan aşırı uyum sağlarım. Nihai blok en sonda bir kez bakılmak üzere
saklı. ML'deki test kümesi disiplini.

---

## 10. Tuzaklar

| tuzak | belirti | önlem |
|---|---|---|
| Oturma süresinde indis kullanmak | "99 s" | `t[indis]`; birim testi |
| Başarısızlıkta oturma = 0 ya da 10 | ortalamalar sessizce bozulur | NaN + `nanmedian` |
| Özet ölçütü başarısızlardan hesaplamak | max\|x\| = 389 m | özet sadece başarılılarda |
| Tek senaryo / tek seed sonucu | şans | blok + Wilson + eşleştirme |
| Her kontrolcüye farklı gürültü | haksız karşılaştırma | senaryoya bağlı `sensor_seed` |
| Herkesin %100 aldığı blok | ayırt etmiyor | zor blok |
| Nihai bloğa erken bakmak | teste göre ayar | kilitli, sona saklı |
| Kontrolcü nesnesini senaryolar arasında paylaşmak | hafıza sızıntısı | `yap()` ile taze nesne |

---

## 11. Kaynak haritası

| konu | kaynak |
|---|---|
| Ortak rastgele sayılar, eşleştirilmiş karşılaştırma | BVR projendeki `scripts/eval_commander.py`, `REQUIREMENTS.md` EVAL |
| Wilson aralığı | Wikipedia "Binomial proportion confidence interval" → Wilson score interval |
| Bootstrap | herhangi bir istatistik kaynağı (yeniden örnekleme) |
| Az tohumla RL raporlamak | Agarwal ve ark. (2021) "Deep RL at the Edge of the Statistical Precipice"; `rliable` |

---

## 12. Kendini sına

1. 200 senaryoda 190 başarı için Wilson aralığını hesapla. 50 senaryoda 47.5'luk orana göre ne kadar dar?
2. Neden eşleştirilmiş karşılaştırma ayrı ortalamalardan daha hassas? Bir cümlede.
3. Bir kontrolcü 50/50 alıyorsa "her zaman başarılı" diyebilir misin? Ne diyebilirsin?
4. Aynı 5 senaryoyu dört farklı kontrolcü kaybediyorsa bu ne düşündürür?
5. θ + θ̇/λ neden başarıyı ayıran bir sayı? Aşama 2'deki hangi kavram?
6. Nihai bloğa projenin ortasında bakarsan hangi hatayı yaparsın?
7. Sonlu farklı LQR neden sarkacı düşürmeden "başarısız" sayıldı?
8. RL ajanını bu düzenekle değerlendirmek için ajan hangi arayüzü sağlamalı?
