# Aşama 7 Ders Notları: Gymnasium Ortamı ve Ödül Tasarımı

> Her bölüm: **Amaç → Sezgi → Karar → Kodda nerede → Mülakatta ne derim.**
> Kod: `envs.py` (`CartPoleContinuousEnv`), testler: `test_all.py` (Aşama 7), deney: `notebooks/07_env.ipynb`,
> kararlar: `KARARLAR.md` → Aşama 7. Önceki aşamalarla bağ: fizik Aşama 1, maliyet Aşama 4, değerlendirme Aşama 6.

---

## İçindekiler
1. [Amaç: simülatörü RL'nin dilinde konuşturmak](#1-amaç-simülatörü-rlnin-dilinde-konuşturmak)
2. [MDP ve bizim karşılıklarımız](#2-mdp-ve-bizim-karşılıklarımız)
3. [Gymnasium arayüzü](#3-gymnasium-arayüzü)
4. [terminated ile truncated](#4-terminated-ile-truncated)
5. [Gözlem ve aksiyon uzayları](#5-gözlem-ve-aksiyon-uzayları)
6. [Ödül tasarımı](#6-ödül-tasarımı)
7. [Kod: CartPoleContinuousEnv](#7-kod-cartpolecontinuousenv)
8. [Doğrulama](#8-doğrulama)
9. [Alt referanslar ve ödül ≠ ölçüt](#9-alt-referanslar-ve-ödül--ölçüt)
10. [Mülakat soruları ve örnek cevaplar](#10-mülakat-soruları-ve-örnek-cevaplar)
11. [Tuzaklar (gerçekten yaşananlar)](#11-tuzaklar-gerçekten-yaşananlar)
12. [Kaynak haritası](#12-kaynak-haritası)
13. [Kendini sına](#13-kendini-sına)

---

## 1. Amaç: simülatörü RL'nin dilinde konuşturmak

RL kütüphaneleri (Stable-Baselines3) `cartpole.py`'yi tanımıyor; tanıdıkları tek şey **Gymnasium arayüzü**. Bu aşamada
simülatörü bu arayüzle **sarmaladık**. Fizik yeniden yazılmadı: Aşama 1'in `rk4_step`'i kullanılıyor.

Gymnasium'un `CartPole-v1`'i ayrık aksiyonlu (±10 N) ve dağılmış kütleli çubuk kullanıyor. Bizimki sürekli kuvvetli ve
kendi türettiğimiz, test edilmiş model. Klasik kontrolcülerle adil karşılaştırma için bu şart.

---

## 2. MDP ve bizim karşılıklarımız

| RL terimi | anlamı | bizde |
|---|---|---|
| durum / gözlem (s / o) | ajanın gördüğü | [x, ẋ, θ, θ̇], isteğe bağlı gürültülü |
| aksiyon (a) | ajanın kararı | a ∈ [−1, 1] → F = a · 10 N |
| geçiş | aksiyondan sonra ne oluyor | RK4, dt = 0.02, substeps = 5 |
| ödül (r) | "bu adım ne kadar iyiydi" | r = 1/(1 + c) |
| bölüm | reset'ten bitişe | en fazla 500 adım |
| politika π(a\|o) | karar kuralı | RL'nin öğreneceği şey. **LQR de bir politika:** a = −K·o / 10 |
| γ (indirim) | geleceğe verilen ağırlık | Aşama 9'da seçilecek. Aşama 4.2'den: ters tepki süresini (~25 adım) kapsamalı → γ ≥ 0.98 |

**Kontrol ↔ RL sözlüğü:** maliyet ↔ (eksi) ödül, kontrolcü ↔ politika, değer fonksiyonu V(s) = sᵀPs ↔ öğrenilen V(s),
Riccati iterasyonu ↔ değer iterasyonu (Aşama 4.2).

---

## 3. Gymnasium arayüzü

- `reset(seed=None, options=None)` → `(gözlem, info)`: oyunu baştan başlat.
- `step(aksiyon)` → `(gözlem, ödül, terminated, truncated, info)`: bir adım oyna.
- `observation_space`, `action_space`: gözlemin ve aksiyonun biçimi (tip, boyut, sınırlar).

**Tohumlama kuralı:** `super().reset(seed=seed)` çağrısı `self.np_random`'ı tohumluyor. **Bütün rastgelelik** (başlangıç
durumu, gözlem gürültüsü) buradan gelmeli → aynı tohum, aynı bölüm (Aşama 6'nın tekrarlanabilirlik ilkesi).

**`options={"s0": ...}`:** belirli bir başlangıç durumunu kurmak için kapı. Aşama 9'da RL'yi Aşama 6'nın **aynı
senaryolarında** değerlendirmek için.

---

## 4. terminated ile truncated

| bayrak | anlamı | bizde | değer hesabında |
|---|---|---|---|
| **terminated** | oyun **kaybedildi**, gelecek yok | \|x\| > 2.4 ya da \|θ\| > π/2 | V(s') = 0 |
| **truncated** | süre **doldu**, gelecek vardı | 500 adım | V(s') **tahmin edilir** (bootstrap) |

**Neden önemli?** Bellman: V(s) = r + γ·V(s'). Algoritma "bitti" durumunda V(s') = 0 koyar. Süre dolmasını ölüm sanarsa,
500. adımdaki mükemmel dengedeki bir durumun "geleceği yok" diye değersiz olduğunu öğrenir ve son adımlara doğru garip
davranır. Gymnasium ikisini ayırıyor; doğru bayrağı kaldırmak ortamın sorumluluğu.

**Neden 12° değil π/2?** Yol haritasının örneği 12°'ydi. Ama zor blokta 34°'ye kadar başlangıç var; 12°'de bitirseydik
bu senaryolar daha ilk adımda "kaybedilirdi". π/2 = sarkaç yatayın ötesine devrildi, dengeleme görevi bitti.

---

## 5. Gözlem ve aksiyon uzayları

### Aksiyon: Box(−1, 1)
- Ağlar ve RL algoritmaları ~1 ölçeğini sever. SAC aksiyonu tanh ile [−1, 1]'e sıkıştırır; PPO'nun başlangıç
  gürültüsü ~1.
- **Ölçeklemeyi ortam yapar** (F = a · F_max): ajanın öğrenmesine gerek olmayan bir sistem bilgisi.
- Motor sınırı ortamda (kırpma), Aşama 3 kararıyla tutarlı: sınırı unutan bir ajan haksız avantaj kazanamaz.
- [−1, 1] **gerçek** bir sınır (ortam kırpıyor), o yüzden yazmak doğru ve zorunlu.

### Gözlem: Box(−∞, +∞), float32
**`observation_space` bir sözdür:** "her gözlem bu kutunun içinde olacak." Bizde tutulamaz:
- Bölümü bitiren son gözlem tanım gereği sınırın dışında (x = 2.41, θ = 1.58).
- Hızların doğal sınırı yok (Aşama 1'de serbest düşüşte θ̇ ≈ 9.2 rad/s).
- Gauss gürültüsü teoride sınırsız.

`check_env` bunun için uyarı veriyor ("probably too low/high"). Bu sezgisel bir uyarı: çoğu zaman "yazar sınırları
düşünmemiş" işareti. Alternatif olarak **geniş fiziksel sınırlar** (x ±3, ẋ ±20, θ ±π, θ̇ ±30) da savunulabilir; o zaman
fizikle gerekçelendirilmeli. Tercihimiz ±∞, çünkü PPO/SAC sürekli gözlemlerde sınırları kullanmıyor. **Asıl önemli olan
sınır değil ölçek**; doğru araç `VecNormalize` (Aşama 9'da değerlendirilecek).

---

## 6. Ödül tasarımı

**İlke:** Ödül, RL'ye ne istediğini söylemenin **tek** yolu. Yanlış tasarlanırsa ajan kuralın boşluğunu bulur
(**ödül sömürüsü**, *reward hacking*).

Başlangıç noktası LQR maliyeti: c = sᵀQs + R·F² (Bryson Q = diag[4, 1, 25, 1], R = 0.01; her terim "izin verilen en büyük"
değerde 1). RL ödülü **maksimize** eder. Ama **bölüm erken bitebiliyor**, ve bu her şeyi değiştiriyor.

| ödül | "kötü ama hayatta": 500 adım θ ≈ 0.3 rad (c = 2.25) | "intihar": tam güç sağa, ~19 adımda raydan çık | LQR (20 bölüm ort.) |
|---|---|---|---|
| **A. r = −c·dt** | **−22.5** | **−20.2** | −6.5 |
| **B. r = 1 − min(c, 2)/2** | **0** | **0** | 421 |
| **C. r = 1/(1 + c)** ✅ | **+154** | +1 | 426 |

- **A, "erken bitirip cezadan kaçma":** ödül hep negatif. Kötü ama hayatta kalan bir politika ölmekten **kötü** puan
  alıyor. Eğitimin başında, ajan dengeleyemezken, öğreneceği en "akıllı" şey kendini hızla öldürmek.
- **B, sinyalsiz bölge:** kırpma yüzünden dikten uzakta ödül 0, ölmekle aynı. "Kötüden biraz daha az kötüye" gitmenin
  değeri görünmüyor. Zor başlangıçlar tam bu bölgede.
- **C, seçilen:** her zaman pozitif (0 < r ≤ 1) → hayatta kalmak her zaman ölmekten iyi. Maliyetle **sürekli** azalıyor:
  dik ve sakin = 1, c = 1 → 0.5, hiçbir yerde düz değil.

**Kritik gözlem:** Üç adayda da LQR açık ara en yüksek puanı aldı. Yol haritasının "LQR en yüksek ödülü almalı" kontrolünü
**hepsi geçiyor.** Tuzaklar iyi politikada değil, **kötü politikalarda**, yani eğitimin başında görünüyor. **Ödülü sadece
iyi politikayla test etmek yetmez:** intihar ve "kötü ama hayatta" gibi uç politikaları da dene.

**Q'daki x² terimi neden önemli?** Yol haritasının örneği: sadece "hayatta kal" ödülü verip konumu cezalandırmazsan ajan
arabayı rayın kenarına götürüp orada bekleyebilir. c'nin içindeki q_x·x² bunu önlüyor.

---

## 7. Kod: CartPoleContinuousEnv

```python
def step(self, action):
    F = float(np.clip(action[0], -1.0, 1.0)) * self.F_max
    h = self.dt / self.substeps
    for _ in range(self.substeps):
        self.s = cp.rk4_step(self.f, self.s, F, h)            # Aşama 1'in RK4'ü
    self.adim += 1
    c = float(self.Q @ self.s**2 + self.R * F**2)             # Q köşegen → Σ q_i s_i²
    reward = 1.0 / (1.0 + c)
    terminated = bool(abs(self.s[0]) > 2.4 or abs(self.s[2]) > np.pi / 2)
    truncated = bool(self.adim >= self.max_adim)
    return self._observation(), reward, terminated, truncated, {"F": F}
```

| ayrıntı | neden |
|---|---|
| `bool(...)` | numpy `np.bool_` Gymnasium'da uyarıya yol açıyor |
| `.astype(np.float32)` | ağlar ve `observation_space` float32 |
| `info={"F": F}` | uygulanan kuvvet → değerlendirmede efor |
| `options={"s0"}` içinde `.copy()` | dışarıdaki diziyle bağ kurmamak (referans tuzağı, Aşama 5) |
| `check_env` modül seviyesinde değil | `import envs` yan etki yapmamalı → `if __name__ == "__main__":` ya da test |

---

## 8. Doğrulama

| test | ne kanıtlıyor |
|---|---|
| `test_env_passes_check_env` | Gymnasium standardı |
| `test_env_reset_reproducible` | aynı tohum → aynı başlangıç ve aynı gürültülü gözlem; `options["s0"]` aynen kuruluyor |
| **`test_env_lqr_matches_run_closed_loop`** | ortamın içinden LQR = Aşama 4'teki `run_closed_loop` yörüngesi (1e-5). **Fizik, ölçekleme ve kırpma doğru.** RL kötü öğrenirse sorunun ortamda olmadığını bu testten biliyoruz |
| `test_env_reward_and_termination_flags` | ödül her zaman (0, 1]; devrilme → terminated (truncated değil); süre → truncated (terminated değil) |

**Gerçek bir hata ve dersi:** `terminated` / `truncated` satırları sadeleştirilirken `step`'in `return` satırı silindi.
Python `return`'süz fonksiyondan sessizce `None` döndürür; hata ancak defterde sonuç kullanılınca çıktı. Değişiklikten
sonra `pytest` çalıştırılsaydı LQR testi anında kırmızı yanardı. **Alışkanlık: dosyayı değiştir → pytest → sonra defter.**

---

## 9. Alt referanslar ve ödül ≠ ölçüt

Zor geliştirme bloğu (50 senaryo, `options["s0"]` ile):

| politika | ort. ödül | ort. uzunluk | düştü (terminated) |
|---|---|---|---|
| rastgele | 2.9 | 27.5 | 50/50 |
| sıfır | 4.0 | 29.6 | 50/50 |
| **LQR** | **384.2** | 426.3 | **8/50** |

- RL'nin **en az** geçmesi gereken: rastgele / sıfır (~3–4). **Yetişmesi gereken:** LQR (384).
- LQR'nin 8 düşüşü = Aşama 6'daki 8 başarısızlık (42/50). **Ama tanımlar farklı:** `terminated` sadece "devrildi / raydan
  çıktı"; Aşama 6'nın başarısı bunun üstüne "sonda dik ve sakin" istiyor. Titreyerek 500 adım hayatta kalan bir politika
  (Aşama 6'daki sonlu farklı LQR gibi) ortamda "düşmedi", Aşama 6'da "başarısız".
- **Sonuç:** RL Aşama 9'da **Aşama 6'nın düzeneğiyle** değerlendirilecek. **Ödül eğitim için bir sinyal, başarı tanımı
  ölçüt.** İkisi karıştırılmaz. (Ödülü yüksek ama görevi yapmayan ajan, ödül sömürüsünün ta kendisi.)

---

## 10. Mülakat soruları ve örnek cevaplar

**S1. Ödül fonksiyonunu nasıl tasarladın, neden böyle?**
LQR maliyetinden başladım. Negatif maliyet erken bitebilen bölümlerde intihar teşviki yaratıyordu: kötü ama hayatta kalan
politika −22.5, kendini öldüren −20.2 alıyordu. Kırpılmış pozitif ödül dikten uzakta sinyalsizdi. Her zaman pozitif ve
maliyetle sürekli azalan r = 1/(1 + c)'yi seçtim. Bunu sadece LQR'yle değil, intihar ve "kötü ama hayatta" politikalarıyla
da test ettim.

**S2. Ajanın ödülü sömürebileceği bir durum gördün mü?**
Tasarım aşamasında iki tane buldum: negatif maliyetle erken ölüm, konum cezası olmayan "hayatta kal" ödülünde rayın
kenarında beklemek. Ayrıca ödülü ölçüt olarak kullanmıyorum; değerlendirme ayrı bir başarı tanımıyla yapılıyor.

**S3. terminated ile truncated farkı ne, neden önemli?**
terminated gerçek bir son (değer 0), truncated yapay bir kesinti (değer tahmin edilmeli). Karıştırılırsa algoritma süre
dolmasını ölüm sanır ve bölüm sonuna yakın durumların değerini yanlış öğrenir.

**S4. Gözlem ve aksiyon uzayını nasıl seçtin, neden aksiyonu [−1, 1]'e ölçekledin?**
Aksiyon [−1, 1], ortam F = a·10 N yapıyor ve kırpıyor: ağlar ~1 ölçeğini sever, SAC'ın tanh'ı bu aralık için. Gözlem tam
durum ve sınırsız; hızların ve gürültünün doğal sınırı yok. Ölçek için normalizasyon kullanıyorum.

**S5. Ortamının doğru olduğunu nasıl bildin?**
Ortamın içinden LQR çalıştırıp Aşama 4'teki doğrulanmış kapalı döngü simülasyonuyla aynı yörüngeyi verdiğini test ettim.
Ayrıca check_env, tohum tekrarlanabilirliği ve bitiş bayraklarının testleri var. LQR alt referanslarda açık ara en yüksek
ödülü aldı.

---

## 11. Tuzaklar (gerçekten yaşananlar)

| tuzak | belirti | önlem |
|---|---|---|
| `step`'te `return` satırının silinmesi | `env.step()` → None | değişiklikten sonra pytest |
| `check_env` modül seviyesinde | her import ekrana yazıyor | `if __name__ == "__main__":` / test |
| `import evaluate as eval` | yerleşik `eval` gölgeleniyor | `as ev` |
| Negatif maliyet ödülü | intihar teşviki | her zaman pozitif ödül; uç politikalarla test |
| Kırpılmış ödül | dikten uzakta sinyal yok | sürekli azalan ödül |
| Süre dolmasını terminated yapmak | bootstrap hatası | truncated |
| 12°'de terminated | zor senaryolar anında biter | π/2 |
| Ödülü ölçüt sanmak | titreyen politika "başarılı" | Aşama 6 düzeneği |
| Sonlu ama tutulamayan gözlem sınırları | sessiz kırpma / uyarı | ±∞ ya da gerçekten geniş fiziksel sınır |

---

## 12. Kaynak haritası

| konu | kaynak |
|---|---|
| Gymnasium arayüzü, terminated / truncated | Gymnasium dokümanı → "Create a Custom Environment", "Basic Usage" |
| SB3 için ortam ipuçları | SB3 dokümanı → "Reinforcement Learning Tips and Tricks", "Using Custom Environments" |
| Ödül hipotezi, MDP | Sutton & Barto, bölüm 3; Spinning Up "Key Concepts" |
| Ödül sömürüsü örnekleri | DeepMind "Specification gaming" blog yazısı ve örnek listesi |

---

## 13. Kendini sına

1. r = −c·dt ödülünde intihar politikası neden "kötü ama hayatta" politikadan daha iyi puan alıyor? Sayılarla.
2. r = 1/(1 + c)'de sarkaç dik ama araba x = 0.5'te sabit duruyorsa adım başına ödül kaç?
3. Süre dolması terminated olarak işaretlenseydi, 499. adımdaki mükemmel bir durumun öğrenilen değeri ne olurdu?
4. Neden aksiyonu ±10 yerine ±1 verip ortamda ölçekliyoruz?
5. Gözlem uzayını x ∈ [−2.4, 2.4] yazsaydın hangi gözlem bu sözü bozardı?
6. LQR'nin ortamdaki "düştü" sayısı ile Aşama 6 başarısızlık sayısı hangi kontrolcüde farklı çıkardı? Neden?
7. `test_env_lqr_matches_run_closed_loop` hangi hataları yakalar (en az üç tane say)?
8. Ödül fonksiyonunu test etmek için hangi politikaları kullanırsın, neden sadece iyi politika yetmez?
