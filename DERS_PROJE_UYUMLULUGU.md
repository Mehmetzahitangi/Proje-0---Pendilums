# Ders – Proje Uyumluluğu: Modern Kontrol Notları ↔ Proje 0

> Kaynak: `C:\Zahit\Yüksek Lisans Dersler\Modern Kontrol\Week 1–11` (el yazısı notlar).
> Amaç: Projede yaptığımız her adımın derste gördüğün hangi tekniğe karşılık geldiğini göstermek.
> Ayrıntılar için: `NOTLAR_Asama1_2.md`, `NOTLAR_Asama3.md`.

---

## 1. Notasyon sözlüğü

| ders notların | bizim proje | anlamı |
|---|---|---|
| x (durum vektörü) | **s** = [x, ẋ, θ, θ̇] | durum. Bizde "x" arabanın konumu olduğu için durumu s ile yazdık |
| u (girdi) | **F** | kontrol girdisi (kuvvet) |
| ẋ = Ax + Bu, y = Cx + Du | ṡ = As + BF | aynı denklem |
| u = −Kx | F = −K·s | durum geri beslemesi; işaret kuralı da aynı |
| A − BK | A − BK | kapalı döngü matrisi |
| K_e (gözlemci kazancı) | Aşama 5'teki Kalman kazancı | — |

---

## 2. Eşleştirme: ders notların → proje

| hafta | derste gördüğün | projede nerede kullandık |
|---|---|---|
| **1** | Doğrusal olmayan sistem, Taylor ile doğrusallaştırma fikri, açık / kapalı döngü | Aşama 3'ün başı: açık döngüde sarkaç düşer, kapalı döngü |
| **2** | **Taylor ile doğrusallaştırma:** Δẋ = ∂f/∂x·Δx + ∂f/∂u·Δu. sin(x)'in 30° civarında doğrusallaştırılması, "uzaklaşınca hata büyür" grafiği | **Aşama 2'nin tamamı.** `f.jacobian(s)` senin ∂f/∂x'in, `f.jacobian([F])` senin ∂f/∂u'n. sinθ ≈ θ grafiği ve "doğrusal model 75–90°'de kopuyor" deneyi, notundaki "doğrusallaştırma noktasından uzaklaştıkça hata büyür" uyarısının sayısal hâli |
| **3–4** | Laplace, durum geçiş matrisi Φ(s) = (sI − A)⁻¹, **transfer fonksiyonu H(s) = CΦ(s)B + D**, benzerlik dönüşümü | Aşama 2, Bölüm 10: **G_x(s) = C(sI − A)⁻¹B** ile F → x transfer fonksiyonu ve sıfırları. Defter 02'deki sympy `(s*I − A).inv()` hesabı senin formülünün kendisi |
| **4** | Kontrolcü kanonik formu | Doğrudan kullanmadık, ama kutup yerleştirmenin arkasındaki yöntem bu (hafta 8) |
| **5** | Jordan formu, **tekrarlanan özdeğerler**, e^{At} | Aşama 2, Bölüm 7.4: arabanın **iki sıfır özdeğeri bir Jordan bloğu** ([[0,1],[0,0]]). "λ = 0 tekrarlı ise kararlı değil, x(t) = x₀ + ẋ₀t büyür" sonucu buradan |
| **5–6** | **Modal ayrıştırma**, e^{At} = V e^{Λt} V⁻¹, karmaşık özdeğer → salınım (−1 ± i√2 örneği) | Aşama 2, Bölüm 7.2: "her özdeğer bir mod". Aşama 3'teki özdeğer okuma rehberi (σ ± iω), senin notundaki e^{σt}(cos ωt + i sin ωt) çözümünün aynısı |
| **5–6** | **e^{At} (matris üsteli)** | **Ayrıklaştırma:** A_d = e^{A·dt}. `cartpole.discretize` fonksiyonu senin e^{At}'ni t = dt'de hesaplıyor |
| **7** | **Kutup yerleştirme**, u = −Kx, karmaşık düzlemde özdeğer konumu ve tepki şekli | **Aşama 3'ün omurgası:** A − BK, özdeğer haritası. Hafta 7'nin ilk sayfasındaki çizim (sol yarı düzlem kararlı, sanal kısım salınım) bizim okuma rehberimiz. Ters tepki deneyindeki `control.place` da kutup yerleştirme |
| **8** | Kontrolcü tasarım adımları, **M = [B AB … Aⁿ⁻¹B]**, Ackermann | **M kontrol edilebilirlik matrisi 𝒞** (Aşama 2, Bölüm 8). Kanonik forma dönüşüm ancak M tersinirken, yani rank n iken mümkün. "Kontrol edilebilir mi?" testimizin derste nereden geldiği bu |
| **9** | **PI kontrolcü: genişletilmiş sistem** ξ̇ = r − y, takip sistemleri | Aşama 3'teki PID'nin **I terimi**, durum uzayı diliyle. Senin notundaki ξ bizim `PID.integral`. Kaskadda "x_ref'e git" takip problemi |
| **10** | Regülatör ve takip farkı, **sistem tipi** (kaç integratör var), bozucu etki | Araba **tip 1/2**: çift integratör olduğu için sabit konum hedefini integral terimi olmadan da hatasız takip ediyor. Kaskadın dış döngüsünde K_i = 0'ın yetmesinin sebebi bu |
| **10–11** | **Gözlemci (Luenberger):** x̂̇ = Ax̂ + Bu + K_e(y − ŷ), hata dinamiği ė = (A − K_eC)e, **gözlemlenebilirlik matrisi O = [C; CA; …]** | Aşama 2, Bölüm 9: **rank 𝒪** testi ("sadece θ ölçülürse rank 2"). **Aşama 5 (Kalman):** Kalman filtresi senin gözlemcinin aynı yapısı, sadece K_e'yi elle değil gürültüye göre **en iyi** şekilde seçiyor |

---

## 3. En önemli üç bağlantı

1. **Aşama 3'teki A − BK = hafta 7.** Tek fark, K'yı bir kutup listesinden değil PD kazançlarından kurmamızdı:
   K = [−K_x, 0, −K_p, −K_d]. Kararlılık haritası, "hangi K'lar kutupları sol yarı düzleme koyar?" sorusunun ızgara
   üzerinde cevaplanmış hâli.
2. **Hafta 9'daki PI'nin ξ'si, Aşama 3'teki anti-windup derdinin kaynağı.** Motor doyunca şişen değişken o
   integratörün kendisi. Derste doyma olmadığı için bu sorun görünmüyor.
3. **Hafta 10–11'deki gözlemci, Aşama 5'in ön çalışması.** Orada ė = (A − K_eC)e'nin özdeğerlerini yerleştiriyorsun.
   Kalman da aynı hata dinamiğini kuruyor; K_e'yi ölçüm ve süreç gürültüsüne göre optimal seçiyor.

---

## 4. Derste olmayan, projede yeni olanlar

| konu | aşama |
|---|---|
| Lagrange ile modelleme, RK4 (mekanik ve sayısal analiz) | 1 |
| Ayrıklaştırma (ZOH) | 2 |
| RHP sıfırı ve ters tepki (sıfırların fiziksel anlamı) | 2–3 |
| PID'nin bilgisayardaki gerçek dünya ayrıntıları: ayrık zaman, anti-windup, türev filtresi, doyma | 3 |
| Kaskad kontrol ve bant genişliği ayrımı | 3 |
| **LQR / Riccati** | 4 |
| Kalman filtresi (gürültü modeliyle optimal gözlemci) | 5 |

### LQR'ye köprü
**LQR, hafta 7–8'deki kutup yerleştirmenin "optimal" versiyonu.**
- **Kutup yerleştirmede** "kutupları −2, −3'e koy" diyip K'yı buluyordun. Ama kutupları **nereye** koyacağını nasıl
  biliyorsun? Çok sola koyarsan kuvvet patlıyor (Aşama 3'teki ters tepki deneyinde 43 N).
- **LQR** soruyu tersinden soruyor: "Kutupları sen seçme. Hataya ve kuvvete ne kadar ceza verdiğini söyle (Q, R),
  en iyi kutupları ve K'yı ben bulayım."
- Sonuç yine u = −Kx ve yine A − BK. Değişen tek şey K'nın nereden geldiği. LQR'den K'yı aldığında ilk iş hafta 7'deki
  gibi A − BK'nın özdeğerlerine bakmak.
