# SimStar ile Otonom Yarış: Pekiştirmeli Öğrenme

SimStar ADAS simülatöründe otonom yarış için DQN, PPO, TD3 ve SAC yaklaşımlarının incelendiği bir pekiştirmeli öğrenme projesi. Proje raporunda **Soft Actor-Critic (SAC)** seçildi; farklı deneylerde kaydedilen PPO sonuçları bazı pistlerde daha yüksek ilerleme gösteriyor. Bu klasör SAC ve PPO kodlarını, modelleri, raporu ve test kayıtlarını bir araya getirir.

Çalışma **Ahmet Naci Baytuğ** tarafından hazırlanmıştır. Deneyler simülasyona aittir; gerçek bir araç üzerinde uygulanmış veya doğrulanmış değildir.

## Modeller

| Model | Güçlü yönü | Zayıf yönü ve bu projedeki durum |
|---|---|---|
| **DQN / Double DQN** | Sabit direksiyon, gaz ve fren kombinasyonlarıyla anlaşılır bir başlangıç sağlar; Double DQN aşırı değer tahminini azaltmayı hedefler. | Önceden belirlenen aksiyonlarla sınırlıdır; ince ve sürekli kontrol zorlaşır. Bu pakette karşılaştırılabilir nihai test CSV'si veya model yok. |
| **PPO** | Sürekli kontrol sağlar; kayıtlı denemelerde Austria ve Magny-Cours'ta yüksek ortalama ilerleme göstermiştir. | **Eğitimi veri toplama açısından maliyetli olabilir:** güncel politikayla yeni deneyimler toplanır ve her veri grubu sınırlı sayıda güncellemede kullanılır. Simülatör deneyimi pahalıysa bu zaman ve hesaplama ihtiyacını artırabilir. Magny-Cours sonuçları da bölümler arasında değişkendir. |
| **TD3** | Sürekli kontrol, çift eleştirmen ve tekrar kullanılan deneyimlerle veri kullanımını iyileştirmeyi hedefler. | Eğitim ayarlarına ve keşif gürültüsüne duyarlı olabilir. Test kayıtları var; model ve eğitim kodu bu pakette yok. |
| **SAC** | Tekrar kullanılan deneyim ve entropi tabanlı keşifle sürekli kontrol sunar. Dutch, Hungary ve İstanbul kayıtlarında yüksek ortalama ilerleme görülür. | Politikayı ve eleştirmenleri birlikte eğitmek ayar ve hesaplama gerektirir; Austria kaydında erken başarısızlık görülmüştür. Eğitim/test kodları ve model pakette bulunuyor. |

PPO veriyi birden fazla optimizasyon turunda kullanabilir, ancak eski deneyimleri SAC ve TD3'teki tekrar oynatma belleği gibi sürekli kullanmaz. Bu nedenle daha çok yeni simülasyon verisine ihtiyaç duyabilir. Bu, **örnek verimliliğine ilişkin genel bir farktır**: gerçek duvar saati, GPU kullanımı ve parasal maliyet donanıma ve ayarlara bağlıdır. Bu projede algoritmaların aynı koşullardaki maliyetleri ölçülmediği için "PPO kesin olarak daha pahalıydı" sonucuna varılmamaktadır. Ayrıntı: [PPO makalesi](https://arxiv.org/abs/1707.06347), [SAC makalesi](https://arxiv.org/abs/1801.01290).

### Kayıtlı testlerin karşılaştırması

Aşağıdaki değerler her CSV'deki **beş bölümün ortalama birikimli tur ilerlemesidir**. %100 bir tur, %200 iki tur demektir. Değerler `results/` altındaki kayıtlardan hesaplanmıştır; bazı bölümler **20.000 adımlık üst sınıra** ulaştığından daha fazla devam edip etmeyecekleri bilinmez.

| Pist | DQN | PPO | TD3 | SAC |
|---|---:|---:|---:|---:|
| Austria | Veri yok | %797,7 | %66,3 | %15,0 |
| Dutch | Veri yok | %646,7 | %603,3 | %677,7 |
| Hungary | Veri yok | %289,6 | %723,5 | %818,5 |
| İstanbul Park | Veri yok | %387,1 | %445,1 | %468,3 |
| Magny-Cours | Veri yok | %413,7 | %56,4 | %56,4 |

Bu tablo bir **keşif karşılaştırmasıdır**: farklı tarihlerde üretilmiş dosyaların test betiği sürümleri, başlangıç koşulları ve tüm ayarları aynı olacak şekilde bağımsız doğrulanmamıştır. Özellikle Magny-Cours kayıtları ayrı test koşularındandır. Bazı PPO sonuçları rapordan daha sonra kaydedilmiştir; bu nedenle rapordaki SAC seçimi bu kayıtların tümünü değerlendirmiş olmayabilir. PPO'nun eğitim maliyeti bir değerlendirme ölçütüdür, ancak modellerin bu projedeki karşılaştırılabilir süre ve GPU maliyetleri eldeki kayıtlarda bulunmamaktadır.

PPO için Austria'da beş bölümün tamamı 20.000 adım sınırına ulaşmış; Magny-Cours'ta ise bölüm ilerlemeleri yaklaşık %70,7–%780,0 arasında değişmiştir. Yüksek ortalamanın yanında bu değişkenliği de dikkate almak gerekir. Gerçek bir sıralama için her modeli aynı test betiği, pist, başlangıç koşulları, seed'ler ve adım sınırıyla yeniden değerlendirmek gerekir.

Örnek dosyalar: [PPO Austria](results/ppo/test_ppo_v2_lastppo_best_austria.csv), [PPO Magny-Cours](results/ppo/test_ppo_v2_lastppo_best_magnycours.csv), [SAC Austria](results/sac/test_v2_sac_best_austria.csv), [TD3 Austria](results/td3/test_td3_best_austria.csv). Tüm tablonun hesaplaması `python results/compare_results.py` ile tekrarlanabilir.

Raporda SAC Austria ortalama hızı 59,0 km/s, ilgili CSV ortalaması 59,4 km/s'tir; farkın nedeni henüz doğrulanmamıştır. Raporda pistlerden biri "Hidden Track" olarak adlandırılır; tablo dosya adlarını kullanır ve gizli pistin kimliği ayrıca doğrulanmamıştır. Testler bu hazırlık sırasında yeniden çalıştırılmamıştır.

## İstanbul Park ve Magny-Cours

Projeye İstanbul Park ve Magny-Cours için ek pist verileri ve görselleştirmeler dahil edilmiştir:

- [İstanbul Park görseli](examples/IstanbulPark_Map_Visualization.png) · [pist verisi](examples/IstanbulPark.pkl)
- [Magny-Cours görseli](examples/MagnyCours_Map_Visualization.png) · [pist verisi](examples/MagnyCours.pkl)

Pist üretiminde OpenStreetMap / Overpass yol verilerinden yararlanılmıştır. Harita verisi © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright), [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/). Pistlerin simülatöre uyarlanması ve görselleştirilmesi bu proje kapsamındadır. Görsellerin ve türetilmiş verilerin paylaşım koşulları yayımlamadan önce gözden geçirilmelidir.

## Klasör yapısı

```text
examples/
  train_sac.py, test_sac.py            SAC eğitim ve test kodu
  test_sac_custom_tracks.py            Ek pist adlarını destekleyen test sürümü
  train_ppo.py, test_ppo.py            PPO eğitim ve test kodu
  checkpoints/                         SAC ve PPO modelleri
  *.pkl, *Map_Visualization.png        Pist verileri ve görselleri
results/                               Saklanan SAC, TD3 ve PPO test CSV'leri
docs/                                  Proje raporu
archive/                               Orijinal PPO arşivi
```

Temel kodlar, model dosyaları ve rapor okunabilir klasör ve dosya adlarıyla düzenlenmiştir. Model adları, test betiklerinin varsayılan yollarıyla uyumludur. TD3 karşılaştırma CSV'leri bulunur; TD3 model dosyası bu pakette yer almaz.

## Çalıştırma

1. **SimStar simülatörünü**, Python API'sini, uyumlu `simstarEnv` ortamını ve standart pist dosyalarını yetkili kaynaklarından ayrı edinin. Bu üçüncü taraf bileşenler bu pakette bulunmaz. SimStar varsayılan olarak `127.0.0.1:8080` üzerinden iletişim kurar.
2. Python ortamını hazırlayın. Geliştirme ortamında Python 3.8 kullanılmıştır; PyTorch, NumPy, pandas ve SimStar API bağımlılıkları gerekir. Farklı kaynaklarda Gym sürümleri uyuşmadığından tam bağımlılık kurulumu henüz doğrulanmamıştır.
3. `simstar` ve `simstarEnv` modüllerinin Python tarafından bulunabildiğini, standart pist dosyalarının çalışma dizininde bulunduğunu ve SimStar'ın çalıştığını kontrol edin. **Bu eksik bağımlılıklar tamamlanmadan aşağıdaki komutlar çalışmaz.**
4. Ardından proje kökünden `examples` dizinine geçerek SAC modelini deneyin:

```bash
cd examples
python test_sac.py austria 5 --model checkpoints/sac_best.pt
```

İstanbul Park ve Magny-Cours için ayrı test sürümü bulunur. Bu betik, kullandığınız SimStar API'de `IstanbulPark` ve `MagnyCours` pist adlarının ve uyumlu ortam desteğinin bulunmasını gerektirir. Standart API'de bu uyarlamalar yoksa yalnızca `.pkl` dosyalarını kopyalamak yeterli değildir:

```bash
python test_sac_custom_tracks.py istanbulpark 5 --model checkpoints/sac_best.pt
python test_sac_custom_tracks.py magnycours 5 --model checkpoints/sac_best.pt
```

Test betikleri sonuca ait CSV dosyaları yazabilir. Pist `.pkl` dosyaları çalışma dizinine göre okunur; bu yüzden komutları `examples` içinden çalıştırın. İki özel pist dosyası SimStar waypoint sınıflarını referans alabilir ve uyumlu API gerektirir. Harita oluşturma betikleri de projede kullanılan özel API uyarlamalarına ihtiyaç duyabilir. Komutlar bu pakette canlı SimStar oturumuyla henüz doğrulanmamıştır. Standart SAC ve PPO test betikleri yalnızca Austria, Dutch ve Hungary seçeneklerini listeler; ek pist desteği ayrı SAC test betiğindedir.

## Kaynaklar ve paylaşım koşulları

- SimStar ve SimStar Python API: **Eatron Technologies**. Proje belgeleri ayrıca İstanbul Teknik Üniversitesi BLG604 ders materyallerine atıf yapar. API'nin kaynak paketinde lisans **Proprietary** olarak belirtilmiştir. API, ortam kodu ve standart pist dosyaları paylaşım koşulları doğrulanmadığından bu pakete dahil edilmemiştir; kullanmak için yetkili bir kaynaktan edinilmelidir.
- OpenStreetMap verisi ve katkıcıları: yukarıdaki ODbL atfına bakın.
- Üçüncü taraf dosyalar kendi lisans koşullarına tabidir. Projenin kullanıcıya ait kodu için ayrıca bir açık kaynak lisansı henüz belirlenmemiştir.
