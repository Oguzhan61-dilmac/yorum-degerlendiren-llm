# Yorum Değerlendiren RNN

Bu proje, IMDB film yorumlarını olumlu veya olumsuz olarak sınıflandıran basit bir tekrarlayan sinir ağı (SimpleRNN) örneğidir. Depo adı isteğe uygun olarak `yorum-degerlendiren-llm` seçilmiştir; ancak model bir metin üreten büyük dil modeli (LLM) değil, duygu analizi yapan küçük bir RNN sınıflandırıcısıdır.

## Gereksinimler

- Python 3.10 veya üzeri
- TensorFlow 2.21.0

## Kurulum ve çalıştırma

Sanal ortam oluşturup etkinleştirin, ardından bağımlılıkları yükleyin:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Bağımlılıkları yükleyip modeli çalıştırın:

```bash
python -m pip install -r requirements.txt
python Rnn.py
```

İlk çalıştırmada Keras, IMDB veri kümesini indirir. Betik, daha kısa eğitim için veri kümesinin bir alt kümesini kullanır ve test kaybı ile doğruluğunu ekrana yazdırır. Model eğitimi donanıma göre birkaç dakika sürebilir.

## Model

- Kelime sözlüğü: 10.000
- Yorum başına en fazla: 200 kelime
- Eğitim/test alt kümeleri: 20.000 / 5.000 yorum
- Katmanlar: Embedding, SimpleRNN (32 birim), sigmoid çıkış
- En fazla 4 epoch; doğrulama kaybına göre erken durdurma

TensorFlow GPU desteği Windows'ta kurulu TensorFlow sürümüne bağlıdır; GPU kullanılmıyorsa eğitim CPU'da devam eder.
