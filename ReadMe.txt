## Açıklama 
Matplotlib, numpy ve OpenCV kullanılarak
geliştirilen hareket takip modeli. Kod modüler yapıda organize edilmiştir.


## Gereksinimler 
pip install -r requirements.txt 
|-"tennis.mp4"


## Çalıştırma 
 python project.py

### Dosya Düzeni 
proje/
|- "project.py"
|- "requirements.txt"
|- "tennis.mp4."
|- "ReadMe.txt"
|- "tennis_court.png"

## Süreç

Kod ekrandaki mavilik oranına göre tenis maçının devam edip etmediğini analiz eder.

Maç devam ediyorsa kort bulunur ve dikdörtgen şeklinde rectify edilir.

Rectify edilmiş kort üzerinde oyuncular ve topun konumları görselleştirilir.

Sonrasında tenis kordu ebatlarına göre oyuncular ve top tennis_court.png üzerinde gösterilirler.