# Instance Segmentation pada Lingkungan Indoor Nyata

Berliana Sarlita Rahajeng
5054241023
Visi Komputer (N)(RKA)

**Tugas Evaluasi CPMK 3 — Visi Komputer (EA234503), ITS**  
**Eksperimen NRP berakhiran 3: proporsi data latih 25%, 50%, dan 100%**

## Ringkasan

Proyek ini membandingkan tiga model *instance segmentation* dari dua keluarga arsitektur, yaitu **YOLO11n-seg** dan **YOLO11s-seg** (*one-stage*) serta **Mask R-CNN** (*two-stage*). Model dilatih menggunakan **Panoptic Indoor Segmentation v5** dan dievaluasi pada dua domain: test set asli dan kumpulan foto indoor pribadi yang dianotasi dalam format COCO Segmentation.

Tujuan eksperimen adalah (1) membandingkan kualitas segmentasi antararsitektur, (2) menguji pengaruh proporsi data latih terhadap performa, (3) menganalisis *domain shift* pada foto pribadi, dan (4) mengidentifikasi serta menguji dugaan penyebab kegagalan prediksi.

## Dataset

| Dataset | Jumlah gambar | Pemanfaatan |
|---|---:|---|
| Panoptic Indoor Segmentation v5 — train | 397 | Pelatihan model |
| Panoptic Indoor Segmentation v5 — validation | 166 | Pemantauan dan validasi pelatihan |
| Panoptic Indoor Segmentation v5 — test | 359 | Evaluasi domain asli |
| Foto indoor pribadi | 40 | Evaluasi domain pribadi dan analisis kegagalan |

Dataset sumber: [Panoptic Indoor Segmentation — Roboflow Universe](https://universe.roboflow.com/panopticindoor/panoptic-indoor-segmentation). Dataset v5 memiliki 75 kategori terdaftar; 74 di antaranya memiliki anotasi dalam pemeriksaan awal. Data pribadi memiliki **290 anotasi mask** pada **40 foto** menurut output notebook final.

### Tautan dataset pribadi (Kaggle)

- **40 foto asli:** [40 Dataset Viskom — Kaggle](https://kaggle.com/datasets/aaf16a497ceb1de7346eed3fce0d2701b2192ad66f00d82b6a7917986f5dd457)
- **Hasil anotasi:** [Personal Annotations — Kaggle](https://kaggle.com/datasets/9608766d615630c064e01bb125216be45ba46a5d7b3092694d52a671fcf9041f)


## Model dan desain eksperimen

| Model | Keluarga arsitektur | 25% | 50% | 100% |
|---|---|:---:|:---:|:---:|
| YOLO11n-seg | One-stage | ✓ | ✓ | ✓ |
| YOLO11s-seg | One-stage | ✓ | ✓ | ✓ |
| Mask R-CNN | Two-stage | ✓ | ✓ | ✓ |

Total **9 konfigurasi pelatihan**, masing-masing diuji pada **2 test set**, sehingga diperoleh **18 hasil evaluasi**. Subset data 25% dan 50% dibentuk secara bersarang menggunakan seed yang konsisten; validation dan test set tidak ditambahkan ke data training. Pengaturan pada notebook final mencakup `IMAGE_SIZE=512`, `YOLO_EPOCHS=25`, `MASK_EPOCHS=10`, dan batas waktu **115 menit per run**.

Metrik utama adalah **COCO mask AP@[.50:.95]** (disebut *Mask AP*), dengan pelengkap **AP50**, **AP75**, grafik training loss, dan AP per kelas.

## Hasil utama

Ringkasan Mask AP untuk model yang dilatih dengan **100% data** (angka persentase, dari output notebook final):

| Model | Test asli (359 foto) | Test pribadi (40 foto) |
|---|---:|---:|
| YOLO11n-seg | 6,59% | 8,61% |
| **YOLO11s-seg** | **10,96%** | 10,20% |
| Mask R-CNN | 6,14% | **10,24%** |

Pada test set asli, **YOLO11s-seg** menghasilkan Mask AP tertinggi di antara ketiga model pada data latih 100%. Pada test pribadi, **Mask R-CNN** unggul sangat tipis atas YOLO11s-seg; selisihnya terlalu kecil untuk dianggap sebagai bukti keunggulan yang pasti.

Nilai Mask AP test asli pada tiga proporsi data latih:

| Model | 25% | 50% | 100% |
|---|---:|---:|---:|
| YOLO11n-seg | 6,79% | 5,67% | 6,59% |
| YOLO11s-seg | 7,78% | 8,29% | 10,96% |
| Mask R-CNN | 2,78% | 3,99% | 6,14% |

Jumlah data latih yang lebih besar berkorelasi dengan peningkatan AP untuk YOLO11s-seg dan Mask R-CNN, tetapi **tidak monoton** pada YOLO11n-seg. Nilai AP secara umum masih rendah dan perlu dipahami dalam konteks ukuran dataset, variasi kelas, kualitas anotasi, dan perbedaan domain.

### Analisis kegagalan

Notebook memeringkat **5 contoh kegagalan** dari test set pribadi dan menyediakan visualisasi gambar asli, *ground truth*, serta prediksi. Sebagai eksperimen pembuktian, kelima contoh diuji dengan ukuran input **320** dan **960**. Output yang tercatat menunjukkan Mask AP masing-masing sekitar **3,20%** dan **11,09%** pada sampel terbatas tersebut. Perbedaan ini merupakan indikasi awal pengaruh resolusi, bukan bukti bahwa semua kegagalan disebabkan oleh resolusi saja.

## Notebook dan reproduksi

Notebook utama: `final-instance-segmentation.ipynb` (atau sesuaikan dengan nama file yang diunggah ke repositori).

**Lingkungan:** Kaggle Notebook dengan GPU, Python, PyTorch, Torchvision, Ultralytics, `pycocotools`, OpenCV, pandas, dan matplotlib.

1. Impor notebook ke Kaggle dan aktifkan **GPU** serta **Internet** bila pengunduhan diperlukan.
2. Tambahkan dataset utama Panoptic Indoor Segmentation v5 melalui *Add Input*, atau gunakan Roboflow API dengan secret `ROBOFLOW_API_KEY` yang diberikan akses ke notebook. **Jangan menyimpan API key di Git.**
3. Tambahkan dua Kaggle Dataset pribadi:
   - Foto asli: `/kaggle/input/datasets/berlianasarlita/40-dataset-viskom`
   - COCO anotasi: `/kaggle/input/datasets/berlianasarlita/personal-annotations/sota_annotations.coco.json`
4. Jalankan sel validasi dataset dan audit nama file. Pemetaan menangani variasi penamaan, termasuk sufiks `_sota`, tanpa menggunakan gambar overlay sebagai input evaluasi.
5. Untuk pelatihan ulang, jalankan konfigurasi `RUN_TRAINING=True`, `RUN_FULL_FACTORIAL=True`. Untuk evaluasi checkpoint yang sudah tersedia, gunakan `RUN_TRAINING=False`, `RUN_EVALUATION=True` dan pastikan output checkpoint dari versi Kaggle sebelumnya ditambahkan sebagai input.
6. Jalankan evaluasi dan visualisasi; periksa status file serta metrik sebelum menarik kesimpulan.

**Catatan:** Sembilan training run dapat melebihi waktu satu sesi GPU. Simpan *checkpoint* dan output setiap versi Kaggle. `Save & Run All` memulai run baru, bukan melanjutkan kernel interaktif yang sudah berjalan.

## Penggunaan AI

AI digunakan sebagai bantuan untuk penyusunan atau debugging kode. Contoh kendala yang pernah dijumpai adalah **ketidakcocokan nama file COCO dengan foto asli** akibat perbedaan penamaan atau sufiks `_sota`; perbaikannya dilakukan melalui audit dan normalisasi pemetaan nama berkas.

## Keterbatasan

- Jumlah dan distribusi objek berbeda antar-kelas; beberapa kelas memiliki anotasi pelatihan yang sangat sedikit.
- Perbedaan domain, kualitas *ground truth* otomatis, dan ukuran sampel memengaruhi interpretasi AP.
- Hasil *domain shift* bersifat observasional; satu selisih kecil tidak cukup untuk menyimpulkan satu arsitektur selalu lebih baik.
- Reproduksibilitas memerlukan versi dependensi, seed, akses dataset, log, dan checkpoint yang konsisten.

## Referensi

- [Dataset Panoptic Indoor Segmentation — Roboflow Universe](https://universe.roboflow.com/panopticindoor/panoptic-indoor-segmentation)
- [Dokumentasi Ultralytics](https://docs.ultralytics.com/)
- [Dokumentasi Torchvision Mask R-CNN](https://pytorch.org/vision/stable/models/mask_rcnn.html)
- [COCO evaluation API (`pycocotools`)](https://github.com/cocodataset/cocoapi)

---
