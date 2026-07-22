# FINAL PROJECT MINI MACHINE LEARNING - UAS
# TOPIK: PREDIKSI HARGA RUMAH (HOUSE PRICE PREDICTION)
# ALGORITMA: LINEAR REGRESSION
Linear Regression adalah salah satu algoritma Machine Learning paling dasar dan populer yang digunakan untuk memprediksi nilai numerik (regresi).

========================================================================
1. DESKRIPSI PROJECT
========================================================================
Project ini merupakan aplikasi Mini Machine Learning sederhana yang dibangun secara lokal menggunakan text editor (VS Code). 
Tujuan dari project ini adalah untuk melakukan prediksi harga rumah berdasarkan fitur-fitur spesifik (seperti rata-rata pendapatan area, usia rumah, 
jumlah kamar, jumlah ruangan, dan populasi area) menggunakan metode Regresi Linear.

Dataset yang digunakan: USA Housing Dataset (Kaggle).

========================================================================
2. STRUKTUR FOLDER & KOMPONEN
========================================================================
Sesuai ketentuan pengumpulan UAS, berikut adalah struktur direktori project:

nama_project/
├── utils.py            -> Berisi fungsi helper (load data, preprocessing, & visualisasi)
├── main.py             -> File source code utama untuk training dan eksekusi model ML
├── Dataset/
│   └── dataset.csv     -> File dataset utama dari Kaggle (USA Housing Dataset)
├── output/
│   ├── hasil_analisis.txt -> Laporan akurasi model (R2 Score) hasil running
│   └── grafik_output.png  -> Visualisasi perbandingan Harga Asli vs Harga Prediksi
└── Readme.txt          -> Dokumentasi dan panduan ini

========================================================================
3. DEPENDENSI / PREREQUISITES
========================================================================
Aplikasi ini dikembangkan menggunakan Python 3 secara lokal. Pastikan beberapa library berikut telah terinstal sebelum menjalankan program:

- pandas
- scikit-learn
- matplotlib
- numpy

Perintah instalasi via terminal/command prompt:
pip install pandas scikit-learn matplotlib numpy

========================================================================
4. CARA MENJALANKAN PROGRAM
========================================================================
1. Buka folder root project (`nama_project/`) menggunakan VS Code atau IDE lokal Anda.
2. Pastikan file dataset sudah berada di dalam folder `Dataset/` dengan nama `dataset.csv`.
3. Buka terminal lokal di VS Code, lalu jalankan perintah berikut:
   
   python main.py

4. Program akan memproses data, melatih model, dan secara otomatis memperbarui file di dalam folder `output/`.

========================================================================
5. LOGIKA & ALUR PROGRAM
========================================================================
- Data Loading & Cleaning: Program membaca data CSV dan membuang fitur non-numerik seperti kolom 'Address' agar data bersih.
- Splitting Data: Data dibagi menjadi 80% Data Training (untuk melatih pola) dan 20% Data Testing (untuk evaluasi keakuratan).
- Model Training & Evaluation: Algoritma Linear Regression melatih model, melakukan prediksi pada data testing, dan mengukur skor menggunakan metrik R-Squared (R2 Score).
- Export Visualisasi: Program melakukan teknik sampling acak (200 sampel) dan alpha blending pada visualisasi scatter plot agar grafik output bersih, estetik, dan mudah dianalisis tanpa penumpukan data yang padat.
