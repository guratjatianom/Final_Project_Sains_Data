import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

def muat_dan_bagi_data(file_path, target_column):
    """Fungsi untuk membaca data dan membaginya menjadi Train & Test set"""
    df = pd.read_csv(file_path)
    
    # 💡 SOLUSI ERROR: Hapus kolom 'Address' karena isinya teks/string yang tidak bisa diproses ML
    if 'Address' in df.columns:
        df = df.drop(columns=['Address'])
    
    # Memisahkan Fitur (X) dan Target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split data menjadi 80% Train (latih) dan 20% Test (uji)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def simpan_hasil_dan_grafik(y_test, y_pred, r2_score):
    """Fungsi untuk membuat grafik evaluasi profesional dan menyimpan laporan beserta sampel harga"""
    os.makedirs('output', exist_ok=True)
    
    # --- MENYIAPKAN 10 SAMPEL PERBANDINGAN HARGA ---
    # Membuat DataFrame sementara untuk mengambil 10 sampel teratas
    df_sampel_harga = pd.DataFrame({
        'Harga Asli (USD)': y_test,
        'Harga Prediksi (USD)': y_pred
    }).head(10)
    
    # 1. Simpan laporan teks ke hasil_analisis.txt (Kini ditambah sampel harga)
    with open('output/hasil_analisis.txt', 'w') as f:
        f.write("=== LAPORAN EVALUASI MODEL MACHINE LEARNING ===\n")
        f.write("Model: Linear Regression (Prediksi Harga Rumah)\n")
        f.write(f"Akurasi Model (R2 Score): {r2_score:.4f}\n")
        f.write("===============================================\n\n")
        
        f.write("=== SAMPEL PERBANDINGAN HARGA (10 DATA PERTAMA) ===\n")
        f.write(df_sampel_harga.to_string(index=False, formatters={
            'Harga Asli (USD)': lambda x: f"${x:,.2f}",
            'Harga Prediksi (USD)': lambda x: f"${x:,.2f}"
        }))
        f.write("\n===============================================\n")
    
    # 2. Gambar grafik (kode grafik tetap sama seperti sebelumnya)
    df_grafik = pd.DataFrame({'Asli': y_test, 'Prediksi': y_pred})
    df_sampel = df_grafik.sample(n=200, random_state=42) if len(df_grafik) > 200 else df_grafik
    
    asli_k = df_sampel['Asli'] / 1000
    prediksi_k = df_sampel['Prediksi'] / 1000
    
    plt.figure(figsize=(9, 7))
    plt.scatter(asli_k, prediksi_k, color='#2b5c8f', alpha=0.6, s=25, label='Sampel Data Prediksi')
    
    min_val = min(asli_k.min(), prediksi_k.min())
    max_val = max(asli_k.max(), prediksi_k.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='#e74c3c', linestyle='--', linewidth=2, label='Prediksi Sempurna')
    
    plt.title('Grafik Evaluasi: Harga Asli vs Harga Prediksi', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Harga Asli (dalam Ribuan USD / K$)', fontsize=11)
    plt.ylabel('Harga Prediksi (dalam Ribuan USD / K$)', fontsize=11)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('output/grafik_output.png', dpi=300)
    plt.close()
    print("Berhasil memperbarui laporan dengan menyertakan sampel perbandingan harga!")