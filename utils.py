import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def muat_dan_bagi_data(file_path, target_column):
    df = pd.read_csv(file_path)
    
    if 'Address' in df.columns:
        df = df.drop(columns=['Address'])
        
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return df, X_train, X_test, y_train, y_test

def simpan_hasil_dan_grafik(df, y_test, y_pred, r2_score):
    os.makedirs('output', exist_ok=True)
    
    rata_harga = df['Price'].mean()
    harga_max = df['Price'].max()
    harga_min = df['Price'].min()
    
    with open('output/hasil_analisis.txt', 'w') as f:
        f.write("=== STATISTIK DATASET & EVALUASI MODEL ===\n")
        f.write(f"Rata-rata Harga Rumah  : ${rata_harga:,.2f}\n")
        f.write(f"Harga Tertinggi        : ${harga_max:,.2f}\n")
        f.write(f"Harga Terendah         : ${harga_min:,.2f}\n")
        f.write("-------------------------------------------\n")
        f.write(f"Akurasi Model (R2 Score): {r2_score:.4f}\n")
        f.write("===========================================\n")

    df_grafik = pd.DataFrame({'Asli': y_test, 'Prediksi': y_pred})
    df_sampel = df_grafik.sample(n=200, random_state=42) if len(df_grafik) > 200 else df_grafik
    
    asli_k = df_sampel['Asli'] / 1000
    prediksi_k = df_sampel['Prediksi'] / 1000
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].scatter(asli_k, prediksi_k, color='#2b5c8f', alpha=0.6, s=25)
    min_val = min(asli_k.min(), prediksi_k.min())
    max_val = max(asli_k.max(), prediksi_k.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], color='#e74c3c', linestyle='--', linewidth=2)
    axes[0].set_title('Scatter Plot: Harga Asli vs Prediksi', fontweight='bold')
    axes[0].set_xlabel('Harga Asli (K$)')
    axes[0].set_ylabel('Harga Prediksi (K$)')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    axes[1].hist(df['Price'] / 1000, bins=30, color='#27ae60', edgecolor='black', alpha=0.7)
    axes[1].set_title('Histogram: Distribusi Harga Rumah', fontweight='bold')
    axes[1].set_xlabel('Harga Rumah (K$)')
    axes[1].set_ylabel('Frekuensi / Jumlah Rumah')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('output/grafik_output.png', dpi=300)
    plt.close()
    print("Berhasil menyimpan file ke dalam folder output")