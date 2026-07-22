from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from utils import muat_dan_bagi_data, simpan_hasil_dan_grafik

def main():
    print("=== Memulai Program Mini Machine Learning ===")
    dataset_path = "Dataset/dataset.csv"
    kolom_target = "Price" 
    
    try:
        df, X_train, X_test, y_train, y_test = muat_dan_bagi_data(dataset_path, kolom_target)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        skor_r2 = r2_score(y_test, y_pred)
        
        simpan_hasil_dan_grafik(df, y_test, y_pred, skor_r2)
        print("=== Program Selesai Dijalankan ===")
        
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    main()