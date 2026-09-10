"""
Time Delay Embedding (Zaman Gecikmeli Gömme) İşlemi
=====================================================
Bu script, Driver_Behavior.csv veri setinden belirli bir zaman serisi kolonunu alarak
kayan pencere (sliding window) yöntemiyle Time Delay Embedding uygular.

Kullanım:
    python time_delay_embedding.py

Çıktı:
    embedded_output.csv  (veya aşağıda belirtilen OUTPUT_FILE)
"""

import pandas as pd
import numpy as np

# ============================================================
# KONFİGÜRASYON PARAMETRELERİ — Buradan kolayca değiştirebilirsiniz
# ============================================================

# Girdi dosyası yolu
INPUT_FILE = r"Driver_Behavior.csv"

# Çıktı dosyası yolu
OUTPUT_FILE = r"embedded_output.csv"

# Üzerinde Time Delay Embedding yapılacak zaman serisi kolonu
# Seçenekler: speed_kmph, accel_x, accel_y, brake_pressure,
#             steering_angle, throttle, lane_deviation,
#             phone_usage, headway_distance, reaction_time
TIME_SERIES_COLUMN = "accel_x"

# Gömme boyutu (embedding dimension) = pencere büyüklüğü
WINDOW_SIZE = 10

# Kaydırma adımı (stride)
STRIDE = 1

# Sürücü / grup ayrımı için kullanılan etiket kolonu
LABEL_COLUMN = "behavior_label"

# ============================================================


def create_time_delay_embedding(series: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """
    Tek bir zaman serisine kayan pencere (sliding window) ile
    Time Delay Embedding uygular.

    Args:
        series:      1-D numpy dizisi (zaman serisi değerleri)
        window_size: Her pencerenin uzunluğu (gömme boyutu d)
        stride:      Pencereler arası kaydırma adımı

    Returns:
        (n_windows, window_size) boyutunda 2-D numpy dizisi
    """
    n = len(series)
    # Oluşturulabilecek pencere sayısı
    n_windows = (n - window_size) // stride + 1

    if n_windows <= 0:
        return np.empty((0, window_size))

    # Tüm pencereleri vektörel olarak oluştur (hızlı)
    indices = np.arange(window_size)[None, :] + stride * np.arange(n_windows)[:, None]
    return series[indices]


def main():
    # --- 1) Veri setini oku ---
    df = pd.read_csv(INPUT_FILE)
    print(f"[OK] Veri seti yuklendi: {df.shape[0]:,} satir, {df.shape[1]} kolon")
    print(f"     Zaman serisi kolonu : {TIME_SERIES_COLUMN}")
    print(f"     Etiket kolonu       : {LABEL_COLUMN}")
    print(f"     Gomme boyutu (d)    : {WINDOW_SIZE}")
    print(f"     Kaydirma adimi      : {STRIDE}")
    print(f"     Etiketler           : {df[LABEL_COLUMN].unique().tolist()}")
    print()

    # --- 2) Her etiket grubu icin ayri ayri embedding uygula ---
    all_embeddings = []

    for label, group_df in df.groupby(LABEL_COLUMN, sort=False):
        series = group_df[TIME_SERIES_COLUMN].values.astype(np.float64)
        embedded = create_time_delay_embedding(series, WINDOW_SIZE, STRIDE)

        if embedded.shape[0] == 0:
            print(f"  [!] '{label}' grubu icin yeterli veri yok (n={len(series)}), atlaniyor.")
            continue

        # Etiket sutununu sonuna ekle
        labels = np.full((embedded.shape[0], 1), label, dtype=object)
        block = np.hstack([embedded, labels])
        all_embeddings.append(block)

        print(f"  [OK] '{label}' -> {len(series):>6,} zaman adimi -> {embedded.shape[0]:>6,} pencere olusturuldu")

    # --- 3) Hepsini birlestir ---
    result = np.vstack(all_embeddings)
    print(f"\n[OK] Toplam olusturulan pencere sayisi: {result.shape[0]:,}")

    # --- 4) DataFrame'e cevir ve disa aktar ---
    col_names = [f"d{i+1}" for i in range(WINDOW_SIZE)] + ["Label"]
    result_df = pd.DataFrame(result, columns=col_names)

    # Sayisal sutunlari float'a cevir
    for col in col_names[:-1]:
        result_df[col] = result_df[col].astype(float)

    result_df.to_csv(OUTPUT_FILE, index=False)
    print(f"[OK] Sonuc dosyasi kaydedildi: {OUTPUT_FILE}")
    print(f"     Boyut: {result_df.shape[0]:,} satir x {result_df.shape[1]} kolon")
    print(f"\nIlk 5 satir:\n{result_df.head()}")


if __name__ == "__main__":
    main()
