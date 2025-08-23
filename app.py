from flask import Flask, render_template, send_file, jsonify, request
import os
import pandas as pd
import json
import numpy as np

app = Flask(__name__)

# Data dummy untuk testing - nanti ganti dengan hasil model sebenarnya
DUMMY_DATA = {
    # Kabupaten Bangkalan
    "IDN.11.1.1_1": {"nilai": 12.4, "kategori": "low", "faktorUtama": ["Akses Sanitasi Baik", "ASI Ekslusif Tinggi"]},
    "IDN.11.1.2_1": {"nilai": 25.7, "kategori": "medium", "faktorUtama": ["BBLR Tinggi", "Akses Air Bersih Terbatas"]},
    "IDN.11.1.3_1": {"nilai": 18.2, "kategori": "low", "faktorUtama": ["Imunisasi Belum Lengkap"]},
    "IDN.11.1.4_1": {"nilai": 31.4, "kategori": "high", "faktorUtama": ["Kemiskinan Tinggi", "Sanitasi Buruk", "Asupan Gizi Rendah"]},
    "IDN.11.1.5_1": {"nilai": 27.6, "kategori": "medium", "faktorUtama": ["Pendidikan Ibu Rendah", "BBLR Tinggi"]},
    "IDN.11.1.6_1": {"nilai": 15.3, "kategori": "low", "faktorUtama": ["Akses Kesehatan Baik", "ASI Ekslusif Tinggi"]},
    "IDN.11.1.7_1": {"nilai": 33.8, "kategori": "high", "faktorUtama": ["Kemiskinan Tinggi", "Pendidikan Rendah", "Pola Asuh Kurang"]},
    "IDN.11.1.8_1": {"nilai": 22.1, "kategori": "medium", "faktorUtama": ["Akses Air Bersih Terbatas", "Sanitasi Kurang"]},
    "IDN.11.1.9_1": {"nilai": 13.7, "kategori": "low", "faktorUtama": ["Program Gizi Baik", "ASI Ekslusif Tinggi"]},
    "IDN.11.1.10_1": {"nilai": 29.2, "kategori": "medium", "faktorUtama": ["Akses Pangan Terbatas", "BBLR Tinggi"]},
    "IDN.11.1.11_1": {"nilai": 16.5, "kategori": "low", "faktorUtama": ["Imunisasi Lengkap", "Program Kesehatan Baik"]},
    "IDN.11.1.12_1": {"nilai": 34.6, "kategori": "high", "faktorUtama": ["Sanitasi Buruk", "Asupan Gizi Rendah", "Kemiskinan Tinggi"]},
    
    # Kabupaten Sampang
    "IDN.11.31.1_1": {"nilai": 35.2, "kategori": "high", "faktorUtama": ["Kemiskinan Tinggi", "Sanitasi Buruk", "Pola Asuh Kurang"]},
    "IDN.11.31.2_1": {"nilai": 28.6, "kategori": "medium", "faktorUtama": ["Pendidikan Ibu Rendah", "Akses Air Bersih Terbatas"]},
    "IDN.11.31.3_1": {"nilai": 19.3, "kategori": "low", "faktorUtama": ["ASI Ekslusif Tinggi", "Imunisasi Lengkap"]},
    "IDN.11.31.4_1": {"nilai": 32.7, "kategori": "high", "faktorUtama": ["Asupan Gizi Rendah", "BBLR Tinggi", "Kemiskinan Tinggi"]},
    "IDN.11.31.5_1": {"nilai": 24.8, "kategori": "medium", "faktorUtama": ["Akses Kesehatan Terbatas", "Pendidikan Rendah"]},
    "IDN.11.31.6_1": {"nilai": 17.2, "kategori": "low", "faktorUtama": ["Program Gizi Baik", "Imunisasi Lengkap"]},
    "IDN.11.31.7_1": {"nilai": 30.9, "kategori": "high", "faktorUtama": ["Sanitasi Buruk", "Akses Pangan Terbatas", "Pola Asuh Kurang"]},
    "IDN.11.31.8_1": {"nilai": 26.3, "kategori": "medium", "faktorUtama": ["BBLR Tinggi", "Pendidikan Ibu Rendah"]},
    "IDN.11.31.9_1": {"nilai": 14.7, "kategori": "low", "faktorUtama": ["Akses Kesehatan Baik", "Program Gizi Baik"]},
    "IDN.11.31.10_1": {"nilai": 33.1, "kategori": "high", "faktorUtama": ["Kemiskinan Tinggi", "Asupan Gizi Rendah", "Akses Air Bersih Terbatas"]},
    
    # Kabupaten Pamekasan
    "IDN.11.27.1_1": {"nilai": 16.9, "kategori": "low", "faktorUtama": ["Program Gizi Baik", "Imunisasi Lengkap"]},
    "IDN.11.27.2_1": {"nilai": 24.5, "kategori": "medium", "faktorUtama": ["Pendidikan Ibu Rendah", "BBLR Sedang"]},
    "IDN.11.27.3_1": {"nilai": 32.1, "kategori": "high", "faktorUtama": ["Kemiskinan Tinggi", "Asupan Gizi Rendah", "Sanitasi Buruk"]},
    "IDN.11.27.4_1": {"nilai": 19.8, "kategori": "low", "faktorUtama": ["ASI Ekslusif Tinggi", "Akses Kesehatan Baik"]},
    "IDN.11.27.5_1": {"nilai": 27.3, "kategori": "medium", "faktorUtama": ["Akses Air Bersih Terbatas", "Pendidikan Rendah"]},
    "IDN.11.27.6_1": {"nilai": 14.2, "kategori": "low", "faktorUtama": ["Program Kesehatan Baik", "Imunisasi Lengkap"]},
    "IDN.11.27.7_1": {"nilai": 31.7, "kategori": "high", "faktorUtama": ["Sanitasi Buruk", "Asupan Gizi Rendah", "Pola Asuh Kurang"]},
    "IDN.11.27.8_1": {"nilai": 22.9, "kategori": "medium", "faktorUtama": ["BBLR Tinggi", "Akses Pangan Terbatas"]},
    "IDN.11.27.9_1": {"nilai": 18.3, "kategori": "low", "faktorUtama": ["ASI Ekslusif Tinggi", "Program Gizi Baik"]},
    "IDN.11.27.10_1": {"nilai": 29.5, "kategori": "medium", "faktorUtama": ["Pendidikan Ibu Rendah", "Akses Air Bersih Terbatas"]},
    "IDN.11.27.11_1": {"nilai": 15.7, "kategori": "low", "faktorUtama": ["Imunisasi Lengkap", "Akses Kesehatan Baik"]},
    "IDN.11.27.12_1": {"nilai": 34.2, "kategori": "high", "faktorUtama": ["Kemiskinan Tinggi", "Sanitasi Buruk", "Asupan Gizi Rendah"]},
    
    # Kabupaten Sumenep
    "IDN.11.34.1_1": {"nilai": 13.6, "kategori": "low", "faktorUtama": ["Program Gizi Baik", "Akses Kesehatan Baik"]},
    "IDN.11.34.2_1": {"nilai": 26.8, "kategori": "medium", "faktorUtama": ["BBLR Tinggi", "Pendidikan Ibu Rendah"]},
    "IDN.11.34.3_1": {"nilai": 34.9, "kategori": "high", "faktorUtama": ["Kemiskinan Tinggi", "Sanitasi Buruk", "Akses Air Bersih Terbatas"]},
    "IDN.11.34.4_1": {"nilai": 17.4, "kategori": "low", "faktorUtama": ["ASI Ekslusif Tinggi", "Imunisasi Lengkap"]},
    "IDN.11.34.5_1": {"nilai": 28.3, "kategori": "medium", "faktorUtama": ["Pendidikan Rendah", "Akses Pangan Terbatas"]},
    "IDN.11.34.6_1": {"nilai": 15.9, "kategori": "low", "faktorUtama": ["Program Kesehatan Baik", "Program Gizi Baik"]},
    "IDN.11.34.7_1": {"nilai": 32.6, "kategori": "high", "faktorUtama": ["Asupan Gizi Rendah", "Sanitasi Buruk", "Pola Asuh Kurang"]},
    "IDN.11.34.8_1": {"nilai": 23.7, "kategori": "medium", "faktorUtama": ["BBLR Tinggi", "Akses Air Bersih Terbatas"]},
    "IDN.11.34.9_1": {"nilai": 19.2, "kategori": "low", "faktorUtama": ["Imunisasi Lengkap", "ASI Ekslusif Tinggi"]},
    "IDN.11.34.10_1": {"nilai": 30.4, "kategori": "high", "faktorUtama": ["Kemiskinan Tinggi", "Sanitasi Buruk", "Pendidikan Rendah"]},
    "IDN.11.34.11_1": {"nilai": 21.6, "kategori": "medium", "faktorUtama": ["Akses Kesehatan Terbatas", "BBLR Tinggi"]},
    "IDN.11.34.12_1": {"nilai": 16.3, "kategori": "low", "faktorUtama": ["Program Gizi Baik", "Imunisasi Lengkap"]},
    "IDN.11.34.13_1": {"nilai": 33.8, "kategori": "high", "faktorUtama": ["Kemiskinan Tinggi", "Asupan Gizi Rendah", "Akses Air Bersih Terbatas"]},
    "IDN.11.34.14_1": {"nilai": 25.2, "kategori": "medium", "faktorUtama": ["Pendidikan Ibu Rendah", "Pola Asuh Kurang"]},
    "IDN.11.34.15_1": {"nilai": 18.7, "kategori": "low", "faktorUtama": ["ASI Ekslusif Tinggi", "Program Kesehatan Baik"]}
}

@app.route('/')
def home():
    # Mengirimkan file HTML yang telah dibuat sebagai respons
    return send_file('templates/index.html')

@app.route('/madura.geojson')
def geojson():
    # Mengirimkan file GeoJSON
    return send_file('geojson/madura.geojson')

@app.route('/api/clustering-data')
def clustering_data():
    # Di sini nantinya kode untuk mengakses hasil model clustering
    try:
        # Prioritas 1: Gunakan file JSON jika ada
        if os.path.exists('clustering_results.json'):
            with open('clustering_results.json', 'r') as f:
                data = json.load(f)
            return jsonify(data)
        
        # Prioritas 2: Gunakan file CSV jika ada
        elif os.path.exists('clustering_results.csv'):
            df = pd.read_csv('clustering_results.csv')
            data = {row['region_id']: {
                'nilai': float(row['stunting_value']),
                'kategori': get_category_from_value(float(row['stunting_value'])),
                'faktorUtama': row['main_factors'].split(',') if 'main_factors' in df.columns and pd.notna(row['main_factors']) else []
            } for _, row in df.iterrows()}
            return jsonify(data)
        
        # Prioritas 3: Gunakan data dummy untuk testing
        else:
            print("Menggunakan data dummy untuk testing")
            return jsonify(DUMMY_DATA)
            
    except Exception as e:
        print(f"Error dalam memuat data clustering: {str(e)}")
        return jsonify({'error': str(e)}), 500

def get_category_from_value(value):
    """Menentukan kategori berdasarkan nilai stunting"""
    if value < 20:
        return 'low'
    elif value < 30:
        return 'medium'
    else:
        return 'high'

@app.route('/api/run-model', methods=['POST'])
def run_model():
    """Endpoint untuk menjalankan model clustering (untuk pengembangan)"""
    try:
        # Di sini nanti kode untuk menjalankan model clustering
        # Ini hanya contoh sederhana
        
        # Buat data dummy baru dengan sedikit variasi
        region_ids = [f"IDN.11.{i}.{j}_1" for i in range(1, 35) for j in range(1, 5)]
        np.random.seed(42)  # Untuk reproduktifitas
        
        results = {}
        factors = [
            "ASI Ekslusif Rendah", "BBLR Tinggi", "Imunisasi Tidak Lengkap", 
            "Sanitasi Buruk", "Akses Air Bersih Terbatas", "Kemiskinan Tinggi",
            "Pendidikan Orang Tua Rendah", "Pola Asuh Kurang", "Asupan Gizi Rendah"
        ]
        
        for region_id in region_ids:
            stunting_value = np.random.uniform(10, 40)  # Nilai acak 10-40%
            
            # Pilih 1-3 faktor secara acak
            num_factors = np.random.randint(1, 4)
            selected_factors = np.random.choice(factors, num_factors, replace=False).tolist()
            
            results[region_id] = {
                "nilai": round(stunting_value, 1),
                "kategori": get_category_from_value(stunting_value),
                "faktorUtama": selected_factors
            }
        
        # Simpan hasil ke file
        with open('clustering_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        return jsonify({"status": "success", "message": "Model berhasil dijalankan", "data_count": len(results)})
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)