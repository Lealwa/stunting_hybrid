from flask import Flask, render_template, send_file, jsonify, request
import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict

app = Flask(__name__)

# Fungsi untuk memuat data clustering dari file
def load_clustering_data():
    # Cek di lokasi utama
    if os.path.exists('clustering_results.json'):
        with open('clustering_results.json', 'r') as f:
            return json.load(f)
    # Cek di folder models (sesuai dengan path file Anda)
    elif os.path.exists('models/clustering_results.json'):
        with open('models/clustering_results.json', 'r') as f:
            return json.load(f)
    # Gunakan dummy data jika tidak ada file
    else:
        # Data dummy untuk fallback jika file tidak ditemukan
        print("File clustering_results.json tidak ditemukan, menggunakan data dummy")
        return {
            # Hanya beberapa contoh data dummy
            "IDN.11.27.1_1": {"nilai": 16.9, "kategori": "low", "faktorUtama": ["Program Gizi Baik"]},
            "IDN.11.27.2_1": {"nilai": 24.5, "kategori": "medium", "faktorUtama": ["Pendidikan Ibu Rendah"]},
            "IDN.11.27.3_1": {"nilai": 32.1, "kategori": "high", "faktorUtama": ["Kemiskinan Tinggi"]}
        }

# Muat data clustering saat aplikasi dimulai
CLUSTERING_DATA = load_clustering_data()

@app.route('/')
def home():
    # Mengirimkan file HTML yang telah dibuat sebagai respons
    return send_file('templates/index.html')

@app.route('/api/forecasting-data')
def forecasting_data():
    """Return historical and forecasting data from pre-generated JSON file"""
    try:
        # Get region parameter if provided
        kecamatan = request.args.get('region', None)
        
        # Check for pre-generated JSON file
        forecast_file = 'models/lstm_forecast_results.json'
        if os.path.exists(forecast_file):
            with open(forecast_file, 'r') as f:
                result = json.load(f)
        else:
            # Fall back to direct processing if file doesn't exist
            return jsonify({'error': 'Forecast data not found'}), 404
        
        # If specific kecamatan requested, return just that data
        if kecamatan and kecamatan in result:
            return jsonify(result[kecamatan])
        
        # Otherwise return all data
        return jsonify(result)
        
    except Exception as e:
        print(f"Error fetching forecasting data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_file(f'static/{filename}')

@app.route('/madura.geojson')
def geojson():
    # Mengirimkan file GeoJSON
    return send_file('geojson/madura.geojson')

@app.route('/api/clustering-data')
def clustering_data():
    # Di sini nantinya kode untuk mengakses hasil model clustering
    try:
        # Gunakan data yang sudah dimuat saat startup
        return jsonify(CLUSTERING_DATA)
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
            
        # Perbarui data dalam memori
        global CLUSTERING_DATA
        CLUSTERING_DATA = results
            
        return jsonify({"status": "success", "message": "Model berhasil dijalankan", "data_count": len(results)})
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/reload-data', methods=['POST'])
def reload_data():
    """Endpoint untuk memuat ulang data dari file"""
    try:
        global CLUSTERING_DATA
        CLUSTERING_DATA = load_clustering_data()
        return jsonify({
            "status": "success", 
            "message": "Data berhasil dimuat ulang", 
            "count": len(CLUSTERING_DATA)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/debug-data')
def debug_data():
    """Endpoint untuk debugging data clustering yang dimuat"""
    return jsonify({
        "data_count": len(CLUSTERING_DATA),
        "sample_keys": list(CLUSTERING_DATA.keys())[:3],
        "sample_data": {k: CLUSTERING_DATA[k] for k in list(CLUSTERING_DATA.keys())[:2]},
        "file_paths": {
            "root": os.path.exists('clustering_results.json'),
            "models": os.path.exists('models/clustering_results.json')
        }
    })

if __name__ == '__main__':
    app.run(debug=True)