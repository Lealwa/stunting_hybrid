from flask import Flask, render_template, send_file, jsonify, request
import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict

app = Flask(__name__)

# Fungsi untuk memuat data clustering dari file
def load_clustering_data():
    """Load clustering data prioritizing yearly data format over old format"""
    # Check for yearly clustering data
    if os.path.exists('models/clustering_results_by_year.json'):
        with open('models/clustering_results_by_year.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    # Fall back to the old single-year data if needed
    elif os.path.exists('clustering_results.json'):
        with open('clustering_results.json', 'r', encoding='utf-8') as f:
            # Wrap in a dictionary with a default year key
            data = json.load(f)
            return {"2024": data}  # Use current year as key
    else:
        # Dummy data with years as keys
        print("Clustering results file not found, using dummy data")
        return {
            "2020": {
                "3527120": {"nilai": 24.2, "kategori": "medium", "faktorUtama": ["Tingkat Stunting Sedang"]},
                "3527020": {"nilai": 19.3, "kategori": "low", "faktorUtama": ["Tingkat Stunting Rendah"]}
            },
            "2021": {
                "3527120": {"nilai": 21.8, "kategori": "medium", "faktorUtama": ["Tingkat Stunting Sedang"]},
                "3527020": {"nilai": 21.5, "kategori": "medium", "faktorUtama": ["Tingkat Stunting Sedang"]}
            },
            "2022": {
                "3527120": {"nilai": 19.5, "kategori": "low", "faktorUtama": ["Tingkat Stunting Rendah"]},
                "3527020": {"nilai": 18.2, "kategori": "low", "faktorUtama": ["Tingkat Stunting Rendah"]}
            },
            "2023": {
                "3527120": {"nilai": 17.3, "kategori": "low", "faktorUtama": ["Tingkat Stunting Rendah"]},
                "3527020": {"nilai": 15.8, "kategori": "low", "faktorUtama": ["Tingkat Stunting Rendah"]}
            },
            "2024": {
                "3527120": {"nilai": 16.1, "kategori": "low", "faktorUtama": ["Tingkat Stunting Rendah"]},
                "3527020": {"nilai": 14.2, "kategori": "low", "faktorUtama": ["Tingkat Stunting Rendah"]}
            }
        }

# Load clustering data at app startup
CLUSTERING_DATA = load_clustering_data()

@app.route('/')
def home():
    """Main route that serves the map visualization interface"""
    return send_file('templates/index.html')

@app.route('/madura.geojson')
def geojson():
    """Serve GeoJSON data for Madura"""
    return send_file('geojson/madura.geojson')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files (CSS, JS, images)"""
    return send_file(f'static/{filename}')

@app.route('/api/clustering-data')
def clustering_data():
    """Return clustering data for a specific year or all years"""
    try:
        # Get requested year parameter, defaulting to the most recent year
        year = request.args.get('year', None)
        
        # Load all years data
        all_years_data = CLUSTERING_DATA
        
        # If no year specified, use the most recent year
        if not year:
            # Sort years and get the most recent
            years = sorted(all_years_data.keys())
            if years:
                year = years[-1]
            else:
                return jsonify({"error": "No years available in data"}), 404
        
        # If year specified, return data for that year if it exists
        if year in all_years_data:
            return jsonify(all_years_data[year])
        else:
            return jsonify({"error": f"Data for year {year} not found"}), 404
            
    except Exception as e:
        print(f"Error fetching clustering data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/available-years')
def available_years():
    """Return list of available years in the clustering data"""
    try:
        # Return list of available years
        years = list(CLUSTERING_DATA.keys())
        years.sort()  # Sort years chronologically
        return jsonify({"years": years})
    except Exception as e:
        print(f"Error fetching available years: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/forecasting-data')
def forecasting_data():
    """Return historical and forecasting data from pre-generated JSON file"""
    try:
        # Get region parameter if provided
        kecamatan = request.args.get('region', None)
        
        # Check for pre-generated JSON file
        forecast_file = 'models/lstm_forecast_results.json'
        if os.path.exists(forecast_file):
            with open(forecast_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
        else:
            # Generate simple dummy forecast data if file doesn't exist
            result = generate_dummy_forecast_data()
        
        # If specific kecamatan requested, return just that data
        if kecamatan and kecamatan in result:
            return jsonify(result[kecamatan])
        
        # Otherwise return all data
        return jsonify(result)
        
    except Exception as e:
        print(f"Error fetching forecasting data: {str(e)}")
        return jsonify({'error': str(e)}), 500

def generate_dummy_forecast_data():
    """Generate dummy forecast data when real data is not available"""
    kecamatans = [
        "Banyuates", "Camplong", "Galis", "Jrengik", "KarangPenang",
        "Kedungdung", "Ketapang", "Omben", "Pamekasan", "Proppo", 
        "Sampang", "Sokobanah", "Sreseh", "Tambelangan", "Torjun"
    ]
    
    result = {}
    historical_years = ["2020", "2021", "2022", "2023", "2024"]
    forecast_years = ["2025", "2026", "2027"]
    all_years = historical_years + forecast_years
    
    # Generate data for each kecamatan
    for kec in kecamatans:
        # Start with random base value between 15-35
        base_value = np.random.uniform(15, 35)
        
        # Generate historical data with slight variations
        historical = [max(5, min(40, base_value + np.random.normal(0, 2))) for _ in range(len(historical_years))]
        
        # Generate forecast with decreasing trend (improvement)
        forecast_start = historical[-1] * 0.9  # 10% improvement
        forecast = [
            forecast_start,
            forecast_start * 0.9,
            forecast_start * 0.85
        ]
        
        # Create full arrays with null placeholders
        full_historical = historical + [None] * len(forecast_years)
        full_forecast = [None] * len(historical_years) + forecast
        
        result[kec] = {
            "labels": all_years,
            "historical": full_historical,
            "forecast": full_forecast
        }
    
    # Create aggregate data for all regions
    all_regions = {
        "labels": all_years,
        "historical": [],
        "forecast": []
    }
    
    # Calculate average for all kecamatans for each year
    for i in range(len(all_years)):
        if i < len(historical_years):
            # For historical years, average all kecamatan values
            avg = np.mean([result[kec]["historical"][i] for kec in kecamatans])
            all_regions["historical"].append(avg)
            all_regions["forecast"].append(None)
        else:
            # For forecast years
            all_regions["historical"].append(None)
            # Average all kecamatan forecasts
            forecast_idx = i - len(historical_years)
            avg = np.mean([result[kec]["forecast"][i] for kec in kecamatans])
            all_regions["forecast"].append(avg)
    
    # Add all_regions to result
    result["all_regions"] = all_regions
    
    return result

def get_category_from_value(value):
    """Determine stunting category based on percentage value"""
    if value < 20:
        return 'low'
    elif value < 30:
        return 'medium'
    else:
        return 'high'

@app.route('/api/run-model', methods=['POST'])
def run_model():
    """Endpoint to run clustering model manually (for development)"""
    try:
        year = request.json.get('year', '2024')  # Default to 2024 if not specified
        
        # Generate dummy clustering results for the specified year
        region_ids = ["3527020", "3527030", "3527040", "3527070", "3527080", 
                      "3527090", "3527100", "3527110", "3527120", "3527130", 
                      "3527140", "3527150"]
        
        # Set seed based on year for consistent but different results per year
        np.random.seed(int(year))
        
        results = {}
        factors = [
            "ASI Ekslusif Rendah", "BBLR Tinggi", "Imunisasi Tidak Lengkap", 
            "Sanitasi Buruk", "Akses Air Bersih Terbatas", "Kemiskinan Tinggi",
            "Pendidikan Orang Tua Rendah", "Pola Asuh Kurang", "Asupan Gizi Rendah"
        ]
        
        # Base value that decreases over time (simulating improvement)
        base_value = 35 - (int(year) - 2020) * 3  # Starts at 35% in 2020, decreases by 3% each year
        
        for region_id in region_ids:
            # Value with some random variation
            stunting_value = max(5, min(45, base_value + np.random.normal(0, 5)))
            
            # Select 1-3 factors randomly
            num_factors = np.random.randint(1, 4)
            selected_factors = np.random.choice(factors, num_factors, replace=False).tolist()
            
            # Add category-specific factors
            category = get_category_from_value(stunting_value)
            if category == 'high':
                selected_factors.append("Perlu Intervensi Prioritas")
            elif category == 'medium':
                selected_factors.append("Perlu Pengawasan")
            else:
                selected_factors.append("Kondisi Baik")
            
            results[region_id] = {
                "nilai": round(stunting_value, 1),
                "kategori": category,
                "faktorUtama": selected_factors
            }
        
        # Update global data with new results for this year
        global CLUSTERING_DATA
        if year not in CLUSTERING_DATA:
            CLUSTERING_DATA[year] = {}
        
        CLUSTERING_DATA[year] = results
        
        # Save updated data to file
        with open('models/clustering_results_by_year.json', 'w', encoding='utf-8') as f:
            json.dump(CLUSTERING_DATA, f, indent=2)
            
        return jsonify({
            "status": "success", 
            "message": f"Model berhasil dijalankan untuk tahun {year}", 
            "data_count": len(results)
        })
    
    except Exception as e:
        print(f"Error running model: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/reload-data', methods=['POST'])
def reload_data():
    """Reload clustering data from files"""
    try:
        global CLUSTERING_DATA
        CLUSTERING_DATA = load_clustering_data()
        return jsonify({
            "status": "success", 
            "message": "Data berhasil dimuat ulang", 
            "years": list(CLUSTERING_DATA.keys())
        })
    except Exception as e:
        print(f"Error reloading data: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/debug-data')
def debug_data():
    """Debug endpoint to check loaded data"""
    try:
        return jsonify({
            "years_available": list(CLUSTERING_DATA.keys()),
            "sample_year": list(CLUSTERING_DATA.keys())[0] if CLUSTERING_DATA else None,
            "sample_regions": list(CLUSTERING_DATA[list(CLUSTERING_DATA.keys())[0]].keys())[:3] if CLUSTERING_DATA else [],
            "file_paths": {
                "yearly_data": os.path.exists('models/clustering_results_by_year.json'),
                "legacy_data": os.path.exists('models/clustering_results.json'),
                "forecast_data": os.path.exists('models/lstm_forecast_results.json')
            }
        })
    except Exception as e:
        print(f"Error in debug data: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/generate-data', methods=['POST'])
def generate_data():
    """Generate sample stunting data for multiple years"""
    try:
        start_year = request.json.get('start_year', 2020)
        end_year = request.json.get('end_year', 2024)
        
        years = range(start_year, end_year + 1)
        region_ids = ["3527020", "3527030", "3527040", "3527070", "3527080", 
                      "3527090", "3527100", "3527110", "3527120", "3527130", 
                      "3527140", "3527150"]
        
        # Factors that may contribute to stunting
        factors = [
            "ASI Ekslusif Rendah", "BBLR Tinggi", "Imunisasi Tidak Lengkap", 
            "Sanitasi Buruk", "Akses Air Bersih Terbatas", "Kemiskinan Tinggi",
            "Pendidikan Orang Tua Rendah", "Pola Asuh Kurang", "Asupan Gizi Rendah"
        ]
        
        yearly_data = {}
        
        for year in years:
            np.random.seed(year)  # For reproducibility but different per year
            
            # Base value that decreases over time (simulating improvement)
            base_value = 35 - (year - 2020) * 3  # Starts at 35% in 2020, decreases by 3% each year
            
            year_results = {}
            
            for region_id in region_ids:
                # Value with some random variation
                stunting_value = max(5, min(45, base_value + np.random.normal(0, 5)))
                
                # Select 1-3 factors randomly
                num_factors = np.random.randint(1, 4)
                selected_factors = np.random.choice(factors, num_factors, replace=False).tolist()
                
                # Add category-specific factors
                category = get_category_from_value(stunting_value)
                if category == 'high':
                    selected_factors.append("Perlu Intervensi Prioritas")
                elif category == 'medium':
                    selected_factors.append("Perlu Pengawasan")
                else:
                    selected_factors.append("Kondisi Baik")
                
                year_results[region_id] = {
                    "nilai": round(stunting_value, 1),
                    "kategori": category,
                    "faktorUtama": selected_factors
                }
            
            yearly_data[str(year)] = year_results
        
        # Save data to file
        with open('models/clustering_results_by_year.json', 'w', encoding='utf-8') as f:
            json.dump(yearly_data, f, indent=2)
        
        # Update global data
        global CLUSTERING_DATA
        CLUSTERING_DATA = yearly_data
        
        return jsonify({
            "status": "success",
            "message": f"Generated data for years {start_year}-{end_year}",
            "years_generated": list(yearly_data.keys())
        })
        
    except Exception as e:
        print(f"Error generating data: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)