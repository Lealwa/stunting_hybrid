import pandas as pd
import json
from collections import defaultdict

def generate_forecast_json():
    """Generate a JSON file with forecast data for the web application"""
    print("Generating forecast data JSON file...")
    
    # Load historical data
    hist_file = 'data/stunting_kecamatan.csv'
    historical_df = pd.read_csv(hist_file)
    
    # Load prediction data
    pred_file = 'data/prediksi_stunting.csv'
    prediction_df = pd.read_csv(pred_file)
    
    # Combine data for response
    result = {}
    
    # Get all unique kecamatan names
    all_kecamatan = sorted(list(set(historical_df['Kecamatan'].unique()) | 
                               set(prediction_df['Kecamatan'].unique())))
            
    # Process each kecamatan
    for kec in all_kecamatan:
        # Filter data for this kecamatan
        hist_kec = historical_df[historical_df['Kecamatan'] == kec].sort_values('Tahun')
        pred_kec = prediction_df[prediction_df['Kecamatan'] == kec].sort_values('Tahun')
        
        # Create time series data
        years_hist = hist_kec['Tahun'].tolist()
        values_hist = hist_kec['Prevalensi (%)'].tolist()
        
        years_pred = pred_kec['Tahun'].tolist()
        values_pred = pred_kec['Prediksi Prevalensi (%)'].tolist()
        
        # Compile data
        result[kec] = {
            "labels": [str(year) for year in years_hist + years_pred],
            "historical": values_hist + [None] * len(years_pred),
            "forecast": [None] * len(years_hist) + values_pred
        }
    
    # Prepare aggregated data for all regions
    all_years = sorted(list(set(historical_df['Tahun'].unique()) | 
                          set(prediction_df['Tahun'].unique())))
    
    # Calculate averages for each year
    historical_means = historical_df.groupby('Tahun')['Prevalensi (%)'].mean().to_dict()
    forecast_means = prediction_df.groupby('Tahun')['Prediksi Prevalensi (%)'].mean().to_dict()
    
    # Create aggregated data
    result["all_regions"] = {
        "labels": [str(year) for year in all_years],
        "historical": [historical_means.get(year, None) for year in all_years],
        "forecast": [forecast_means.get(year, None) for year in all_years]
    }
    
    # Save to a JSON file
    with open('models/lstm_forecast_results.json', 'w') as f:
        json.dump(result, f, indent=2)
        
    print(f"Forecast data saved to models/lstm_forecast_results.json")
    print(f"Processed {len(all_kecamatan)} kecamatan with forecasts")

if __name__ == "__main__":
    generate_forecast_json()