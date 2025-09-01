import json
import geopandas as gpd

# Read your existing GeoJSON
gdf = gpd.read_file("geojson/indonesia4.geojson")

# Print available columns to verify
print("Available columns:", gdf.columns.tolist())

# Filter for Madura regencies
madura_regencies = ['Bangkalan', 'Sampang', 'Pamekasan', 'Sumenep']
madura_gdf = gdf[gdf['NAME_2'].isin(madura_regencies)]

# Check if we found any features
print(f"Found {len(madura_gdf)} features for Madura")

# Save as a new GeoJSON file if features were found
if len(madura_gdf) > 0:
    madura_gdf.to_file("madura.geojson", driver="GeoJSON")
    print("Successfully created madura.geojson")
else:
    print("No Madura regencies found in the data. Check the actual regency names in your data:")
    print(gdf['NAME_2'].unique())