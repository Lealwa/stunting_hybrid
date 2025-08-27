import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import tensorflow.keras.backend as K
import json
import os

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Read and prepare data
print("Loading data...")
data_stunting = pd.read_csv('data/data_stunting_sampang_full.csv', 
                           sep=';',
                           encoding='utf-8',
                           quotechar='"',
                           on_bad_lines='skip')

# Data preprocessing
print("\nPreparing data...")
# Convert percentage to float if necessary
data_stunting['% Stunting'] = data_stunting['% Stunting'].astype(float)

# Group by Puskesmas and take average of stunting metrics
grouped_data = data_stunting.groupby('Puskesmas').agg({
    'Sangat\nPendek': 'mean',
    'Pendek': 'mean',
    'Normal': 'mean',
    'Tinggi': 'mean',
    'Jumlah Balita\nDiukur TB dan\natau PB': 'mean',
    'Jumlah\nBalita\nStunting': 'mean',
    '% Stunting': 'mean'
}).reset_index()

print(f"Found {grouped_data.shape[0]} unique health centers (Puskesmas)")

# Extract features for clustering
features = grouped_data.drop(['Puskesmas'], axis=1)

# Save Puskesmas names for later reference
puskesmas_names = grouped_data['Puskesmas'].values

# Standardize data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# Define parameters
n_clusters = 3  # Setting to 3 clusters: low, medium, high
batch_size = 8  # Smaller batch size for smaller dataset
latent_dim = 2  # Reduced latent dimensions
input_dim = scaled_data.shape[1]
epochs_pretrain = 150
epochs_cluster = 150

# Define the autoencoder model
print("\nBuilding autoencoder model...")
input_layer = Input(shape=(input_dim,))
encoded = Dense(8, activation='relu')(input_layer)
latent = Dense(latent_dim, activation='relu')(encoded)
decoded = Dense(8, activation='relu')(latent)
output_layer = Dense(input_dim, activation='linear')(decoded)

# Autoencoder for pretraining
autoencoder = Model(input_layer, output_layer)
encoder = Model(input_layer, latent)
autoencoder.compile(optimizer='adam', loss='mse')

print("Training autoencoder...")
# Pretrain autoencoder
history = autoencoder.fit(scaled_data, scaled_data, 
                epochs=epochs_pretrain, 
                batch_size=batch_size,
                verbose=1)
                
# Get latent representations
latent_features = encoder.predict(scaled_data)

# Initialize cluster centers using K-means
print("Initializing clusters with K-means...")
kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=0)
kmeans.fit(latent_features)
cluster_centers = kmeans.cluster_centers_

# Define clustering layer
class ClusteringLayer(Layer):
    def __init__(self, n_clusters=n_clusters, alpha=1.0, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        
    def build(self, input_shape):
        self.clusters = self.add_weight(
            shape=(self.n_clusters, input_shape[1]),
            initializer='glorot_uniform',
            name='clusters',
            dtype='float32')
        super(ClusteringLayer, self).build(input_shape)
        
    def call(self, inputs):
        # Student's t-distribution
        q = 1.0 / (1.0 + K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha)
        q = K.pow(q, (self.alpha + 1.0) / 2.0)
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

# Target distribution for DEC
def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

# Build DEC model
print("Building clustering model...")
clustering_layer = ClusteringLayer(n_clusters=n_clusters)
dec_input = Input(shape=(latent_dim,))
dec_output = clustering_layer(dec_input)
dec_model = Model(inputs=dec_input, outputs=dec_output)

# Initialize with correct dtype
cluster_centers_array = np.array(cluster_centers, dtype='float32')
dummy_input = np.zeros((1, latent_dim), dtype='float32')
_ = dec_model.predict(dummy_input)  # This builds the model and creates weights

# Set cluster centers
print("Setting cluster centers...")
clustering_layer.clusters.assign(cluster_centers_array)

# Combined model
combined_input = Input(shape=(input_dim,))
combined_latent = encoder(combined_input)
combined_output = clustering_layer(combined_latent)
combined_model = Model(inputs=combined_input, outputs=combined_output)

# Compile model
combined_model.compile(optimizer=Adam(learning_rate=0.001), loss='kld')

# Initialize cluster predictions
print("Starting DEC training...")
q = combined_model.predict(scaled_data)
p = target_distribution(q)

# Track silhouette scores
silhouette_scores = []

# Train DEC model
print("Training DEC model...")
for epoch in range(epochs_cluster):
    if epoch % 10 == 0:
        q = combined_model.predict(scaled_data)
        p = target_distribution(q)
        y_pred = q.argmax(1)
        
        if epoch > 0:
            silhouette = silhouette_score(latent_features, y_pred)
            silhouette_scores.append(silhouette)
            print(f"Epoch {epoch}: Silhouette Score = {silhouette:.4f}")
    
    combined_model.fit(scaled_data, p, 
                      epochs=1, 
                      batch_size=batch_size,
                      verbose=0)

# Final cluster assignments
print("Generating final cluster assignments...")
q = combined_model.predict(scaled_data)
clusters = q.argmax(1)

# Map clusters to meaningful categories (low, medium, high)
# Sort clusters based on mean stunting percentage
cluster_stunting_means = []
for i in range(n_clusters):
    mask = clusters == i
    cluster_stunting_means.append((i, features.loc[mask, '% Stunting'].mean()))

# Sort clusters by stunting percentage (ascending)
sorted_clusters = sorted(cluster_stunting_means, key=lambda x: x[1])
cluster_mapping = {}
categories = ['low', 'medium', 'high']
for i, (cluster_idx, _) in enumerate(sorted_clusters):
    cluster_mapping[cluster_idx] = categories[i]

print("\nCluster mapping based on stunting percentage:")
for cluster_idx, category in cluster_mapping.items():
    print(f"Cluster {cluster_idx} -> {category} stunting")

# Apply mapping to get categories
cluster_categories = [cluster_mapping[c] for c in clusters]

# Create results dataframe
results = pd.DataFrame({
    'puskesmas': puskesmas_names,
    'cluster': clusters,
    'category': cluster_categories,
    'stunting_percentage': features['% Stunting'].values
})

print("\nClustering Results:")
print(results)

# Print statistics for each cluster
print("\nCluster Statistics:")
for category in ['low', 'medium', 'high']:
    mask = results['category'] == category
    mean_stunting = results.loc[mask, 'stunting_percentage'].mean()
    count = mask.sum()
    print(f"Category {category}: {count} puskesmas, mean stunting: {mean_stunting:.2f}%")

# Mapping from Puskesmas to GeoJSON region IDs
# This mapping would need to be updated with actual region IDs from your GeoJSON file
# For demonstration purposes, using a pattern
region_mapping = {}
for i, puskesmas in enumerate(puskesmas_names):
    # Replace this with actual mapping from puskesmas to GeoJSON region IDs
    region_mapping[puskesmas] = f"IDN.11.27.{i+1}_1"  # Pamekasan region IDs

# Create the final result JSON for the web application
result_json = {}
for _, row in results.iterrows():
    puskesmas = row['puskesmas']
    region_id = region_mapping.get(puskesmas, "unknown")
    
    # Determine factors based on category
    if row['category'] == 'high':
        factors = ["Tingkat Stunting Tinggi", "Perlu Intervensi Prioritas", "Akses Gizi Terbatas"]
    elif row['category'] == 'medium':
        factors = ["Tingkat Stunting Sedang", "Perlu Pengawasan", "Program Gizi Perlu Ditingkatkan"]
    else:
        factors = ["Tingkat Stunting Rendah", "Kondisi Baik", "Program Gizi Berjalan Baik"]
    
    # Store data in the format expected by the web app
    result_json[region_id] = {
        "nilai": float(row['stunting_percentage']),
        "kategori": row['category'],
        "faktorUtama": factors
    }

# Save clustering results for web visualization
os.makedirs('results', exist_ok=True)
with open('clustering_results.json', 'w') as f:
    json.dump(result_json, f, indent=2)

print("\nResults saved to clustering_results.json")

# Visualization using PCA
print("Creating visualizations...")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# Define colors for categories
category_colors = {'low': '#91cf60', 'medium': '#fc8d59', 'high': '#d73027'}

plt.figure(figsize=(12, 8))
for category in ['low', 'medium', 'high']:
    mask = np.array(cluster_categories) == category
    plt.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                c=category_colors[category], label=f"{category.title()} Stunting", alpha=0.7)

# Add labels for each point
for i, txt in enumerate(puskesmas_names):
    plt.annotate(txt, (pca_result[i, 0], pca_result[i, 1]), fontsize=8)


encoder.save('results/encoder_model.h5')
combined_model.save('results/dec_model.h5')
print("Models saved successfully.")
print("Clustering process completed!")