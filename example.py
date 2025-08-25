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

# Read and prepare data
data_stunting = pd.read_csv('data_stunting.csv', 
                           sep=';',
                           encoding='utf-8',
                           quotechar='"',
                           on_bad_lines='skip')

data_stunting = data_stunting.drop(['Nama Lengkap', 'No.', 'Jenis Kelamin (L/P)', 
                                    'Tanggal Lahir (DD-MM-YY)', 'Nama Ortu', 
                                    'Desa Domisili', 'Status Gizi (TB/U)', 'status stunting'], 
                                   axis=1)  

# Convert comma to dot in numeric columns if needed
for col in data_stunting.columns:
    if data_stunting[col].dtype == 'object':
        data_stunting[col] = data_stunting[col].str.replace(',', '.').astype(float)

# Standardize data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_stunting)

# Define parameters
n_clusters = 2
batch_size = 16
latent_dim = 3
input_dim = scaled_data.shape[1]
epochs_pretrain = 100
epochs_cluster = 100

# Define the autoencoder model
input_layer = Input(shape=(input_dim,))
encoded = Dense(10, activation='relu')(input_layer)
latent = Dense(latent_dim, activation='relu')(encoded)
decoded = Dense(10, activation='relu')(latent)
output_layer = Dense(input_dim, activation='linear')(decoded)

# Autoencoder for pretraining
autoencoder = Model(input_layer, output_layer)
encoder = Model(input_layer, latent)
autoencoder.compile(optimizer='adam', loss='mse')

print("Training autoencoder...")
# Pretrain autoencoder
autoencoder.fit(scaled_data, scaled_data, 
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

# Define clustering layer - FIXED VERSION
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
            dtype='float32')  # Explicitly set dtype
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

# Build DEC model - FIXED APPROACH
clustering_layer = ClusteringLayer(n_clusters=n_clusters)
dec_input = Input(shape=(latent_dim,))
dec_output = clustering_layer(dec_input)
dec_model = Model(inputs=dec_input, outputs=dec_output)

# Create a fixed-up way to initialize the weights
# Make sure cluster_centers is a numpy array with the correct dtype
print("Building clustering model...")
cluster_centers_array = np.array(cluster_centers, dtype='float32')
dummy_input = np.zeros((1, latent_dim), dtype='float32')
_ = dec_model.predict(dummy_input)  # This builds the model and creates weights

# Use assign directly rather than K.set_value
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

# Train DEC model
print("Training DEC model...")
for epoch in range(epochs_cluster):
    if epoch % 10 == 0:
        q = combined_model.predict(scaled_data)
        p = target_distribution(q)
        y_pred = q.argmax(1)
        
        if epoch > 0:
            silhouette = silhouette_score(latent_features, y_pred)
            print(f"Epoch {epoch}: Silhouette Score = {silhouette:.4f}")
    
    combined_model.fit(scaled_data, p, 
                      epochs=1, 
                      batch_size=batch_size,
                      verbose=0)

# Final cluster assignments
print("Generating final cluster assignments...")
q = combined_model.predict(scaled_data)
clusters = q.argmax(1)

# Visualization using PCA
print("Creating visualizations...")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='viridis', alpha=0.7)
plt.title('Deep Embedded Clustering of Stunting Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, linestyle='--', alpha=0.7)

# Also visualize in the latent space
plt.figure(figsize=(10, 6))
scatter = plt.scatter(latent_features[:, 0], latent_features[:, 1], c=clusters, cmap='viridis', alpha=0.7)
plt.title('Deep Embedded Clustering in Latent Space')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, linestyle='--', alpha=0.7)

plt.show()

# Print some statistics about the clusters
print("\nCluster Distribution:")
for i in range(n_clusters):
    print(f"Cluster {i}: {np.sum(clusters == i)} samples")

# Compare with original features
cluster_stats = []
for i in range(n_clusters):
    cluster_data = data_stunting.iloc[clusters == i]
    cluster_stats.append(cluster_data.mean())

cluster_stats_df = pd.DataFrame(cluster_stats)
print("\nCluster Centers (Original Features):")
print(cluster_stats_df)