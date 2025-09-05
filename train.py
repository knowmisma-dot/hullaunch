import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tensorflow.keras import models, layers
import xgboost as xgb
import joblib

# Generate synthetic data
np.random.seed(42)
n_customers = 20000
data = pd.DataFrame({
    "age": np.random.randint(18, 60, n_customers),
    "income": np.random.randint(20000, 150000, n_customers),
    "family_size": np.random.randint(1, 6, n_customers),
    "spend_cat_A": np.random.uniform(0, 1, n_customers),
    "spend_cat_B": np.random.uniform(0, 1, n_customers),
    "spend_cat_C": np.random.uniform(0, 1, n_customers)
})

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# Autoencoder
input_dim = X_scaled.shape[1]
encoding_dim = 4

input_layer = layers.Input(shape=(input_dim,))
encoded = layers.Dense(8, activation='relu')(input_layer)
encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
decoded = layers.Dense(8, activation='relu')(encoded)
decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = models.Model(input_layer, decoded)
encoder = models.Model(input_layer, encoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_scaled, X_scaled, epochs=10, batch_size=64, verbose=1)

embeddings = encoder.predict(X_scaled)

# Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
persona_labels = kmeans.fit_predict(embeddings)
data["persona"] = persona_labels

# XGBoost Affinity Model
X = embeddings
y = (data["spend_cat_A"] + data["spend_cat_B"]*2 + data["spend_cat_C"]*3) > 2
y = y.astype(int)
model = xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False)
model.fit(X, y)

# Save artifacts
joblib.dump(scaler, "scaler.pkl")
joblib.dump(kmeans, "kmeans.pkl")
joblib.dump(model, "xgb_model.pkl")
encoder.save("encoder_model.keras")
data.to_csv("customer_data.csv", index=False)
