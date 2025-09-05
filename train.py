"""
train.py
Generates synthetic ERP data (20k customers), trains a TF-IDF + Keras autoencoder to produce embeddings,
clusters personas with KMeans, trains an XGBoost classifier to predict adoption (heuristic labels),
and saves artifacts used by app.py.

Designed to be run once (at build or first app run). Keeps training light (small epochs) for CPU builds.
"""

import os, json, time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import joblib
import xgboost as xgb

# Keras
import tensorflow as tf
from tensorflow.keras import layers, models

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def generate_data(n_customers=20000, n_products=200, n_txns_per_cust_mean=6, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    regions = ['North','South','East','West']
    city_tiers = [1,2,3]
    genders = ['Female','Male']
    lifestyles = ['Homemaker','Professional','Student','Family','Senior']
    categories = ['Shampoo','Soap','Skincare','Home Care','Foods','Beverages','Oral Care','Deo','Baby']
    pack_types = ['Sachet','Bottle','Tube','Pouch']
    price_points = [10,20,49,79,99,129,149,199,249,299,399]

    customers = []
    for cid in range(1, n_customers+1):
        region = rng.choice(regions)
        city_tier = int(rng.choice(city_tiers, p=[0.35,0.4,0.25]))
        gender = rng.choice(genders, p=[0.48,0.52])
        age = int(np.clip(rng.normal(32,9), 15, 80))
        lifestyle = rng.choice(lifestyles)
        base_spend = {1:6000,2:3200,3:1600}[city_tier]
        spending_power = int(max(400, base_spend + rng.normal(0, base_spend*0.3)))
        customers.append([cid, age, gender, region, city_tier, lifestyle, spending_power])
    customers = pd.DataFrame(customers, columns=['customer_id','age','gender','region','city_tier','lifestyle','spending_power'])
    customers.to_csv('customers.csv', index=False)

    # products
    products = []
    for pid in range(1, n_products+1):
        cat = rng.choice(categories)
        price = int(rng.choice(price_points))
        pack = rng.choice(pack_types)
        usage = rng.choice(['Daily','Weekly','Occasional'], p=[0.7,0.2,0.1])
        premium = 1 if price >= 149 else 0
        products.append([pid, f'Brand{rng.integers(1,50)}', cat, price, pack, usage, premium])
    products = pd.DataFrame(products, columns=['product_id','brand','category','price','pack_type','usage_occasion','premium_flag'])
    products.to_csv('products.csv', index=False)

    # transactions
    txns = []
    tx_id = 1
    for _, cust in customers.iterrows():
        n_txns = max(1, int(rng.poisson(n_txns_per_cust_mean)))
        for _ in range(n_txns):
            prod = products.sample(1, random_state=int(rng.integers(1,1<<30))).iloc[0]
            qty = int(max(1, rng.integers(1,3)))
            amount = qty * prod['price']
            date = datetime.now() - timedelta(days=int(rng.integers(0,365)))
            txns.append([tx_id, int(cust['customer_id']), int(prod['product_id']), qty, float(amount), date.date().isoformat()])
            tx_id += 1
    txns = pd.DataFrame(txns, columns=['transaction_id','customer_id','product_id','qty','amount','date'])
    txns['date'] = pd.to_datetime(txns['date'])
    txns.to_csv('txns.csv', index=False)

    return customers, products, txns

def build_features(customers, products, txns):
    tx = txns.merge(products[['product_id','category','price','premium_flag','usage_occasion','pack_type']], on='product_id', how='left')
    tx['price_band'] = pd.cut(tx['price'], bins=[-1,50,100,150,250,1e9], labels=['<=50','51-100','101-150','151-250','>250'])
    cust_cat = tx.pivot_table(index='customer_id', columns='category', values='amount', aggfunc='sum', fill_value=0)
    cust_prem = tx.groupby(['customer_id','premium_flag'])['amount'].sum().unstack(fill_value=0)
    if cust_prem.shape[1]==0:
        cust_prem = pd.DataFrame(0, index=cust_cat.index, columns=[0,1])
    cust_prem.columns = [f'premium_{int(c)}' for c in cust_prem.columns]
    cust_usage = tx.pivot_table(index='customer_id', columns='usage_occasion', values='qty', aggfunc='sum', fill_value=0)
    cust_pack = tx.pivot_table(index='customer_id', columns='pack_type', values='qty', aggfunc='sum', fill_value=0)
    cust_demo = customers.set_index('customer_id')[['age','gender','region','city_tier','spending_power']]
    cust_feat = cust_demo.join([cust_cat, cust_prem, cust_usage, cust_pack]).fillna(0)
    cust_feat.to_csv('cust_feat.csv')
    return cust_feat, cust_cat, tx

def tfidf_and_matrix(cust_feat, cust_cat):
    tfidf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True)
    cat_arr = cust_cat.values.astype(float)
    cat_tfidf = tfidf.fit_transform(cat_arr).toarray()
    other_num = cust_feat.drop(columns=cust_cat.columns, errors='ignore').select_dtypes(include=[float,int]).fillna(0)
    scaler = StandardScaler()
    final_X = np.hstack([other_num.values, cat_tfidf])
    final_X = scaler.fit_transform(final_X)
    np.save('final_X.npy', final_X)
    np.save('cust_index.npy', np.array(list(other_num.index)))
    np.save('cat_columns.npy', np.array(cust_cat.columns.tolist()))
    joblib.dump(scaler, 'scaler.pkl')
    return final_X, other_num

def train_autoencoder(X, encoding_dim=64, epochs=10):
    input_dim = X.shape[1]
    from tensorflow.keras import layers, models
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation='relu')(inp)
    x = layers.Dense(128, activation='relu')(x)
    encoded = layers.Dense(encoding_dim, activation='relu', name='bottleneck')(x)
    x = layers.Dense(128, activation='relu')(encoded)
    x = layers.Dense(256, activation='relu')(x)
    out = layers.Dense(input_dim, activation='linear')(x)
    autoencoder = models.Model(inp, out)
    encoder = models.Model(inp, encoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X, X, epochs=epochs, batch_size=512, validation_split=0.02, verbose=1)
    embeddings = encoder.predict(X, batch_size=512)
    np.save('persona_embeddings.npy', embeddings)
    encoder.save('encoder_model.keras')
    return embeddings

def discover_personas(embeddings, customers, tx, n_personas=8):
    kmeans = KMeans(n_clusters=n_personas, n_init=10, random_state=RANDOM_SEED)
    clusters = kmeans.fit_predict(embeddings)
    cust_index = np.load('cust_index.npy')
    persona_map = pd.DataFrame({'customer_id': cust_index, 'persona_cluster': clusters})
    persona_map = persona_map.merge(customers.set_index('customer_id'), left_on='customer_id', right_index=True, how='left')
    def label_cluster(df):
        avg_spend = df['spending_power'].mean()
        median_tier = int(df['city_tier'].median()) if 'city_tier' in df.columns else 2
        top_cat = tx[tx['customer_id'].isin(df['customer_id'])].groupby('category')['amount'].sum().sort_values(ascending=False)
        top_cat_name = top_cat.index[0] if not top_cat.empty else 'General'
        if avg_spend >= 6000:
            prefix = 'Young Urban Professional'
        elif avg_spend >= 3500:
            prefix = 'Value-Seeking Family'
        else:
            prefix = 'Price-Sensitive Homemaker'
        return f"{prefix} – Tier {median_tier}, prefers {top_cat_name}"
    cluster_labels = {}
    for c in range(n_personas):
        members = persona_map[persona_map['persona_cluster']==c]
        cluster_labels[c] = label_cluster(members) if len(members)>0 else f'Cluster {c}'
    persona_map['persona_label'] = persona_map['persona_cluster'].map(cluster_labels)
    persona_map.to_csv('persona_map.csv', index=False)
    with open('cluster_labels.json','w') as f:
        json.dump({str(k):v for k,v in cluster_labels.items()}, f)
    return persona_map, cluster_labels

def train_xgboost(embeddings, cust_feat):
    # Heuristic adoption label: combine spending_power and category activity
    cust_index = np.load('cust_index.npy')
    cust_df = cust_feat.reindex(cust_index).fillna(0)
    cat_sum = cust_df.select_dtypes(include=[np.number]).sum(axis=1).values
    spend = cust_df['spending_power'].values if 'spending_power' in cust_df.columns else np.zeros(len(cat_sum))
    score = (spend / (spend.max()+1e-9)) * 0.6 + (cat_sum / (cat_sum.max()+1e-9)) * 0.4
    y = (score + 0.15 * np.random.rand(len(score))) > 0.5
    y = y.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(embeddings, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.08, use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)
    joblib.dump(clf, 'xgb_model.pkl')
    return clf

def main():
    if os.path.exists('xgb_model.pkl') and os.path.exists('encoder_model.keras'):
        print('Artifacts already exist. Skipping training.')
        return
    customers, products, txns = generate_data(n_customers=20000)
    cust_feat, cust_cat, tx = build_features(customers, products, txns)
    final_X, other_num = tfidf_and_matrix(cust_feat, cust_cat)
    embeddings = train_autoencoder(final_X, encoding_dim=64, epochs=8)
    persona_map, cluster_labels = discover_personas(embeddings, customers, tx, n_personas=8)
    clf = train_xgboost(embeddings, cust_feat)
    print('Training complete. Artifacts saved: encoder_model.keras, persona_embeddings.npy, xgb_model.pkl, persona_map.csv')

if __name__ == '__main__':
    main()
