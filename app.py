"""
app.py - Streamlit app for HUL Virtual Product Trials (Hugging Face Space)
This app will attempt to load pretrained artifacts. If not found, it runs train.py (which saves artifacts).
Designed to be CPU-friendly; models are trained with small epochs in train.py.
"""
import os, json, time, subprocess
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import joblib

st.set_page_config(layout="wide", page_title="HUL Virtual Product Trials")

st.title("HUL Virtual Product Trials — Demo (20k customers)")

# Ensure artifacts present
ARTIFACTS = ['encoder_model.keras','persona_embeddings.npy','xgb_model.pkl','persona_map.csv','cat_columns.npy','final_X.npy','cust_index.npy','cluster_labels.json']
missing = [a for a in ARTIFACTS if not os.path.exists(a)]
if missing:
    st.info("Training artifacts not found. Running `train.py` now — this will create models and data (runs once).")
    with st.spinner("Running training (may take a few minutes)..."):
        # Run train.py in-process
        subprocess.run(['python','train.py'], check=True)
    st.success("Training finished. Reload the app to use models. (Rerun if necessary)")
    time.sleep(1)

# Load artifacts
encoder = None
try:
    import tensorflow as tf
    encoder = tf.keras.models.load_model('encoder_model.keras')
except Exception as e:
    st.error(f"Failed to load encoder: {e}")

embeddings = np.load('persona_embeddings.npy')
xgb_model = joblib.load('xgb_model.pkl')
persona_map = pd.read_csv('persona_map.csv')
with open('cluster_labels.json','r') as f:
    cluster_labels = json.load(f)

cust_index = np.load('cust_index.npy')
final_X = np.load('final_X.npy')
cat_cols = np.load('cat_columns.npy')

# Sidebar: create product
st.sidebar.header("Virtual Product Playground")
prod_name = st.sidebar.text_input("Product name", "Dove Herbal Shampoo")
category = st.sidebar.selectbox("Category", list(cat_cols))
price = st.sidebar.slider("Price (₹)", 10, 499, 149)
pack = st.sidebar.selectbox("Pack type", ["Sachet","Bottle","Tube","Pouch"])
usage = st.sidebar.selectbox("Usage", ["Daily","Weekly","Occasional"])
region_filter = st.sidebar.selectbox("Region (All)", ["All","North","South","East","West"])
run_trial = st.sidebar.button("Run Virtual Trial")

if run_trial:
    st.subheader(f"Running trial: {prod_name} — {category} @ ₹{price}")
    # Build product vector aligned with final_X layout: last len(cat_cols) are TF-IDF category slots
    pvec = np.zeros(final_X.shape[1])
    tail_start = final_X.shape[1] - len(cat_cols)
    for i,c in enumerate(cat_cols):
        pvec[tail_start + i] = 1.0 if c == category else 0.0
    pvec[0] = price / 500.0  # price influence heuristic

    # product embedding
    prod_emb = encoder.predict(pvec.reshape(1,-1))
    sims = cosine_similarity(embeddings, prod_emb).flatten()
    sims_norm = (sims - sims.min()) / (sims.max() - sims.min() + 1e-9)

    purchase_probs = xgb_model.predict_proba(embeddings)[:,1]

    df = pd.DataFrame({'customer_id': cust_index, 'similarity': sims_norm, 'purchase_prob': purchase_probs})
    df = df.merge(persona_map[['customer_id','persona_cluster','persona_label','region']], on='customer_id', how='left')

    if region_filter != 'All':
        df = df[df['region'] == region_filter]

    persona_summary = df.groupby('persona_cluster').agg(customers=('customer_id','count'), avg_similarity=('similarity','mean'), avg_purchase=('purchase_prob','mean')).reset_index()
    persona_summary['label'] = persona_summary['persona_cluster'].apply(lambda x: cluster_labels.get(str(int(x)), f'Cluster {int(x)}'))

    st.markdown("### Persona-level Results")
    st.dataframe(persona_summary[['label','customers','avg_similarity','avg_purchase']].rename(columns={'label':'Persona','customers':'Customer Count','avg_similarity':'AvgCompat','avg_purchase':'AvgPurchaseProb'}))

    st.markdown("### Top Personas")
    top = persona_summary.sort_values('avg_similarity', ascending=False).head(5)
    st.dataframe(top[['label','customers','avg_similarity','avg_purchase']].rename(columns={'label':'Persona','customers':'Count','avg_similarity':'AvgCompat','avg_purchase':'AvgPurchaseProb'}))

    st.markdown("### Explainability (Top contributing categories per persona)")
    for _, r in top.iterrows():
        cl = int(r['persona_cluster'])
        st.markdown(f"**{cluster_labels.get(str(cl),'Cluster')}**")
        member_ids = df[df['persona_cluster']==cl]['customer_id'].values
        idx_map = {int(x):i for i,x in enumerate(cust_index)}
        positions = [idx_map[int(x)] for x in member_ids if int(x) in idx_map]
        if len(positions)==0:
            st.write('No members in cluster (filtered region?)')
            continue
        mean_feat = final_X[positions].mean(axis=0)
        contrib = mean_feat * pvec
        top_idx = np.argsort(contrib)[-5:][::-1]
        names = []
        for ix in top_idx:
            if ix >= tail_start:
                names.append(cat_cols[ix - tail_start])
            elif ix==0:
                names.append('Price influence')
            else:
                names.append(f'feat_{ix}')
        st.write('Top features:', ', '.join(names))
