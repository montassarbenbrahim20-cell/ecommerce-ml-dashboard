# ============================================================
# 📚 IMPORT
# ============================================================

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import timedelta
import numpy as np
# ============================================================
# 🎨 CONFIG DESIGN
# ============================================================

st.set_page_config(page_title="E-commerce Dashboard", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛒 E-commerce Sales Dashboard")

# ============================================================
# 📥 LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv("Ecommerce_Sales_Data_2024_2025.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df['order_date'] = pd.to_datetime(df['order_date'])
    return df

df = load_data()

# ============================================================
# 🔎 FILTRES INTERACTIFS
# ============================================================

st.sidebar.header("🔎 Filtres")

regions = st.sidebar.multiselect("Région", df['region'].unique(), default=df['region'].unique())
products = st.sidebar.multiselect("Produit", df['product_name'].unique())

filtered_df = df[df['region'].isin(regions)]

if products:
    filtered_df = filtered_df[filtered_df['product_name'].isin(products)]

# ============================================================
# 📊 KPI (CARTES)
# ============================================================

col1, col2, col3 = st.columns(3)

col1.metric("💰 Ventes Totales", f"{filtered_df['sales'].sum():,.0f}")
col2.metric("📦 Quantité", f"{filtered_df['quantity'].sum():,.0f}")
col3.metric("📈 Profit", f"{filtered_df['profit'].sum():,.0f}")

# ============================================================
# 📈 ÉVOLUTION DES VENTES
# ============================================================

st.subheader("📈 Evolution des ventes")

daily_sales = filtered_df.groupby('order_date')['sales'].sum().reset_index()

st.line_chart(daily_sales.set_index('order_date')['sales'])

# ============================================================
# 📊 TOP PRODUITS
# ============================================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("🏆 Top produits")
    top_products = filtered_df.groupby('product_name')['sales'].sum().sort_values(ascending=False).head(10)
    st.bar_chart(top_products)

# ============================================================
# 🌍 TOP RÉGIONS
# ============================================================

with col2:
    st.subheader("🌍 Top régions")
    top_regions = filtered_df.groupby('region')['sales'].sum().sort_values(ascending=False)
    st.bar_chart(top_regions)

# ============================================================
# 🤖 MODÈLE
# ============================================================

model = joblib.load("model.pkl")
features = joblib.load("features.pkl")
# ============================================================
# 🔮 SIMULATION (1 JOUR)
# ============================================================

st.subheader("🔮 Simulation de prédiction (1 jour)")

col1, col2, col3 = st.columns(3)

month = col1.slider("Mois", 1, 12, 6)
day = col2.slider("Jour", 1, 31, 15)
dayofweek = col3.slider("Jour semaine", 0, 6, 2)

# déterminer weekend automatiquement
is_weekend = 1 if dayofweek >= 5 else 0

# dernières valeurs historiques
last = daily_sales['sales'].iloc[-1]

# rolling features
rm3 = daily_sales['sales'].rolling(3).mean().iloc[-1]
rs3 = daily_sales['sales'].rolling(3).std().iloc[-1]

rm7 = daily_sales['sales'].rolling(7).mean().iloc[-1]
rs7 = daily_sales['sales'].rolling(7).std().iloc[-1]

# input modèle
input_data = pd.DataFrame([[
    month, day, dayofweek, is_weekend,
    last, last, last, last,
    rm3, rs3, rm7, rs7,
    len(daily_sales)
]], columns=features)

# prédiction
prediction = model.predict(input_data)[0]

# stock intelligent
stock = prediction + (rs7 * 1.5)

# affichage
c1, c2 = st.columns(2)
c1.success(f"📈 Vente prévue : {prediction:.2f}")
c2.info(f"📦 Stock recommandé : {stock:.2f}")

# ============================================================
# 📅 PRÉVISION MULTI-JOURS
# ============================================================

import plotly.graph_objects as go
from datetime import timedelta
import numpy as np

st.subheader("📅 Prévision avancée des ventes")

# slider nombre de jours
n_days = st.slider("Nombre de jours à prévoir", 1, 30, 7)

# historique
sales_history = list(daily_sales['sales'].values)
dates_history = list(daily_sales['order_date'])

future_preds = []
future_dates = []

last_date = dates_history[-1]

# boucle prédiction
for i in range(n_days):

    last = sales_history[-1]

    rm3 = np.mean(sales_history[-3:])
    rs3 = np.std(sales_history[-3:])
    
    rm7 = np.mean(sales_history[-7:])
    rs7 = np.std(sales_history[-7:])

    # prochaine date réelle
    next_date = last_date + timedelta(days=1)
    last_date = next_date

    future_dates.append(next_date)

    input_data = pd.DataFrame([[
        next_date.month,
        next_date.day,
        next_date.weekday(),
        1 if next_date.weekday() >= 5 else 0,
        last, last, last, last,
        rm3, rs3, rm7, rs7,
        len(sales_history)
    ]], columns=features)

    pred = model.predict(input_data)[0]

    future_preds.append(pred)
    sales_history.append(pred)

# ============================================================
# 📊 GRAPHIQUE INTERACTIF
# ============================================================

fig = go.Figure()

# historique
fig.add_trace(go.Scatter(
    x=dates_history[-30:],
    y=daily_sales['sales'].values[-30:],
    mode='lines',
    name='Historique'
))

# futur
fig.add_trace(go.Scatter(
    x=future_dates,
    y=future_preds,
    mode='lines+markers',
    name='Prévision',
    line=dict(dash='dash')
))

fig.update_layout(
    title="📈 Prévision des ventes",
    xaxis_title="Date",
    yaxis_title="Sales",
    template="plotly_white"
)

st.plotly_chart(fig)

# ============================================================
# 📊 TABLEAU
# ============================================================

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Vente prédite": future_preds
})

st.dataframe(forecast_df)

# ============================================================
# 📦 STOCK FUTUR
# ============================================================

forecast_df["Stock recommandé"] = forecast_df["Vente prédite"] + (rs7 * 1.5)

st.dataframe(forecast_df)



