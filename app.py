import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier



# CONFIG
st.set_page_config(page_title="Stress Analysis", layout="wide")

st.title("📊 Análise de Estresse em Estudantes")



# CARREGAR DATASET
DATA_PATH = "data.csv"

df = pd.read_csv(DATA_PATH)

st.subheader("Dados")
st.dataframe(df.head())



# DETECTAR COLUNA TARGET
possible_targets = ["stress_level", "stress", "Stress_Level", "Stress"]

target_col = None
for col in possible_targets:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    st.error("❌ Nenhuma coluna de stress encontrada!")
    st.write("Colunas disponíveis:", df.columns)
    st.stop()

st.info(f"Coluna alvo detectada: {target_col}")



# CORRELAÇÃO
st.subheader("📊 Correlação")

corr = df.corr()

fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(corr, cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.write("Correlação com o nível de estresse:")
st.write(corr[target_col].sort_values(ascending=False))



# FEATURE IMPORTANCE
st.subheader("Principais causadores de estresse")

X = df.drop(target_col, axis=1)
y = df[target_col]

rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

importances = pd.DataFrame({
    "feature": X.columns,
    "importance": rf.feature_importances_
}).sort_values(by="importance", ascending=False)

st.dataframe(importances)

fig2, ax2 = plt.subplots(figsize=(8,5))
sns.barplot(data=importances.head(10), x="importance", y="feature", ax=ax2)
st.pyplot(fig2)



# MODELOS
st.subheader("🤖 Modelos")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = accuracy_score(y_test, preds)

st.write("📈 Acurácia:")
st.write(results)



# SIMULAÇÃO
st.subheader("Simulação de melhoria")

top_features = importances.head(5)["feature"].tolist()

feature = st.selectbox("Escolha um fator:", top_features)
value = st.slider("Melhoria:", 0, 5, 1)

df_sim = df.copy()
df_sim[feature] += value

X_sim = scaler.transform(df_sim.drop(target_col, axis=1))

model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

preds_sim = model.predict(X_sim)

st.write("Distribuição após melhoria:")
st.write(pd.Series(preds_sim).value_counts())