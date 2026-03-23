import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# CONFIG
st.set_page_config(page_title="Análise de Estresse", layout="wide")
st.title("Análise de Estresse em Estudantes")

# CARREGAR DADOS
df = pd.read_csv("data.csv")

# ================================
# TRADUÇÃO DAS COLUNAS
# ================================
traducao = {
    "anxiety_level": "Nível de Ansiedade",
    "self_esteem": "Autoestima",
    "sleep_quality": "Qualidade do Sono",
    "study_hours": "Horas de Estudo",
    "academic_performance": "Desempenho Acadêmico",
    "social_support": "Apoio Social",
    "depression": "Depressão",
    "noise_level": "Nível de Ruído",
    "living_conditions": "Condições de Moradia",
    "stress_level": "Nível de Estresse"
}

# Renomeia as colunas no dataframe
df = df.rename(columns=traducao)

st.subheader("Dados")
st.dataframe(df.head())


# IDENTIFICAR COLUNA ALVO (já traduzida)
coluna_alvo = "Nível de Estresse"

if coluna_alvo not in df.columns:
    st.error("Coluna de estresse não encontrada.")
    st.stop()


# SEPARAÇÃO
X = df.drop(coluna_alvo, axis=1)
y = df[coluna_alvo]


# PADRONIZAÇÃO
scaler = StandardScaler()
X_padronizado = scaler.fit_transform(X)


# TREINO / TESTE
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X_padronizado, y, test_size=0.2, random_state=42
)


# MODELO
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_treino, y_treino)


# IMPORTÂNCIA DAS VARIÁVEIS
importancias = pd.DataFrame({
    "Variável": X.columns,
    "Impacto": abs(modelo.coef_).mean(axis=0)
}).sort_values(by="Impacto", ascending=False)


# ================================
# GRÁFICO (JÁ TRADUZIDO)
# ================================
st.subheader("Principais fatores que aumentam o estresse")

fig, ax = plt.subplots(figsize=(8, 5))

sns.barplot(
    data=importancias.head(10),
    x="Impacto",
    y="Variável",
    ax=ax
)

ax.set_xlabel("Impacto no Estresse")
ax.set_ylabel("Fatores")

st.pyplot(fig)


# ================================
# CORRELAÇÃO (TRADUZIDA)
# ================================
st.subheader("Correlação entre variáveis")

correlacao = df.corr()

fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.heatmap(correlacao, cmap="coolwarm", ax=ax2)

st.pyplot(fig2)

st.write("Correlação com o estresse:")
st.dataframe(correlacao[coluna_alvo].sort_values(ascending=False))


# ================================
# RESULTADO FINAL
# ================================
st.subheader("Principais causadores do estresse")

top_fatores = importancias.head(5)["Variável"].tolist()

for i, fator in enumerate(top_fatores, 1):
    st.write(f"{i}. {fator}")