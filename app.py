# Importações
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Configurações
st.set_page_config(page_title="Análise de Estresse", layout="wide")
st.title("Análise de Estresse em Estudantes")

# Carregar Dados
df = pd.read_csv("data.csv")

# Tradução das Colunas
traducao = {
    "anxiety_level": "Nível de Ansiedade",
    "self_esteem": "Autoestima",
    "mental_health_history": "Histórico de Saúde Mental",
    "depression": "Depressão",
    "headache": "Dor de Cabeça",
    "blood_pressure": "Pressão Arterial",
    "sleep_quality": "Qualidade do Sono",
    "breathing_problem": "Problemas Respiratórios",
    "noise_level": "Nível de Ruído",
    "living_conditions": "Condições de Moradia",
    "safety": "Segurança",
    "basic_needs": "Necessidades Básicas",
    "academic_performance": "Desempenho Acadêmico",
    "study_load": "Carga de Estudo",
    "teacher_student_relationship": "Relação Professor-Aluno",
    "future_career_concerns": "Preocupação com a Carreira",
    "social_support": "Apoio Social",
    "peer_pressure": "Pressão dos Colegas",
    "extracurricular_activities": "Atividades Extracurriculares",
    "bullying": "Bullying",
    "stress_level": "Nível de Estresse"
}

df = df.rename(columns=lambda x: traducao.get(x, x))

st.subheader("Pré-visualização dos dados")
st.dataframe(df.head())

# Coluna Alvo
coluna_alvo = "Nível de Estresse"

if coluna_alvo not in df.columns:
    st.error("Coluna de estresse não encontrada.")
    st.stop()

# Separação
df_modelo = df.copy()

X = df_modelo.drop(coluna_alvo, axis=1)
y = df_modelo[coluna_alvo]

# Padronização
scaler = StandardScaler()
X_padronizado = scaler.fit_transform(X)

# Divisão treino/teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X_padronizado, y, test_size=0.2, random_state=42
)

# Modelo CORRETO
modelo = LinearRegression()
modelo.fit(X_treino, y_treino)

# ================================
# DESEMPENHO DO MODELO
# ================================
y_pred = modelo.predict(X_teste)

r2 = r2_score(y_teste, y_pred)
rmse = mean_squared_error(y_teste, y_pred) ** 0.5

st.subheader("Desempenho do modelo")
st.write(f"R² (explicação do modelo): {r2:.2f}")
st.write(f"Erro médio (RMSE): {rmse:.2f}")

# ================================
# COEFICIENTES
# ================================
coeficientes = pd.DataFrame({
    "Variável": X.columns,
    "Coeficiente": modelo.coef_
}).sort_values(by="Coeficiente", ascending=False)

st.subheader("Coeficientes do modelo (impacto real)")
st.dataframe(coeficientes)

# Gráfico
st.subheader("Impacto das variáveis no estresse")

fig_coef, ax_coef = plt.subplots(figsize=(10, 6))
sns.barplot(
    data=coeficientes.head(10),
    x="Coeficiente",
    y="Variável",
    ax=ax_coef
)

st.pyplot(fig_coef)

# ================================
# CORRELAÇÃO
# ================================
st.subheader("Correlação entre variáveis")

correlacao = df_modelo.corr(numeric_only=True)

fig2, ax2 = plt.subplots(figsize=(12, 7))
sns.heatmap(correlacao, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)

st.write("Correlação com o estresse:")
st.dataframe(correlacao[coluna_alvo].sort_values(ascending=False))

# ================================
# ANÁLISE DE DEPRESSÃO (DEBUG)
# ================================
st.subheader("Relação entre Depressão e Estresse (média)")

st.dataframe(
    df.groupby("Depressão")["Nível de Estresse"].mean()
)

# ================================
# RESULTADOS
# ================================
st.subheader("Principais fatores que aumentam o estresse")

top_fatores = coeficientes.head(5)["Variável"].tolist()

for i, fator in enumerate(top_fatores, 1):
    st.write(f"{i}. {fator}")

st.subheader("Fatores que reduzem o estresse")

fatores_negativos = coeficientes.tail(5)["Variável"].tolist()

for i, fator in enumerate(fatores_negativos, 1):
    st.write(f"{i}. {fator}")

# ================================
# CONCLUSÃO
# ================================
st.subheader("Conclusão")

st.write(f"""
O modelo de regressão linear foi utilizado para prever o nível de estresse.

O modelo conseguiu explicar aproximadamente {r2*100:.1f}% da variação do estresse.

Coeficientes positivos indicam aumento do estresse,
enquanto coeficientes negativos indicam redução.

Os resultados devem ser interpretados considerando possíveis inconsistências nos dados.
""")