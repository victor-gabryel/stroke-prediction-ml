# Stroke Prediction ML - Versão 03

# Análise de Estresse com Regressão Linear

## Visão Geral

Este documento explica detalhadamente cada parte do código utilizado para analisar e prever o nível de estresse em estudantes utilizando regressão linear.

---

## Importações

```python
import streamlit as st
```

Importa a biblioteca Streamlit, usada para criar a interface web interativa.

```python
import pandas as pd
```

Importa o Pandas, utilizado para manipulação e análise de dados em formato de tabelas.

```python
import matplotlib.pyplot as plt
import seaborn as sns
```

Bibliotecas usadas para geração de gráficos e visualizações.

```python
from sklearn.model_selection import train_test_split
```

Função que divide os dados em conjuntos de treino e teste.

```python
from sklearn.preprocessing import StandardScaler
```

Classe responsável por padronizar os dados (normalização).

```python
from sklearn.linear_model import LinearRegression
```

Modelo de regressão linear usado para prever valores contínuos.

```python
from sklearn.metrics import r2_score, mean_squared_error
```

Métricas para avaliar o desempenho do modelo.

---

## Configuração da Aplicação

```python
st.set_page_config(page_title="Análise de Estresse", layout="wide")
```

Define o título da aba do navegador e o layout da página.

```python
st.title("Análise de Estresse em Estudantes")
```

Define o título principal exibido na interface.

---

## Carregamento dos Dados

```python
df = pd.read_csv("data.csv")
```

Carrega o arquivo CSV contendo os dados dos estudantes.

---

## Tradução das Colunas

```python
traducao = { ... }
```

Dicionário que mapeia nomes das colunas em inglês para português.

```python
df = df.rename(columns=lambda x: traducao.get(x, x))
```

Renomeia as colunas do DataFrame usando o dicionário de tradução.

---

## Visualização Inicial

```python
st.subheader("Pré-visualização dos dados")
```

Cria um subtítulo na interface.

```python
st.dataframe(df.head())
```

Exibe as primeiras linhas do dataset.

---

## Definição da Variável Alvo

```python
coluna_alvo = "Nível de Estresse"
```

Define qual coluna será prevista pelo modelo.

```python
if coluna_alvo not in df.columns:
```

Verifica se a coluna existe no dataset.

```python
st.error("Coluna de estresse não encontrada.")
st.stop()
```

Exibe erro e interrompe execução caso não exista.

---

## Preparação dos Dados

```python
df_modelo = df.copy()
```

Cria uma cópia dos dados para evitar alterações no original.

```python
X = df_modelo.drop(coluna_alvo, axis=1)
```

Seleciona as variáveis independentes (entrada do modelo).

```python
y = df_modelo[coluna_alvo]
```

Seleciona a variável alvo (o que será previsto).

---

## Padronização

```python
scaler = StandardScaler()
```

Cria o objeto de padronização.

```python
X_padronizado = scaler.fit_transform(X)
```

Aplica a padronização nos dados (média 0, desvio padrão 1).

---

## Divisão Treino/Teste

```python
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X_padronizado, y, test_size=0.2, random_state=42
)
```

Divide os dados:

* 80% para treino
* 20% para teste
  O `random_state` garante reprodutibilidade.

---

## Treinamento do Modelo

```python
modelo = LinearRegression()
```

Cria o modelo de regressão linear.

```python
modelo.fit(X_treino, y_treino)
```

Treina o modelo com os dados de treino.

---

## Avaliação do Modelo

```python
y_pred = modelo.predict(X_teste)
```

Faz previsões usando os dados de teste.

```python
r2 = r2_score(y_teste, y_pred)
```

Calcula o R², que indica o quanto o modelo explica os dados.

```python
rmse = mean_squared_error(y_teste, y_pred) ** 0.5
```

Calcula o erro médio das previsões.

```python
st.write(...)
```

Exibe os resultados na interface.

---

## Coeficientes do Modelo

```python
coeficientes = pd.DataFrame({
    "Variável": X.columns,
    "Coeficiente": modelo.coef_
})
```

Cria uma tabela com o impacto de cada variável.

```python
.sort_values(by="Coeficiente", ascending=False)
```

Ordena do maior impacto positivo para o menor.

---

## Gráfico de Impacto

```python
sns.barplot(...)
```

Cria um gráfico de barras mostrando os fatores mais influentes.

```python
st.pyplot(fig_coef)
```

Exibe o gráfico no Streamlit.

---

## Correlação

```python
correlacao = df_modelo.corr(numeric_only=True)
```

Calcula a correlação entre todas as variáveis numéricas.

```python
sns.heatmap(...)
```

Exibe um mapa de calor da correlação.

---

## Análise Específica (Depressão)

```python
df.groupby("Depressão")["Nível de Estresse"].mean()
```

Calcula a média do estresse para cada nível de depressão.

---

## Resultados Finais

```python
coeficientes.head(5)
```

Seleciona os principais fatores que aumentam o estresse.

```python
coeficientes.tail(5)
```

Seleciona os fatores que reduzem o estresse.

```python
st.write(...)
```

Exibe os resultados numerados.

---

## Conclusão

```python
st.write(f"...")
```

Gera uma explicação automática com base nos resultados do modelo.

---

## Interpretação Geral

* Coeficientes positivos indicam aumento do estresse
* Coeficientes negativos indicam redução do estresse
* R² indica o quanto o modelo consegue explicar os dados
* RMSE indica o erro médio das previsões

---

## Observação Importante

Os resultados dependem diretamente da qualidade dos dados.
Se os dados estiverem inconsistentes, o modelo pode aprender padrões incorretos.