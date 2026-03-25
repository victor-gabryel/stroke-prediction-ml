# Análise de Estresse em Estudantes — README

## Visão Geral

Este projeto utiliza Machine Learning (Regressão Linear) para analisar os principais fatores que influenciam o nível de estresse em estudantes.

A aplicação foi desenvolvida com Streamlit, permitindo uma interface interativa com visualizações e análise de dados.

---

## Tecnologias Utilizadas

* Python
* Streamlit
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

---

## Estrutura do Projeto

```
📁 stroke-prediction-ml/
│__ app.py
│__ data.csv
|__ gerar_dataset.py
│__ README.md
|__ requirements.txt
```

---

## Como Executar

1. Instale as dependências:

```bash
pip install streamlit -r requirements.txt
```

2. Execute o projeto:

```bash
streamlit run app.py
```

---

## Etapas do Código

### 1. Importação das Bibliotecas

```python
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

Responsáveis por:

* Interface web (Streamlit)
* Manipulação de dados (Pandas)
* Visualização (Matplotlib e Seaborn)

---

### 2. Configuração da Página

```python
st.set_page_config(page_title="Análise de Estresse", layout="wide")
st.title("Análise de Estresse em Estudantes")
```

Define:

* Título da página
* Layout expandido
* Título principal

---

### 3. Carregamento dos Dados

```python
df = pd.read_csv("data.csv")
```

Lê o dataset contendo informações dos estudantes.

---

### 4. Tradução das Colunas

```python
df = df.rename(columns=lambda x: traducao.get(x, x))
```

Converte nomes das colunas de inglês para português para melhor interpretação.

---

### 5. Pré-visualização dos Dados

```python
st.dataframe(df.head())
```

Mostra as primeiras linhas do dataset.

---

### 6. Definição da Variável Alvo

```python
coluna_alvo = "Nível de Estresse"
```

Essa é a variável que o modelo irá prever.

---

### 7. Separação dos Dados

```python
X = df_modelo.drop(coluna_alvo, axis=1)
y = df_modelo[coluna_alvo]
```

* X: variáveis independentes
* y: variável dependente (estresse)

---

### 8. Padronização dos Dados

```python
scaler = StandardScaler()
X_padronizado = scaler.fit_transform(X)
```

Normaliza os dados para melhorar o desempenho do modelo.

---

### 9. Divisão em Treino e Teste

```python
train_test_split(...)
```

Divide os dados em:

* 80% treino
* 20% teste

---

### 10. Treinamento do Modelo

```python
modelo = LinearRegression()
modelo.fit(X_treino, y_treino)
```

Cria e treina o modelo de regressão linear.

---

## Avaliação do Modelo

```python
r2_score
mean_squared_error
```

* R²: quanto o modelo explica os dados
* RMSE: erro médio da previsão

---

## Análise dos Coeficientes

```python
modelo.coef_
```

Indica o impacto de cada variável:

* Valores positivos aumentam o estresse
* Valores negativos reduzem o estresse

---

## Visualizações

### Gráfico de Impacto

Mostra as variáveis que mais influenciam o estresse.

### Heatmap de Correlação

Exibe a relação entre todas as variáveis do dataset.

---

## Análise Específica

```python
df.groupby("Depressão")["Nível de Estresse"].mean()
```

Mostra a média de estresse baseada na depressão.

---

## Resultados

O sistema identifica automaticamente:

### Fatores que Aumentam o Estresse

Baseado nos maiores coeficientes

### Fatores que Reduzem o Estresse

Baseado nos menores coeficientes

---

## Conclusão

* O modelo utiliza regressão linear para prever o estresse
* O R² indica o nível de explicação dos dados
* Os coeficientes mostram quais fatores mais influenciam

Importante:
Os resultados dependem da qualidade dos dados. Dados inconsistentes podem afetar a precisão do modelo.