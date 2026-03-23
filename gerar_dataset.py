import kagglehub
import pandas as pd
import os

dataset_path = "mdsultanulislamovi/student-stress-monitoring-datasets"

# baixa dataset
path = kagglehub.dataset_download(dataset_path)

print("📁 Baixado em:", path)

# lista arquivos
files = os.listdir(path)
print("📄 Arquivos:", files)

# pega CSV automaticamente
csv_file = [f for f in files if f.endswith(".csv")][0]

# lê CSV
df = pd.read_csv(os.path.join(path, csv_file))

print(df.head())

# salva
df.to_csv("data2.csv", index=False)

print("✅ data.csv criado com sucesso!")