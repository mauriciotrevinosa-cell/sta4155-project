from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
fires = pd.read_csv(ROOT / "Fires.csv")
salaries = pd.read_csv(ROOT / "Salaries.csv")
print(fires.head())
print(salaries.head())
# Construyan aqui sus modelos y expliquen sus decisiones.
