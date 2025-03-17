import pandas as pd
import numpy as np

def find_optimal_production_year(df, min_sample_size=50, max_age=20):
    df_clean = df.dropna(subset=["Rok_produkcji", "Cena"])
    
    df_clean["Rok_produkcji"] = df_clean["Rok_produkcji"].astype(int)
    
    current_year = 2021
    
    best_year = None
    min_std_dev = float('inf')
    valid_years = []
    
    for start_year in range(current_year - max_age, current_year):
        temp_df = df_clean[df_clean["Rok_produkcji"] >= start_year]
        
        if len(temp_df) < min_sample_size:
            continue
        
        std_dev = temp_df["Cena"].std()
        
        valid_years.append((start_year, len(temp_df), std_dev))
        
        if std_dev < min_std_dev:
            min_std_dev = std_dev
            best_year = start_year
    
    return best_year, valid_years


import pandas as pd
import matplotlib.pyplot as plt

TRAIN_FILE_NAME = "data/sales_ads_train.csv"
df = pd.read_csv(TRAIN_FILE_NAME)

df = df.dropna(subset=["Model_pojazdu", "Rodzaj_paliwa", "Cena", "Rok_produkcji"])

paliwa = {
    "Spalinowe": ["Gasoline", "Diesel"],
    "Hybrydowe": ["Hybrid"],
    "Elektryczne": ["Electric"]
}

df["Paliwo_typ"] = df["Rodzaj_paliwa"].apply(
    lambda x: "Spalinowe" if x in paliwa["Spalinowe"] else "Hybrydowe" if x in paliwa["Hybrydowe"] else "Elektryczne" if x in paliwa["Elektryczne"] else "Inne"
)

model_fuel_counts = df.groupby(["Model_pojazdu", "Paliwo_typ"]).size().unstack(fill_value=0)

models_with_spalinowe = model_fuel_counts[model_fuel_counts["Spalinowe"] > 0].index
models_with_elektryczne_or_hybrydowe = model_fuel_counts[(model_fuel_counts["Elektryczne"] > 0) | (model_fuel_counts["Hybrydowe"] > 0)].index
models_filtered = models_with_spalinowe.intersection(models_with_elektryczne_or_hybrydowe)

filtered_df = df[df["Model_pojazdu"].isin(models_filtered)]

# Znalezienie optymalnego roku produkcji
optimal_year, _ = find_optimal_production_year(filtered_df)
filtered_df = filtered_df[filtered_df["Rok_produkcji"] >= optimal_year]

price_summary = filtered_df.groupby(["Model_pojazdu", "Paliwo_typ"])["Cena"].mean().reset_index()

top_10_models = filtered_df["Model_pojazdu"].value_counts().head(10).index
top_10_df = price_summary[price_summary["Model_pojazdu"].isin(top_10_models)]

pivot_data = top_10_df.pivot(index="Model_pojazdu", columns="Paliwo_typ", values="Cena").fillna(0)

plt.figure(figsize=(14, 8))
pivot_data[["Spalinowe", "Hybrydowe", "Elektryczne"]].plot(kind="bar", width=0.8, colormap="viridis")

plt.title("Średnia cena dla modeli z wersją spalinową oraz hybrydową/elektryczną (od roku {})".format(optimal_year), fontsize=16)
plt.xlabel("Model pojazdu", fontsize=12)
plt.ylabel("Średnia cena", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.legend(title="Typ paliwa")
plt.grid(True, axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()