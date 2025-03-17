import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

df = pd.read_csv("data/cleaned_sales_ads_train.csv")

def compute_price_correlation(df):
    results = []
    grouped = df.groupby(["Marka_pojazdu", "Model_pojazdu", "Pojemnosc_cm3", "Rodzaj_paliwa", "Moc_KM"])
   
    le = LabelEncoder()
    
    for _, group in grouped:
        if len(group) < 2:
            continue
        
        years = group["Rok_produkcji"].dropna().unique()
        if len(years) == 0:
            continue
        year_bins = np.arange(min(years), max(years) + 5, 4)
        group["Year_Bin"] = pd.cut(group["Rok_produkcji"], bins=year_bins, labels=[f"{int(start)}-{int(end-1)}" for start, end in zip(year_bins[:-1], year_bins[1:])], right=False)
        
        for year_range in group["Year_Bin"].dropna().unique():
            subset = group[group["Year_Bin"] == year_range]
            
            if subset["Kraj_pochodzenia"].nunique(dropna=True) > 1:
                subset = subset.copy()
                subset["Kraj_pochodzenia"] = subset["Kraj_pochodzenia"].fillna("Unknown")
                
                subset["Kraj_pochodzenia_encoded"] = le.fit_transform(subset["Kraj_pochodzenia"])
                
                correlation = subset["Kraj_pochodzenia_encoded"].corr(subset["Cena"])
                
                liczba_wystapien = len(subset)
                srednia_cena = subset["Cena"].mean()
                
                results.append({
                    "Marka": subset["Marka_pojazdu"].iloc[0],
                    "Model": subset["Model_pojazdu"].iloc[0],
                    "Pojemnosc": subset["Pojemnosc_cm3"].iloc[0],
                    "Rodzaj_paliwa": subset["Rodzaj_paliwa"].iloc[0],
                    "Moc_KM": subset["Moc_KM"].iloc[0],
                    "Rok_produkcji": year_range,
                    "Korelacja_Cena_Kraj": correlation if not np.isnan(correlation) else None,
                    "Liczba_wystapien": liczba_wystapien,
                    "Srednia_cena": srednia_cena
                })
    
    return pd.DataFrame(results)

correlation_results = compute_price_correlation(df)
print("Koniec: ", correlation_results)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"data/origin/price_correlation_by_country_{timestamp}.csv"

correlation_results.to_csv(filename, index=False)
print(f"Zapisano plik jako: {filename}")