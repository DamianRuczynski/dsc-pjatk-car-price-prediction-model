import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

df = pd.read_csv("data/sales_ads_train.csv")

current_year = datetime.now().year
df['Wiek_samochodu'] = current_year - df['Rok_produkcji']

used_cars_below_100km_older_than_4 = df[(df["Stan"] == "Used") & 
                                         (df["Przebieg_km"] < 100) & 
                                         (df["Wiek_samochodu"] > 4) & 
                                         df["Przebieg_km"].notna()]

print(f"Liczba samochodów 'Used' z przebiegiem poniżej 100 km oraz wiekiem starszym niż 4 lata: {len(used_cars_below_100km_older_than_4)}")
used_cars = df[(df["Stan"] == "Used") & df["Przebieg_km"].notna()]
plt.figure(figsize=(10, 6))
sns.histplot(used_cars["Przebieg_km"], kde=False, color="lightcoral", bins=30)
plt.xticks(range(0, int(used_cars["Przebieg_km"].max()) + 10000, 10000))
plt.title("Mileage Distribution for 'Used' Cars", fontsize=16)
plt.xlabel("Mileage (km)", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.show()
