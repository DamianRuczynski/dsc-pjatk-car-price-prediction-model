import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/sales_ads_train.csv")
new_cars = df[(df["Stan"] == "New") & (df["Przebieg_km"] <= 2000) & df["Przebieg_km"].notna()]

plt.figure(figsize=(10, 6))
sns.histplot(new_cars["Przebieg_km"], kde=False, color="skyblue", bins=20)

plt.xticks(range(0, 2100, 100))

plt.title("Range of Mileage for 'New' Cars (Filtered up to 2000 km)", fontsize=16)
plt.xlabel("Mileage (km)", fontsize=14)
plt.ylabel("Frequency", fontsize=14)

plt.show()
