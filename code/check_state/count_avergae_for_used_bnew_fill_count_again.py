import pandas as pd

df = pd.read_csv("data/sales_ads_train.csv")

mean_price = df[df["Stan"].notna()]["Cena"].mean()
mean_mileage = df[df["Stan"].notna()]["Przebieg_km"].mean()
mean_year = df[df["Stan"].notna()]["Rok_produkcji"].mean()

df["Stan"].fillna(df.apply(lambda row: "New" if row["Przebieg_km"] < 500 else "Used", axis=1), inplace=True)

mean_price_after = df["Cena"].mean()
mean_mileage_after = df["Przebieg_km"].mean()
mean_year_after = df["Rok_produkcji"].mean()



print(f"Mean price before filling: {mean_price}")
print(f"Mean mileage before filling: {mean_mileage}")
print(f"Mean year before filling: {mean_year}")
print("------------------------------------------")
print(f"Mean price after filling: {mean_price_after}")
print(f"Mean mileage after filling: {mean_mileage_after}")
print(f"Mean year after filling: {mean_year_after}")
