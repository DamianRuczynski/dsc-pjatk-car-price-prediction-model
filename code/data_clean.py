import pandas as pd

TRAIN_FILE_NAME = "data/sales_ads_train.csv"
TEST_FILE_NAME = "data/sales_ads_test.csv"
df = pd.read_csv(TRAIN_FILE_NAME)  

drop_columns = [
    "ID", "Lokalizacja_oferty", "Data_pierwszej_rejestracji", "Data_publikacji_oferty", "Wyposazenie", "Emisja_CO2", "Kolor"
]
df = df.drop(columns=drop_columns)

df = df[df["Waluta"].isin(["PLN", "EUR"])]


df["Wiek_pojazdu"] = 2021 - df["Rok_produkcji"]

df.to_csv("data/cleaned_sales_ads_train.csv", index=False)
