import pandas as pd

df = pd.read_csv("accepted_2007_to_2018Q4.csv", nrows=50000)
df.to_csv("lending_club_sample.csv", index=False)

print(f"Done! Saved {len(df)} rows.")