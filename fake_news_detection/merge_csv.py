import pandas as pd

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = "FAKE"
true["label"] = "REAL"

data = pd.concat([fake, true], ignore_index=True)

data.to_csv("news.csv", index=False)

print("Merged file saved as news.csv")
