import pandas as pd
import matplotlib.pyplot as plt


df1 = pd.read_csv("uber-raw-data-apr14.csv")
df2 = pd.read_csv("uber-raw-data-may14.csv")
df3 = pd.read_csv("uber-raw-data-jun14.csv")


data = pd.concat([df1, df2, df3])

data['Date/Time'] = pd.to_datetime(data['Date/Time'])

data['Hour'] = data['Date/Time'].dt.hour
data['Month'] = data['Date/Time'].dt.month


data['Hour'].value_counts().sort_index().plot(kind='bar')
plt.title("Trips by Hour")
plt.savefig("static/hour.png")
plt.clf()


data['Month'].value_counts().sort_index().plot(kind='bar')
plt.title("Trips by Month")
plt.savefig("static/month.png")

print("Charts generated")
