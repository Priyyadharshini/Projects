import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


data = pd.read_csv("parkinsons.csv")


X = data.drop(['name','status'], axis=1)
y = data['status']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)


model = SVC(kernel='linear')
model.fit(X_train, y_train)


pickle.dump(model, open("model.pkl","wb"))
pickle.dump(scaler, open("scaler.pkl","wb"))

print("Model trained and saved")
