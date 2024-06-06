import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib

data = pd.read_csv(r"C:\Users\Айгерим\OneDrive\Рабочий стол\aitu\Capstone project\Traffic.csv")

data['Traffic Situation'] = data['Traffic Situation'].replace({'low': 0, 'normal': 1, 'heavy': 2, 'high': 3})
data['Day of the week'] = data['Day of the week'].replace({'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7})

print(data.columns)

if 'Temp' in data.columns:
    data = data.drop(columns=['Temp'], axis=1)

features = ['Date', 'Day of the week', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total']
if 'hour' in data.columns and 'minute' in data.columns and 'AM/PM' in data.columns:
    features.extend(['hour', 'minute', 'AM/PM'])

X = data[features]
y = data['Traffic Situation'].values

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
train_X = sc.fit_transform(train_X)
test_X = sc.transform(test_X)

cl1 = LogisticRegression()
cl2 = SVC(probability=True)
cl3 = RandomForestClassifier()

voting_cl = VotingClassifier(estimators=[('lr', cl1), ('svc', cl2), ('rf', cl3)], voting='hard')

voting_cl.fit(train_X, train_y)

voting_pred = voting_cl.predict(test_X)
accuracy = accuracy_score(test_y, voting_pred)
print(f'Voting Classifier Accuracy: {accuracy}')

joblib.dump(voting_cl, 'voting_model.pkl')
joblib.dump(sc, 'scaler.pkl')
