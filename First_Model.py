# First Model and Data Testing
# 12 February 2020

# Import modules
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Retrieve data from pickles
table1 = pd.read_pickle("TRAININGDATA.SALAMP.xlsx K3-Part0.p")
table2 = pd.read_pickle("TRAININGDATA.SALAMP.xlsx K3-Part1.p")
table = table1.append(table2)

# Fix DataFrame column names
table.columns = [v for v in table.columns[:1]] + ["MIC_val"] + [v for v in table.columns[2:]]
dummies = pd.get_dummies(table.Antibiotic)
table = pd.concat([table, dummies], axis=1)
X = table.drop(table.columns[1], axis=1).drop("Antibiotic", axis=1)
y = table.iloc[:, 1]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
y_test_up = pd.DataFrame([str(float(x) + 1) for x in y_test]).iloc[:, 0]
y_test_down = pd.DataFrame([str(float(x) - 1) for x in y_test]).iloc[:, 0]

# Fit Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, max_features="auto", random_state=42)
rf_clf.fit(X_train, y_train)
y_predict_test = rf_clf.predict(X_test)
score = accuracy_score(y_predict_test, y_test)
score += accuracy_score(y_predict_test, y_test_up)
score += accuracy_score(y_predict_test, y_test_down)
print("Random Forest: %f" % score)

