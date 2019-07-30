try:
    from sklearn.externals import joblib
except ImportError:
    import joblib

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


from trains import Task

task = Task.init(project_name="examples", task_name="joblib test")

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()  # sklearn LogisticRegression class
model.fit(X_train, y_train)

joblib.dump(model, 'model.pkl', compress=True)

loaded_model = joblib.load('model.pkl')
result = loaded_model.score(X_test, y_test)
print(result)
