import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("dataset.csv")
x = dataset[['x', 'y']].values
y = dataset['label'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)
logisticRegr.predict(x_test[0].reshape(1, -1))
logisticRegr.predict(x_test[0:10])
predictions = logisticRegr.predict(x_test)
score = logisticRegr.score(x_test, y_test)

print(score)

