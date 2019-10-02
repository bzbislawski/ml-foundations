import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dataset = pd.read_csv("dataset.csv")
x = dataset[['x', 'y']].values
y = dataset['label'].values

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
# logisticRegr = LogisticRegression()
# logisticRegr = SGDClassifier(alpha=0.0001)
#
# logisticRegr.fit(x_train, y_train)
# logisticRegr.predict(x_test[0].reshape(1, -1))
# logisticRegr.predict(x_test[0:10])
# predictions = logisticRegr.predict(x_test)
# score_train = logisticRegr.score(x_train, y_train)
# score = logisticRegr.score(x_test, y_test)

# print(logisticRegr.predict([[8.1676,4.6457]]))
#
# print(score_train, score)


scores = np.zeros([50,10])
# number of iterations when training
for i in range(50):
    # shuffling the data
    for iteration in range(10):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
        clf = SGDClassifier(loss='log', max_iter=i*2+1)
        clf.fit(x_train, y_train)
        scores[i][iteration] = clf.score(x_test, y_test)

print(scores)
plt.errorbar(list(np.arange(1,101,2)),np.mean(scores,axis=1),yerr=np.std(scores,axis=1))
plt.axis([0, 100, 0.6, 1.05])

plt.show()

