import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

x = np.array([100, 120, 150, 170, 200, 200, 202, 203, 205, 210, 215, 250, 270, 300, 305, 310])
y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1])


logreg = LogisticRegression(C=1.0, solver='lbfgs', multi_class='ovr')
x = x.reshape(-1, 1)
logreg.fit(x, y)

prices = np.arange(130, 280, 0.5)
probabilities = []
for i in prices:
    p_loss, p_win = logreg.predict_proba([[i]])[0]
    probabilities.append(p_win)

print(probabilities)
plt.scatter(prices, probabilities)
plt.title("Logistic Regression Model")
plt.xlabel('Price')
plt.ylabel('Status (1:Won, 0:Lost)')
plt.show()
