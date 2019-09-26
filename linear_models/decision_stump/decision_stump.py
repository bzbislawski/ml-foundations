import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ballet_dance = 0
rugby_player = 1


def decision_stump(data):
    print('Decision stump algorithm')
    lowest_error_number_record = 99999
    best_t = 0
    errors_landscape_array = []
    for t in range(1, 120):
        number_of_errors_for_function = 0
        for data_element in data:
            weight = data_element[0]
            profession = data_element[2]
            if not (t > weight and profession == ballet_dance) and not (t < weight and profession == rugby_player):
                number_of_errors_for_function += 1

        errors_landscape_array.append([number_of_errors_for_function])
        if number_of_errors_for_function <= lowest_error_number_record:
            lowest_error_number_record = number_of_errors_for_function
            best_t = t
    return best_t, np.array(errors_landscape_array)


dataset = pd.read_csv("decision_stump_dataset.csv")
data = dataset[['weight', 'height', 'profession']].values
data_height = dataset[['height']].values
data_weight = dataset[['weight']].values

t, errors_landscape = decision_stump(data)
print("Best t is: " + str(t))


plt.subplot(1, 2, 1)
plt.plot(data_weight, data_height, 'ro')
plt.axis([30, 130, 110, 190])
plt.title('Decision stump')
plt.xlabel('weight')
plt.ylabel('height')

y = np.linspace(110, 190, 300)
x = 0 * y + t
plt.plot(x, y, '--r', label='t')

plt.subplot(1, 2, 2)
plt.axis([1, 120, 0, 5])
plt.plot(np.linspace(1, 120, 119), errors_landscape, '.-')

plt.title('Error landscape')
plt.xlabel('t')
plt.ylabel('errors count')

plt.show()
