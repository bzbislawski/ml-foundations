import math
import pandas as pd


def calculate_distance(coordinates_one, coordinates_two):
    dimension = len(coordinates_one)
    result = 0
    for i in range(0, dimension - 1):
        result = result + (coordinates_one[i] - coordinates_two[i]) * (coordinates_one[i] - coordinates_two[i])

    return math.sqrt(result)


dataset = pd.read_csv("nearest_neighbour_dataset.csv")
data = dataset[['weight', 'height', 'label']].values

test_example = [93, 154]
predicted_label = 0
shortest_distance = 10000000
for example in data:
    distance = calculate_distance(test_example, example)
    if distance < shortest_distance:
        shortest_distance = distance
        predicted_label = example[2]

print("Predicted label: " + str(predicted_label))
print("Shortest distance is: " + str(shortest_distance))
