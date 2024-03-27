import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def read_and_plot(file_name, title):
    # Reading the file and processing the data
    data = []
    with open(file_name, 'r') as file:
        for line in file:
            parts = line.split(',')
            training_accuracy = float(parts[0].split(":")[1].strip())
            testing_accuracy = float(parts[1].split(":")[1].strip())
            data_size = int(parts[2].split(":")[1].strip())
            data.append([training_accuracy, testing_accuracy, data_size])

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['TrainingAccuracy', 'TestingAccuracy', 'DataSize'])

    # Calculate the average accuracy for each data size
    avg_df = df.groupby('DataSize').mean()

    # Plotting
    plt.plot(avg_df.index, avg_df['TrainingAccuracy'], label='Average Training Accuracy')
    plt.plot(avg_df.index, avg_df['TestingAccuracy'], label='Average Testing Accuracy')
    plt.xlabel('Data Size')
    plt.ylabel('Average Accuracy')
    plt.title(title)
    plt.legend()
    plt.show()




def load_matrix(filename):
    with open(filename, 'r') as file:
        matrix = np.array([list(map(float, line.split())) for line in file])
    return matrix

def count_unique(matrix):
    unique_rows = np.unique(matrix, axis=0)
    return len(unique_rows)

# Load matrices
X1 = load_matrix('X_train.txt')
X2 = load_matrix('X_test.txt')

# Print the shape of X1 and X2
print(f"Shape of X1: {X1.shape}")
print(f"Shape of X2: {X2.shape}")

# Count unique vectors
unique_in_X1 = count_unique(X1)
unique_in_X2 = count_unique(X2)
unique_combined = count_unique(np.vstack((X1, X2)))

# Print results
print(f"Unique items in X1: {unique_in_X1}")
print(f"Unique items in X2: {unique_in_X2}")
print(f"Total unique items in both: {unique_combined}")


# Reading loss data
iter, train_loss, test_loss = [], [], []
with open('loss_data.txt', 'r') as file:
    for line in file:
        i, tl, tsl = map(float, line.split())
        iter.append(i)
        train_loss.append(tl)
        test_loss.append(tsl)

# Plotting loss
plt.figure()
plt.plot(iter, train_loss, label='Training Loss')
plt.plot(iter, test_loss, label='Testing Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training vs Testing Loss')
plt.legend()
plt.show()

# Reading loss data
iter, train_loss, test_loss = [], [], []
with open('task2_loss_data.txt', 'r') as file:
    for line in file:
        i, tl, tsl = map(float, line.split())
        iter.append(i)
        train_loss.append(tl)
        test_loss.append(tsl)

# Plotting loss
plt.figure()
plt.plot(iter, train_loss, label='Training Loss')
plt.plot(iter, test_loss, label='Testing Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Task 2 Training vs Testing Loss')
plt.legend()
plt.show()


# Replace 'accuracy_v_datasize_fit.txt' with your actual file name
read_and_plot('accuracy_v_datasize_fit.txt', 'Task 1 Accuracy vs Data Size')
read_and_plot('accuracy_v_datasize_fitMore.txt', 'Task 2 Accuracy vs Data Size')


# the following is my python script. in the script I load in X_train and X_test which will have a shape in the form of Shape of X1: (n_train_diagram, 1601)
# Shape of X2: (n_test_diagrams, 1601) where each diagram is a 1601 long vector that has been one-hot encoded. originally, the diagrams were 20x20 matrices which contained 4 different colored wires randomly placed one at a time in alternating order of rows and columns. this results in a diagram with four different colored wires 2 laid horrizontally and 2 laid vertically. 

# the one-hot encoding works as follows
# empty: [0, 0, 0, 0]
# Red: [1, 0, 0, 0]
# Blue: [0, 1, 0, 0]
# Yellow: [0, 0, 1, 0]
# Green: [0, 0, 0, 1]


