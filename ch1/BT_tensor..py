import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import json
import matplotlib.pyplot as plt

# Disable TensorFlow oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def build_tree_from_structure(structure):
    if not structure:
        return None
    value = structure[0]
    left_subtree = structure[1] if len(structure) > 1 else None
    right_subtree = structure[2] if len(structure) > 2 else None

    node = TreeNode(value)
    node.left = build_tree_from_structure(left_subtree) if left_subtree else None
    node.right = build_tree_from_structure(right_subtree) if right_subtree else None

    return node

def print_tree(node, level=0):
    if node is not None:
        print_tree(node.right, level + 1)
        print(' ' * 4 * level + '->', node.value)
        print_tree(node.left, level + 1)

def read_output_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            data = json.load(file)
            print("Parsed Data:", data)
            return data
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return None

def parse_quantum_output(quantum_output):
    if not quantum_output:
        print("No valid quantum output found.")
        return []
    counts = quantum_output['results'][0]['data']['counts']
    return [int(key, 16) for key in counts.keys()]

def preprocess_data(data):
    X = np.array(data).reshape(-1, 1)
    y = np.array([1 if x % 2 == 1 else 0 for x in data])  
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def build_and_train_model(X_train, y_train):
    model = Sequential([
        Input(shape=(1,)),
        Dense(8, activation='relu'),
        Dense(4, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=1)
    return model, history

def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print("Test Accuracy:", accuracy)

    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)

    print("Predictions:", y_pred.flatten())
    print("True Labels:", y_test)

    # Save evaluation results
    results = {
        "Test Accuracy": accuracy,
        "Predictions": y_pred.flatten().tolist(),
        "True Labels": y_test.tolist()
    }
    with open("evaluation_results.json", 'w', encoding='utf-8') as file:
        json.dump(results, file, indent=4)

def save_output_to_file(file_path, values, quantum_output, quantum_tree_output, mod8_output):
    data = {
        "values": values,
        "quantum_output": quantum_output,
        "quantum_tree_output": quantum_tree_output,
        "mod8_output": mod8_output
    }
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

def convert_to_integers(data):
    return [int(x) for x in data.split(',') if x.strip().isdigit()]

def plot_model_history(history):
    plt.figure(figsize=(12, 8))

    # Plot training & validation accuracy values
    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.grid(True)

    # Plot training & validation loss values
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()

    # Save history data
    history_data = {
        "accuracy": history.history['accuracy'],
        "loss": history.history['loss']
    }
    with open("training_history.json", 'w', encoding='utf-8') as file:
        json.dump(history_data, file, indent=4)

def main():
    print("Reading data.")
    data = read_output_file('output.txt')
    print(f"Type of data: {type(data)}")

    if data is None:
        print("Failed to read data.")
        return

    if not isinstance(data, dict):
        print("Data is not a dictionary.")
        return

    try:
        values = list(map(int, data.get('values', [])))
    except ValueError:
        print("Error converting 'values' to integers.")
        values = []

    quantum_output = data.get('quantum_output', {})

    try:
        quantum_tree_output = convert_to_integers(data.get('quantum_tree_output', ""))
    except ValueError:
        print("Error converting 'quantum_tree_output' to integers.")
        quantum_tree_output = []

    try:
        mod8_output = convert_to_integers(data.get('mod8_output', ""))
    except ValueError:
        print("Error converting 'mod8_output' to integers.")
        mod8_output = []

    print("Values:", values)
    print("Quantum Output:", quantum_output)
    print("Quantum Tree Output:", quantum_tree_output)
    print("Mod8 Output:", mod8_output)
    
    if quantum_output:
        quantum_data = parse_quantum_output(quantum_output)
        if quantum_data:
            print("Quantum Data:", quantum_data)
            X_train, X_test, y_train, y_test = preprocess_data(quantum_data)
            model, history = build_and_train_model(X_train, y_train)
            evaluate_model(model, X_test, y_test)
            quantum_tree_output = quantum_data  
            mod8_output = [x % 8 for x in quantum_data]
            save_output_to_file("outputResults.txt", values, quantum_output, quantum_tree_output, mod8_output)
            plot_model_history(history)
        else:
            print("Quantum data is empty or invalid.")
    else:
        print("No valid quantum output to process.")

if __name__ == "__main__":
    main()
