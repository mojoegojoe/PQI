import os
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import json
import matplotlib.pyplot as plt
# Disable TensorFlow oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def read_output_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading file: {e}")
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
        Dense(4, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(3, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = LearningRateScheduler(lambda epoch: 0.001 * (0.95 ** epoch), verbose=1)
    checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)
    
    history = model.fit(X_train, y_train, epochs=10001, batch_size=1, verbose=1, validation_split=0.2,
                        callbacks=[early_stopping, lr_scheduler, checkpoint])
    return model, history

def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)
    new_result = {
        "Test Accuracy": accuracy,
        "Test Loss": loss,
        "Predictions": y_pred.flatten().tolist(),
        "True Labels": y_test.tolist()
    }

    try:
        with open("evaluation_results.json", 'r+', encoding='utf-8') as file:
            try:
                results = json.load(file)
            except json.JSONDecodeError:
                results = []
            results.append(new_result)
            file.seek(0)
            json.dump(results, file, indent=4)
    except FileNotFoundError:
        with open("evaluation_results.json", 'w', encoding='utf-8') as file:
            json.dump([new_result], file, indent=4)

def plot_training_history(history, save_plot=True, data_path='./training_data.csv', image_path='./training_plot.png'):
    try:
        # Prepare data for plotting
        if os.path.exists(data_path) and os.path.getsize(data_path) > 0:
            # Load existing data from CSV
            existing_data = np.loadtxt(data_path, delimiter=',', skiprows=1)  # Assuming header exists
            if existing_data.ndim == 1:  # Single row fix
                existing_data = existing_data.reshape(1, -1)
            epochs = np.arange(existing_data.shape[0] + len(history.history['accuracy']))
            accuracies = np.concatenate((existing_data[:, 1], history.history['accuracy']))
            losses = np.concatenate((existing_data[:, 2], history.history['loss']))
        else:
            # Initialize data if none exists
            epochs = np.arange(len(history.history['accuracy']))
            accuracies = history.history['accuracy']
            losses = history.history['loss']

        # Save updated data to CSV
        np.savetxt(data_path, np.column_stack((epochs, accuracies, losses)), delimiter=',', header='Epoch,Accuracy,Loss', comments='', encoding='utf-8')

        # Plotting
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, accuracies, label='Accuracy', color="black")
        plt.title('Model Accuracy History')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, losses, label='Loss', color='grey')
        plt.title('Model Loss History')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()

        # Save plot image
        if save_plot:
            plt.savefig(image_path)
        plt.close()

        print(f"Updated data saved to {data_path} and plot saved to {image_path}")
    except Exception as e:
        print(f"An error occurred: {e}")



def main():
    while True:
        data = read_output_file('output.txt')
        if data:
            quantum_output = data.get('quantum_output', {})
            if quantum_output:
                quantum_data = parse_quantum_output(quantum_output)
                if quantum_data:
                    X_train, X_test, y_train, y_test = preprocess_data(quantum_data)
                    model, history = build_and_train_model(X_train, y_train)
                    plot_training_history(history)
                    evaluate_model(model, X_test, y_test)
                else:
                    print("Quantum data is empty or invalid.")
            else:
                print("No valid quantum output to process.")
        time.sleep(0.1) 

if __name__ == "__main__":
    main()