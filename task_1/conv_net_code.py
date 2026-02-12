import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Generate Synthetic "Malware" Data
# Malware = random noise + vertical line pattern (simulating code structure)
# Benign = pure random noise
def generate_data(n=200):
    X = []
    y = []
    for i in range(n):
        img = np.random.rand(28, 28, 1)
        if i < n // 2: # Malware
            # Add "signature": a vertical line representing a code block
            img[:, 12:16] = 1.0 
            y.append(1) # Class 1: Malware
        else: # Benign
            y.append(0) # Class 0: Benign
        X.append(img)
    return np.array(X), np.array(y)

X, y = generate_data()

# --- NEW: Save the generated data to a file ---
# This allows the examiner to "download" and verify your data
np.savez('aimlFin2026_1_baklaga/task_1/malware_data.npz', X=X, y=y)
print("Dataset saved to malware_data.npz")

# 2. Define CNN Architecture
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2)
])

model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])

# Train
history = model.fit(X, y, epochs=5, verbose=0)

# 3. Save Visualization
plt.figure(figsize=(10,6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title('CNN Malware Classification Performance')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.savefig('aimlFin2026_1_baklaga/task_1/cnn_viz.png')
print("Visualization saved as cnn_viz.png")
