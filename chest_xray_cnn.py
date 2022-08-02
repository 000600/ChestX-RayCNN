import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, AUC
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt

# Define paths
train_path = ' < PATH TO TRAIN SET IMAGES > '
test_path = ' < PATH TO TEST SET IMAGES > '
val_path = ' < PATH TO VALIDATION SET IMAGES > '

# Set batch size and epochs
batch_size = 64
epochs = 10

# Load training data
train_generator = ImageDataGenerator(rescale = 1 / 255, zoom_range = 0.01, rotation_range = 0.05, width_shift_range = 0.05, height_shift_range = 0.05)
train_iter = train_generator.flow_from_directory(train_path, class_mode = 'categorical', color_mode = 'grayscale', batch_size = batch_size)

# Load test data
test_generator = ImageDataGenerator(rescale = 1 / 255)
test_iter = test_generator.flow_from_directory(test_path, class_mode = 'categorical', color_mode = 'grayscale', batch_size = batch_size)

# Load validation data
val_generator = ImageDataGenerator(rescale = 1 / 255)
val_iter = val_generator.flow_from_directory(val_path, class_mode = 'categorical', color_mode = 'grayscale', batch_size = batch_size)

# Define classes
class_map = {0 : "COVID-19", 1 : "Normal", 2 : 'Pneumonia', 3 : "Tuberculosis"}

# Initialize Adam Optimizer
opt = Adam(learning_rate = 0.001)

# Create model
model = Sequential()

# Input layer
model.add(Input(train_iter.image_shape))

# Image processing layers
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu'))
model.add(Conv2D(filters = 30, kernel_size = (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = 2, strides = 2, padding = 'valid'))
model.add(Conv2D(filters = 30, kernel_size = (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = 2, strides = 2, padding = 'valid'))
model.add(Conv2D(filters = 30, kernel_size = (3, 3), activation = 'relu'))

# Flatten layer
model.add(Flatten())

# Hidden layers
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation = 'relu'))

# Output layer
model.add(Dense(4, activation = 'softmax'))  # Softmax activation function since the model is multiclass

# Configure early stopping
early_stopping = EarlyStopping(min_delta = 0.001, patience = 10, restore_best_weights = True)

# Compile and train model
model.compile(optimizer = opt, loss = CategoricalCrossentropy(), metrics = [CategoricalAccuracy(), AUC()])
history = model.fit(train_iter, steps_per_epoch = int(round(train_iter.samples / train_iter.batch_size)), epochs = epochs, validation_data = val_iter, validation_steps = int(round(val_iter.samples / batch_size)))

# Visualize  loss and validation loss
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epoch_list = [i for i in range(epochs)]

plt.plot(epoch_list, loss, label = 'Loss')
plt.plot(epoch_list, val_loss, label = 'Validation Loss')
plt.title('Validation and Training Loss Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Visualize accuracy and validation accuracy
accuracy = history_dict['categorical_accuracy']
val_accuracy = history_dict['val_categorical_accuracy']

plt.plot(epoch_list, accuracy, label = 'Training Accuracy')
plt.plot(epoch_list, val_accuracy, label =' Validation Accuracy')
plt.title('Validation and Training Accuracy Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Visualize AUC and validation AUC
auc = history_dict['auc']
val_auc = history_dict['val_auc']

plt.plot(epoch_list, auc, label = 'Training AUC')
plt.plot(epoch_list, val_auc, label = 'Validation AUC')
plt.title('Validation and Training AUC Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.legend()
plt.show()

# View test accuracy
test_loss, test_acc, test_auc = model.evaluate(test_iter, verbose = 0) # Change verbose to 1 or 2 for more information
print(f'\nTest accuracy: {test_acc * 100}%')

# View model's predictions compared to actual labels

# Get inputs
sample_inputs, sample_labels = val_iter.next()

# Change this number to view more or less input images and corresponding predictins and lables
num_viewed_inputs = 10

# Get inputs and corresponding labels and predictions
sample_inputs = sample_inputs[:num_viewed_inputs]
sample_labels = sample_labels[:num_viewed_inputs]
sample_predictions = model.predict(sample_inputs)

# Combine lists
img_pred_label = enumerate(zip(sample_inputs, sample_predictions, sample_labels))

# Loop through combined list to display the image, the model's prediction on that image, and the actual label of that image
for i, (img, pred, label) in img_pred_label:
  # Model's prediction on sample photo
  predicted_class = np.argmax(pred) # Get the index with the highest probability assigned to it by the model
  certainty = pred[predicted_class]

  # Actual values
  actual_class = np.argmax(label)

  # View results
  print(f"Model's Prediction ({certainty}% certainty): {predicted_class} ({class_map[predicted_class]}) | Actual Class: {actual_class} ({class_map[actual_class]})")

  # Visualize input images
  plt.axis('off')
  plt.imshow(img[:, :, 0], cmap = 'gray')
  plt.tight_layout()
  plt.show()
