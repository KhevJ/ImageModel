# %%
from tensorflow import keras
import tensorflow as tf
# import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
import os
from PIL import Image
from pathlib import Path
from tensorflow.keras.models import Sequential

# %%
#Preprocessing
image_count = len(list(Path('/Users/khevinjugessur/Documents/ENEL525/Project/pngImages').glob('*/*.png')))
list_ds = tf.data.Dataset.list_files(str("/Users/khevinjugessur/Documents/ENEL525/Project/pngImages/*/*"), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
print(image_count)
print(len(list_ds))

# %%
batch_size = 16
img_height = 180
img_width = 180

class_names = np.array(sorted([item.name for item in Path('/Users/khevinjugessur/Documents/ENEL525/Project/pngImages').iterdir() if item.name != "LICENSE.txt"]))
print(class_names)


# %%
# val_size = int(image_count * 0.2)
# train_ds = list_ds.skip(val_size)
# val_ds = list_ds.take(val_size)
train_size = int(image_count * 0.7)
val_size = int(image_count * 0.15)
test_size = image_count - train_size - val_size  # rest is for testing

# Split the dataset
train_ds = list_ds.skip(val_size + test_size)  # skip validation and test size
val_ds = list_ds.skip(test_size).take(val_size)  # take the validation size from the remaining
test_ds = list_ds.take(test_size) 

print(tf.data.experimental.cardinality(train_ds).numpy())
print(tf.data.experimental.cardinality(val_ds).numpy())
print(tf.data.experimental.cardinality(test_ds).numpy())

# %%
def get_label(file_path):
  # Convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  one_hot = parts[-2] == class_names
  # Integer encode the label
  return tf.argmax(one_hot)

def decode_img(img):
  # Convert the compressed string to a 3D uint8 tensor
  img = tf.io.decode_png(img, channels=3)
  # Resize the image to the desired size
  return tf.image.resize(img, [img_height, img_width])

def process_path(file_path):
  label = get_label(file_path)
  # Load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label



# %%
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)

for image, label in train_ds.take(1):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())

# %%
def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)

image_batch, label_batch = next(iter(val_ds))

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].numpy().astype("uint8"))
  label = label_batch[i]
  plt.title(class_names[label])
  plt.axis("off")

# %%
num_classes = len(class_names)

# model = Sequential([
#   layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
#   layers.Conv2D(16, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(32, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(64, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Flatten(),
#   layers.Dense(128, activation='relu'),
#   layers.Dense(num_classes)
# ])

# %%
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# %%
# model.summary()

# %%
# epochs=10
# history = model.fit(
#   train_ds,
#   validation_data=val_ds,
#   epochs=epochs
# )

# %%
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()

# %%
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

# %%
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")

# %%
# Model training
model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(128, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.4),#0.2
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dropout(0.4),
  layers.Dense(num_classes, name="outputs")
])

# %%
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

# %%
epochs = 100
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# %%


# history =model.fit(
#   train_ds,
#   validation_data=val_ds,
#   epochs=100,
#   initial_epoch=epochs
# )
# epochs = 100


# %%
#Testing model
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.show()

# %%


def test_configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds
test_ds = list_ds.take(test_size) 
test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_ds = test_configure_for_performance(test_ds)
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# %%


tf.keras.utils.plot_model(model, show_shapes=True)

# %%
test_ds = list_ds.take(test_size) 
test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_ds = test_configure_for_performance(test_ds)

image_batch, label_batch = next(iter(test_ds))
plt.figure(figsize=(12, 10))
for i in range(6):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].numpy().astype("uint8"))
  label = label_batch[i]
  predictions = model.predict(tf.expand_dims(image_batch[i].numpy(), axis=0) )
  score = tf.nn.softmax(predictions[0])
  plt.title(f"{class_names[label]}  {class_names[tf.argmax(score).numpy()]} ")
  plt.axis("off")

# %%
test_ds = list_ds.take(test_size) 
test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_ds = test_configure_for_performance(test_ds)

test_pred= model.predict(test_ds)
test_pred = np.argmax(test_pred, axis=1)


def compute_confusion_matrix(true, pred):
    k = len(np.unique(true)) #Number of classes
    result = np.zeros([k, k])
    for i in range(len(true)):
        result[true[i]][pred[i]] += 1
    return result

test_labels = []
for _, labels in test_ds.unbatch():  # Ensure the dataset is unbatched
    test_labels.append(labels.numpy())
test_labels = np.array(test_labels)



confusion_mx = compute_confusion_matrix(test_labels, test_pred)
print(confusion_mx)
diagonal_predictions = np.trace(confusion_mx)
# print(diagonal_predictions)
# print(len(y_test))
Accuracy = ((diagonal_predictions)/len(test_labels))*100
print("Accuracy:", Accuracy, "%")


