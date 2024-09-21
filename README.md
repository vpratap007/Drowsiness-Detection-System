

---

# Drowsiness Detection using CNN and ResNet50

This project focuses on building a deep learning model to detect drowsiness based on eye states (open/closed). It employs two different Convolutional Neural Networks (CNNs): a custom-built CNN and a transfer learning model based on ResNet50.

## Table of Contents

- [Dataset](#dataset)
- [Requirements](#requirements)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [How to Use](#how-to-use)
- [References](#references)

## Dataset

The dataset contains images categorized into two classes:
1. **Open Eyes**
2. **Closed Eyes**

There are approximately 48,000 images in total, split as follows:
- **Training Set:** 38,400 images
- **Validation Set:** 9,600 images

You can replace the dataset paths in the code with your local directory paths.

## Requirements

Ensure you have the following installed:

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- OpenCV
- PIL (Pillow)

You can install the dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Model Architecture

### CNN Model

The CNN model uses three convolutional layers with ReLU activations, followed by max pooling layers. The model ends with a fully connected layer and a softmax output layer for classification.

- Input size: (32, 32, 3)
- 32 filters in the first two Conv2D layers, 64 filters in the third layer
- Dropout layer with a 0.25 dropout rate
- Output: Softmax layer with 2 units (for 2 classes: open and closed eyes)

### ResNet50 Model

This model uses a pretrained ResNet50 architecture with ImageNet weights, followed by fully connected layers for classification:
- Frozen ResNet50 layers
- Dense layer with 128 neurons
- Dropout of 0.5
- Output: Softmax layer with 2 units

The model is trained with categorical cross-entropy loss and Adam optimizer.

## Training

The models are trained for 10 epochs each using the following command:

```python
history = cnn_model.fit(train_generator, validation_data=validation_generator, epochs=10)
history = resnet50_model.fit(train_generator, validation_data=validation_generator, epochs=10)
```

You can adjust the number of epochs based on your hardware capacity and time constraints.

### Metrics

During training, we track the following metrics:
- Categorical Accuracy
- Precision
- Recall

The training data is augmented using `ImageDataGenerator`.

## Evaluation

To evaluate the model, the validation set is used. Performance metrics such as accuracy, precision, recall, and loss are plotted to observe the model's learning behavior.

```python
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'])
plt.show()
```

## Results

- **CNN Model Accuracy:** Approximately 95% validation accuracy
- **ResNet50 Model Accuracy:** Approximately 87% validation accuracy (with fine-tuning)

Both models show strong performance in distinguishing between open and closed eyes.

## How to Use

1. Clone the repository:

   ```bash
   git clone https://github.com/vpratap007/drowsiness-detection.git
   cd drowsiness-detection
   ```

2. Download the dataset and place it in the `input/drowsiness-detection/` folder.

3. Train the models by running:

   ```bash
   python train_cnn.py
   python train_resnet.py
   ```

4. Use the saved models to make predictions:

   ```python
   from keras.models import load_model
   import cv2
   import numpy as np

   model = load_model('./cnn.h5')  # Or use resnet50.h5 for the ResNet model
   image = cv2.imread('path_to_image')
   image = cv2.resize(image, (32, 32))
   image = np.expand_dims(image, axis=0)
   pred = np.argmax(model.predict(image))
   print("The predicted class is", pred)
   ```

## References

1. [ResNet50 Documentation](https://keras.io/api/applications/resnet/)
2. [TensorFlow ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)

---
