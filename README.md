
# Intel Image Classification using a 3-Layer Convolutional Neural Network (CNN)

This project implements an end-to-end Image Classification pipeline using a 3-layer Convolutional Neural Network (CNN) built with TensorFlow/Keras. The model is trained and evaluated on the **Intel Image Classification Dataset** obtained from Kaggle.

---

## Project Overview & Structure

The Jupyter Notebook is systematically divided into three main parts, fulfilling all academic requirements:

### Part A: Data Understanding and Preprocessing
* **Dataset:** Utilizes the Intel Image Classification dataset comprising natural scene images across 6 distinct classes (*Buildings, Forest, Glacier, Mountain, Sea, Street*).
* **Preprocessing Pipeline:** 
  * Resized all input images to a uniform dimension of `150x150` pixels.
  * Performed pixel normalization by scaling values to `[0, 1]` (`rescale=1.0/255.0`).
  * Applied **Data Augmentation** (rotation, zoom, shear, and horizontal flips) on the training set to improve model generalization and prevent overfitting.
  * Split the dataset into training, validation, and test sets using ImageDataGenerator.

### Part B: CNN Model Development and Training
* **Architecture:** Designed a sequential CNN consisting of:
  * 3 Convolutional layers (`Conv2D`) with increasing filter sizes (32, 64, 128) combined with `ReLU` activation functions.
  * Max Pooling layers (`MaxPooling2D`) for spatial dimension reduction.
  * Flatten layer followed by fully connected Dense layers (`512` units).
  * Dropout regularization (`Dropout(0.5)`) to mitigate overfittin.
  * Output layer with `Softmax` activation for 6-class multi-class classification.
* **Training:** Compiled using the `adam` optimizer and `categorical_crossentropy` loss function, trained over 10 epoch.
* **Visualizations:** Plotted and analyzed training vs. validation accuracy and loss curves.

### Part C: Model Evaluation and Interpretation
* **Performance Metrics:** Evaluated the trained model on the unseen test dataset.
* 
* **Results:** 
  * Achieved an overall test accuracy of **80.53%**.
  * Generated a detailed **Classification Report** (Precision, Recall, F1-score per class) and **Confusion Matrix** to examine class-wise performance.
  * Highlighted strong recognition capabilities in categories like *forest* and *street*.

---

## Requirements & Dependencies
Make sure you have the following libraries installed in your Python/Anaconda environment:
* Python 3.x
* TensorFlow / Keras
* Scikit-Learn
* Matplotlib
* Kagglehub



---
