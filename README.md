Pokémon Image Classification using CNN 🧠🐱‍👤
This project implements a Convolutional Neural Network (CNN) to classify Pokémon images based on their primary type (such as Fire, Water, Grass, etc.).
The model is trained using labeled Pokémon images and predicts the Pokémon type from a given input image.

Project Overview
Problem Type: Image Classification
Model Used: Convolutional Neural Network (CNN)
Framework: TensorFlow / Keras
Input: Pokémon image (PNG format)
Output: Predicted Pokémon Type

Dataset
pokemon.csv
Contains Pokémon names and their primary types (Type1)
images/ folder
Contains Pokémon images named using Pokémon names in lowercase
Example: aerodactyl.png

Libraries Used
Python
TensorFlow / Keras
NumPy
Pandas
scikit-learn
Matplotlib

Data Preprocessing
Images resized to 128 × 128
Pixel values normalized to range [0, 1]
Labels encoded using LabelEncoder
Converted labels to one-hot encoding
Dataset split into 80% training and 20% testing

CNN Model Architecture
Input Image (128x128x3)
→ Conv2D (32 filters, 3×3, ReLU)
→ MaxPooling2D
→ Conv2D (64 filters, 3×3, ReLU)
→ MaxPooling2D
→ Flatten
→ Dense (128 units, ReLU)
→ Dense (Softmax – Pokémon Type)
Model Compilation
Optimizer: Adam
Loss Function: Categorical Crossentropy
Evaluation Metric: Accuracy
Model Training
Epochs: 10
Batch Size: 32
Validation: Test dataset
Prediction Example
The trained model predicts the Pokémon type from a single image and visualizes the result.

The output displays:
Pokémon image
Predicted Pokémon type

Key Learning Outcomes
Understanding CNN architecture
Image preprocessing for deep learning
Multi-class classification using Softmax
Model training and evaluation
Version control using Git and GitHub
