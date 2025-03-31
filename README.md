

# 🫁 Lung Cancer Detector using CNN – 96% Accuracy

## 📊 **Project Overview**
This project implements a Convolutional Neural Network (CNN) model to detect lung cancer with **96% accuracy**. The model uses medical image data to classify between cancerous and non-cancerous cases, providing a robust solution for early detection.

## 🚀 **Features**
- **High Accuracy:** Achieves **96% accuracy**, making it a reliable tool for lung cancer detection.
- **Deep Learning Architecture:** Uses a CNN to efficiently extract features from medical images.
- **Preprocessing and Augmentation:** Includes data preprocessing steps like resizing, normalization, and image augmentation to enhance model performance.
- **Visualization:** Displays sample images, training progress, and model evaluation metrics.

## 🛠️ **Technologies Used**
- Python
- TensorFlow/Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- OpenCV (for image processing)

## 📁 **Dataset**
The model is trained on a lung and colon cancer dataset consisting of labeled medical images. Each image is categorized into cancerous and non-cancerous classes.

## ⚙️ **Model Architecture**
The CNN model consists of:
- **Input Layer:** Accepts medical images in a standardized format.
- **Convolutional Layers:** Extracts relevant features using multiple filters.
- **Pooling Layers:** Reduces dimensionality while retaining important features.
- **Fully Connected Layers:** Makes predictions based on extracted features.
- **Activation Function:** Uses ReLU and softmax activation for classification.

## 📊 **Performance**
- **Accuracy:** 96%
- **Loss:** Displays minimal loss during training.
- **Confusion Matrix & Classification Report:** Evaluates model performance with precision, recall, and F1-score.

## ▶️ **How to Run**
1. Clone the repository:
   ```bash
   git clone <your-github-repo-link>
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook Lung_Colon_Cancer_(CNN)ACC_96_.ipynb
   ```

## 📊 **Model Evaluation**
- **Accuracy Plot:** Displays training and validation accuracy over epochs.
- **Loss Plot:** Shows the decrease in loss, indicating model improvement.
- **Confusion Matrix:** Provides insights into false positives and false negatives.

## 📚 **Future Enhancements**
- Improve model generalization by adding more diverse datasets.
- Implement fine-tuning techniques to boost accuracy.
- Deploy the model using Flask or FastAPI for real-world applications.

## 💡 **Contributions**
Contributions are welcome! Feel free to create pull requests or raise issues.

## 📄 **License**
This project is licensed under the MIT License.

