# 🩺 Breast Cancer Detection using Artificial Neural Network (ANN)

A machine learning & deep learning project that predicts whether a breast tumor is **Malignant (Cancerous)** or **Benign (Non-Cancerous)** using clinical features from the Breast Cancer Wisconsin dataset.

---

## 📌 Project Overview

Breast cancer is one of the most common cancers worldwide. Early detection significantly improves survival rates.  
This project uses an **Artificial Neural Network (ANN)** to classify tumors based on diagnostic measurements.

The model is trained on the **Breast Cancer Wisconsin dataset** and achieves high accuracy with proper preprocessing, feature scaling, and evaluation.

---

## 🎯 Objective

- Build a **binary classification model**
- Predict:
  - `0` → Malignant (Cancerous)
  - `1` → Benign (Non-Cancerous)
- Ensure the model is **deployment-ready**

---

## 📊 Dataset Information

- Source: `sklearn.datasets.load_breast_cancer`
- Total Samples: **569**
- Features: **30 numerical features**
- Target Classes:
  - Malignant
  - Benign

Examples of features:
- Mean radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness

---

## 🛠️ Tech Stack & Libraries

- **Python**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **TensorFlow / Keras**
- **Joblib**

---

## 🔄 Project Workflow

1. **Data Loading**
   - Load dataset from Scikit-learn
   - Convert to Pandas DataFrame

2. **Exploratory Data Analysis (EDA)**
   - Statistical summary
   - Boxplots to detect outliers

3. **Outlier Treatment**
   - Quantile-based capping (5% – 95%)
   - IQR-based clipping

4. **Feature Engineering**
   - Feature-target separation
   - Train-test split (80/20)

5. **Feature Scaling**
   - StandardScaler for normalization

6. **Model Building**
   - Artificial Neural Network (ANN)
   - ReLU activation in hidden layers
   - Sigmoid activation in output layer

7. **Model Training**
   - Binary Crossentropy loss
   - Adam optimizer
   - Validation split for performance monitoring

8. **Evaluation**
   - Accuracy score
   - Confusion matrix
   - Classification report
   - Prediction confidence

9. **Model Saving**
   - Trained model saved as `model.h5`
   - Scaler saved as `scaler.pkl`

---

## 🧠 Model Architecture

Input Layer → 30 features
Hidden Layer → 64 neurons (ReLU)
Hidden Layer → 32 neurons (ReLU)
Output Layer → 1 neuron (Sigmoid)

yaml
Copy code

---

## 📈 Model Performance

- Test Accuracy: **~97–99%**
- High precision and recall for both classes
- Balanced classification with minimal false negatives

---

## 📊 Evaluation Metrics Used

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## 💾 Saved Files

| File Name     | Description                     |
|----------------|---------------------------------|
| `model.h5`     | Trained ANN model               |
| `scaler.pkl`   | StandardScaler object           |

---

## 🚀 Future Improvements

- Add ROC-AUC curve
- Threshold tuning
- Deploy using **Streamlit / FastAPI**
- Add real-time patient input UI
- Explainability using SHAP / LIME

---

## 🧪 How to Run the Project

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow joblib
bash
Copy code
python app.py
👨‍💻 Author
Shehzad Khan
Aspiring Data Scientist | Machine Learning & AI Enthusiast

📜 License
This project is for educational and research purposes only.

yaml
Copy code
