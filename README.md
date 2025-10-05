# 🧠 A Lightweight Random Forest Model for Intrusion Detection using UNSW-NB15

This project presents a **lightweight Random Forest (RF)** model for **network intrusion detection**, designed to balance **high detection accuracy** with **low computational cost**. The model performs efficiently even in **real-time** and **resource-constrained** environments, making it suitable for modern cybersecurity systems.

---

## ⚙️ Workflow Overview

### **1. Load and Prepare Data**
- Loads datasets:  
  - `UNSW_NB15_training-set.csv`  
  - `UNSW_NB15_testing-set.csv`
- Removes unnecessary columns: `id`, `attack_cat`
- Encodes labels:  
  - `0` → **Normal**  
  - `1` → **Attack**

---

### **2. Encode and Scale Features**
- Categorical features encoded using **LabelEncoder**  
- Numerical features standardized using **StandardScaler**

---

### **3. Handle Class Imbalance**
- Applies **SMOTE (Synthetic Minority Over-sampling Technique)**  
  to balance *Normal* and *Attack* samples.

---

### **4. Model Training and Hyperparameter Tuning**
- Uses **GridSearchCV** for optimal hyperparameter selection  
- Trains each model on the SMOTE-balanced dataset  
- Includes baseline and tuned models for comparison

---

### **5. Evaluation Metrics**
Each model is evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

✅ **Confusion Matrices** are plotted for visual comparison.  
✅ A **combined bar chart** summarizes all test metrics.

---

### **6. Model Saving**
- Trained models saved in the `/models` directory as `.pkl` files  
- Encoders and scalers stored for easy reuse during deployment

---

### **7. Visualization**
- Generates visual comparisons of model performance  
- Includes confusion matrices and bar charts for clarity

---

## 🚀 Quick Start

### **1. Clone the Repository**
```bash
git clone https://github.com/hadeswithming23/UNSWDLI.git
cd UNSWDLI
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Run the Streamlit App**
```bash
streamlit run unsw_app.py
```
