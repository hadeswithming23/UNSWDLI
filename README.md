
# A Lightweight RF Model for Intrusion Detection using UNSW-NB15

Our work introduces a lightweight Random Forest (RF) model designed for efficient network intrusion detection with minimal computational overhead. The model achieves high accuracy and F1-score while remaining suitable for real-time and resource-constrained environments.

## Workflow Steps

1. Load and Prepare Data

Loads training and testing sets (UNSW_NB15_training-set.csv, UNSW_NB15_testing-set.csv).

Drops unnecessary columns (id, attack_cat).

Encodes labels:

0 → Normal

1 → Attack

2. Encode and Scale Features

Categorical columns encoded using LabelEncoder.

Numerical features scaled with StandardScaler.

3. Handle Class Imbalance

Applies SMOTE to balance normal vs. attack samples.

4. Model Training and Hyperparameter Tuning

Uses GridSearchCV to find optimal hyperparameters.

Trains each model on the full SMOTE-balanced dataset.

5. Evaluation Metrics

Each model is evaluated based on Accuracy, Precision, Recall and F1-score

Confusion matrices are visualized for each model.

6. Model Saving

Best models saved in /models as .pkl files.

Encoders and scaler saved for later reuse.

7. Visualization

Combined bar chart comparing all models on test metrics.


### Quick Start
1. Clone the Repository
git clone https://github.com/hadeswithming23/UNSWDLI.git

cd UNSWDLI

2. Install Dependencies
pip install -r requirements.txt

3. Run the Streamlit App
streamlit run unsw_app.py

Then open your browser at http://localhost:8501 to interact with the model.
