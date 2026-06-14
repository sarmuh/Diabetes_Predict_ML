# Diabetes Prediction ML

This project aims to build Machine Learning models to predict whether a patient has diabetes based on their medical indicators (blood pressure, insulin, body mass index, etc.). The project was developed in the Google Colab environment and pushed directly to GitHub.

## 📊 Dataset
The project uses the "Diabetes" dataset. The dataset includes the following columns:
* `Pregnancies`: Number of pregnancies
* `Glucose`: Glucose level
* `BloodPressure`: Blood pressure
* `SkinThickness`: Skin thickness
* `Insulin`: Insulin level
* `BMI`: Body Mass Index
* `DiabetesPedigreeFunction`: Diabetes pedigree function (genetic predisposition)
* `Age`: Patient's age
* `Outcome`: Result (0 - healthy, 1 - diabetic)

## 🛠 Data Preprocessing
Before training the models, the following operations were performed on the data:
1. **Handling invalid values:** Incorrectly recorded `0` (zero) values in the `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` columns were replaced with `NaN`.
2. **Imputing missing values:** The `KNNImputer(n_neighbors=7)` algorithm was used to fill in the missing (`NaN`) values.
3. **Class balancing:** To eliminate the large disparity between the number of healthy and diabetic patients in the dataset, the **SMOTE** (Synthetic Minority Over-sampling Technique) method was applied. This artificially augmented the data for diabetic patients to ensure balance.

## 🤖 Models and Hyperparameter Tuning
Two powerful algorithms were primarily used in this project, and their hyperparameters were optimized:

1. **Random Forest Classifier:**
   * Searched for the best parameters using `HalvingGridSearchCV`.
   * Optimal parameters found: `max_samples=0.8`, `n_estimators=500`.
2. **XGBoost Classifier (with GPU):**
   * Utilized an NVIDIA GPU (`device='cuda'`) and the **cuDF** library to accelerate training.
   * Searched for optimal parameters via `HalvingRandomSearchCV`.
   * Best parameters: `subsample=1.0`, `n_estimators=800`, `max_depth=3`, `learning_rate=0.01`, and `min_child_weight=7`.

## 📈 Results and Evaluation
Model performance on the Test data was evaluated using the following metrics:
* **Accuracy:** ~77.2% (for XGBoost)
* **Classification Report:** Precision, Recall, and F1-score metrics were analyzed.
* **Jaccard Score:** ~0.57
* **Confusion Matrix:** True predictions (True Positives, True Negatives) and errors (False Positives, False Negatives) were visualized beautifully using a Heatmap.

## 💻 Technologies and Libraries Used
The project is written entirely in Python.
* **Data Analysis:** `pandas`, `numpy`, `cuDF`
* **Visualization:** `matplotlib.pyplot`, `seaborn`
* **Machine Learning:** `scikit-learn`, `xgboost`, `imblearn` (for SMOTE)

## 🚀 Getting Started
The project is designed with Google Colab in mind. To test the code:
1. Clone the repository to your local machine.
2. Open the `Diabetes_Predict_ML.ipynb` file (or open it directly in Google Colab via the GitHub interface).
3. To speed up XGBoost, go to the Colab menu and select **Runtime -> Change runtime type -> Hardware accelerator: T4 GPU** (or another GPU).
4. Execute all code cells from top to bottom (`Run all`).
