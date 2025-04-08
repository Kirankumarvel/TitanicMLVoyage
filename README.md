# 🚢 TitanicMLVoyage

**TitanicMLVoyage** is a complete machine learning pipeline project that explores survival prediction on the famous Titanic dataset. It covers preprocessing, feature engineering, model training using **Random Forest** and **Logistic Regression**, hyperparameter tuning with **GridSearchCV**, and evaluation using classification metrics and visualizations.

> 📊 Whether you're a beginner or brushing up on your ML workflow skills, this project demonstrates how to go from raw data to actionable insights.

---

## 🔧 Features

- 📦 Preprocessing pipelines for both numerical and categorical data
- 🚫 Handles missing data and irrelevant features
- 🔍 Hyperparameter tuning with GridSearchCV
- 🧠 Model comparison: RandomForestClassifier vs LogisticRegression
- 📈 Evaluation using classification reports, confusion matrix, and feature importance
- 🧪 Cross-validation with StratifiedKFold
- 📊 Visualizations using seaborn and matplotlib

---

## 🗂️ Project Structure

```
TitanicMLVoyage/
│
├── data/                  # (Optional) Data files or scripts to load dataset
├── notebook.py        # Main Jupyter notebook with all tasks
├── README.md              # Project documentation
├── requirements.txt       # Dependencies
└── assets/                # Saved plots or images (optional)
```

---

## 📚 Dataset

This project uses the built-in `titanic` dataset from the `seaborn` library, containing passenger information such as:

- Age, sex, and class
- Number of siblings/spouses aboard
- Port of embarkation
- Fare, and more...

The target variable is `survived` (0 = No, 1 = Yes).

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Kirankumarvel/TitanicMLVoyage.git
cd TitanicMLVoyage
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter Notebook
```bash
jupyter notebook
```

---

## 📌 Key Highlights

| Task                         | Description |
|-----------------------------|-------------|
| Data Cleaning                | Dropped irrelevant and sparse features |
| Feature Engineering          | Auto-detect numerical and categorical features |
| Pipeline Construction        | Used `ColumnTransformer` & `Pipeline` from `sklearn` |
| Model Evaluation             | Confusion matrix, classification report, test accuracy |
| Model Comparison             | Random Forest vs Logistic Regression |

---

## 📷 Visuals

Confusion Matrix Example:

<p align="center">
  <img src="assets/confusion_matrix_rf.png" width="400">
</p>

---

## 🧠 Future Ideas

- Use ensemble models like XGBoost or VotingClassifier  
- Add SHAP/Permutation feature importance  
- Build a Streamlit or Flask app for live predictions  
- Improve data imputation and outlier handling

---

## 💬 License

MIT License — feel free to fork, experiment, and contribute.

---

## 🙌 Acknowledgements

- [Scikit-learn](https://scikit-learn.org)
- [Seaborn](https://seaborn.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- Inspired by Titanic Kaggle competition

---

### 🌊 Ready to set sail? Let's navigate the seas of machine learning together in **TitanicMLVoyage**.
