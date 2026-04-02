# 🎓 Student Performance Predictor

A machine learning project that predicts whether a student will **pass or fail** based on personal, social, and academic factors. Built as a capstone for the *Fundamentals of AI and ML* course.

---

## 📌 Problem Statement

In many educational institutions, students who are at risk of failing are identified only after it is too late to intervene. This project builds an **early-warning classifier** using supervised learning that can flag at-risk students based on observable characteristics, enabling teachers and counsellors to provide timely support.

---

## 🧠 ML Concepts Used

| Concept | Application |
|---|---|
| Supervised Learning | Binary classification (Pass / Fail) |
| Decision Tree Classifier | Main model — interpretable and visual |
| Train/Test Split | 80/20 stratified split |
| Model Evaluation | Accuracy, Precision, Recall, F1-Score, Confusion Matrix |
| Feature Importance | Identifying the strongest predictors of student outcomes |

---

## 📂 Project Structure

```
student-performance-predictor/
│
├── student_performance_predictor.py   # Main ML script
├── student-mat.csv                    # Dataset
├── results.png                        # Output: confusion matrix + feature importance chart
├── README.md                          # This file
└── report/
    └── Project_Report.pdf             # Full project report
```

---

## 📊 Dataset

**Source:** [UCI Machine Learning Repository — Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)

- **File needed:** `student-mat.csv` (Mathematics course data)
- **Students:** 395
- **Features:** 33 attributes covering demographics, family background, study habits, and grades
- **Target:** Final grade `G3` → converted to `Pass (≥10)` or `Fail (<10)`

> **Download the dataset manually** from the UCI link above and place `student-mat.csv` in the root project folder before running the script.

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/student-performance-predictor.git
cd student-performance-predictor
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

> Python 3.8 or higher is recommended.

### 3. Add the dataset
Download `student-mat.csv` from the UCI link above and place it in the project root folder.

### 4. Run the script
```bash
python student_performance_predictor.py
```

---

## 📈 Sample Output

After running, you will see:

- **Accuracy score** and full **classification report** in the terminal
- A saved image `results.png` containing:
  - Confusion Matrix (actual vs predicted outcomes)
  - Feature Importance bar chart (which factors matter most)
- Human-readable **decision rules** extracted from the tree
- A **demo prediction** for a sample student profile

Example terminal output:
```
Accuracy : 91.14%

Classification Report:
              precision    recall  f1-score   support
        Fail       0.79      0.73      0.76        15
        Pass       0.93      0.95      0.94        64

    accuracy                           0.91        79
```

---

## 🔑 Key Findings

- **G2 (second-period grade)** is the strongest predictor of final outcome — prior performance is highly predictive.
- **Number of past failures** is the second most important feature.
- Students who aspire to **higher education** (`higher = yes`) significantly outperform those who do not.
- Lifestyle factors like `goout` and `Walc` (weekend alcohol consumption) have a measurable negative impact.

---

## 🛠️ Dependencies

| Package | Version |
|---|---|
| Python | ≥ 3.8 |
| pandas | ≥ 1.3 |
| numpy | ≥ 1.21 |
| scikit-learn | ≥ 1.0 |
| matplotlib | ≥ 3.4 |
| seaborn | ≥ 0.11 |

Install all at once:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## 🔮 Future Improvements

- Add a web interface using Streamlit so teachers can input student details and get a prediction
- Try Random Forest or Gradient Boosting for better accuracy
- Incorporate the Portuguese language course dataset (`student-por.csv`) for broader generalisation
- Use SHAP values for more detailed model explainability

---

## 📄 License

This project is submitted as part of an academic course evaluation. Dataset credit: P. Cortez and A. Silva, University of Minho, Portugal.

---

## 👤 Author

**Shivam Rathore**  
Course: Fundamentals of AI and ML  
