# Disease Prediction System 🏥

## Project Overview
This project is a healthcare machine learning pipeline designed to predict the likelihood of disease in patients based on medical history and demographic data. 

**Key Objective:** To assist healthcare professionals in early diagnosis by building a model that prioritizes **Recall (Sensitivity)** to minimize the risk of missing positive disease cases (False Negatives).

## 🚀 Key Features
* **End-to-End Pipeline:** From data preprocessing (handling missing values, encoding) to model evaluation.
* **Imbalance Handling:** Utilized Stratified Splitting and `class_weight='balanced'` to manage the 35% vs 65% class imbalance.
* **Comparative Analysis:** Evaluated **Decision Tree** vs. **Support Vector Machine (SVM)**.
* **Business-Centric Evaluation:** Prioritized Recall over Accuracy to align with healthcare safety standards.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Scikit-learn, Pandas, NumPy
* **Models:** Decision Tree Classifier, C-Support Vector Classification (SVC)

## 📊 Results
| Model | Accuracy | Recall (Sensitivity) | Precision | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **SVM (Recommended)** | **81.3%** | **90.5%** | 67.4% | 0.77 |
| Decision Tree | 71.7% | 74.3% | 57.4% | 0.65 |

**Conclusion:** The **SVM model** was selected for deployment because it achieved a **90.5% Recall rate**, significantly reducing the chance of undiagnosed cases compared to the Decision Tree.

## 💻 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/disease-prediction-system.git](https://github.com/YOUR_USERNAME/disease-prediction-system.git)
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the pipeline:
   ```bash
   python MAIN.py
