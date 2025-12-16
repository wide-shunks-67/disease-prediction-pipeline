import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_and_preprocess_data(filepath):
    """
    Loads data and creates the preprocessing pipeline.
    """
    # 1. Load Data
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print("❌ Error: 'patient_data.csv' not found.")
        return None, None, None, None, None

    # Define feature groups
    numeric_features = ['Age', 'BloodPressure', 'Cholesterol', 'BMI']
    categorical_features = ['Gender', 'SmokingStatus', 'ExerciseLevel', 'FamilyHistory']

    # --- PART A: Data Preprocessing ---
    
    # Numerical Strategy: Median Imputation + Scaling
    # Justification: Median is robust to outliers common in medical data (like BP spikes).
    # Scaling is required for SVM to calculate distances correctly.
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical Strategy: One-Hot Encoding
    # Justification: Converts categorical text to binary format. 'drop="first"' prevents multicollinearity.
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Split Data
    # Justification: 'stratify=y' ensures the test set maintains the original class distribution (35% disease).
    X = df.drop('Disease', axis=1)
    y = df['Disease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    return X_train, X_test, y_train, y_test, preprocessor

def evaluate_model(name, model, X_test, y_test):
    """
    Predicts and prints evaluation metrics.
    """
    y_pred = model.predict(X_test)
    
    print(f"\n=== {name} Performance ===")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return recall_score(y_test, y_pred)

def main():
    # Load and Split
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data('patient_data.csv')
    
    if X_train is None:
        return

    # Transform Data
    # Fit on Train, Transform on Test to avoid data leakage
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # --- PART B: Model Implementation ---

    # Model 1: Decision Tree
    # Justification: 'balanced' weights handle class imbalance; max_depth=5 prevents overfitting.
    dt_model = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
    dt_model.fit(X_train_processed, y_train)
    
    # Model 2: Support Vector Machine (SVM)
    # Justification: RBF kernel handles non-linear relationships often found in biological data.
    svm_model = SVC(kernel='rbf', C=1.0, class_weight='balanced', random_state=42)
    svm_model.fit(X_train_processed, y_train)

    # Evaluate
    dt_recall = evaluate_model("Decision Tree", dt_model, X_test_processed, y_test)
    svm_recall = evaluate_model("Support Vector Machine (SVM)", svm_model, X_test_processed, y_test)

    # --- PART C: Final Recommendation ---
    print("\n--- 🏥 Recommendation ---")
    print(f"Decision Tree Recall: {dt_recall:.2%}")
    print(f"SVM Recall:           {svm_recall:.2%}")
    
    if svm_recall > dt_recall:
        print("\n✅ RECOMMENDED MODEL: SVM")
        print("Reasoning: In healthcare, missing a positive case (False Negative) is dangerous.")
        print("The SVM model has higher Recall, meaning it catches more disease cases.")
    else:
        print("\n✅ RECOMMENDED MODEL: Decision Tree")
        print("Reasoning: The Decision Tree offers comparable performance with better interpretability.")

if __name__ == "__main__":
    main()
