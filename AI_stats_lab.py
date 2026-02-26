"""
Linear & Logistic Regression Lab

Follow the instructions in each function carefully.
DO NOT change function names.
Use random_state=42 everywhere required.
"""

import numpy as np

from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================================================
# QUESTION 1 – Linear Regression Pipeline (Diabetes)
# =========================================================

def diabetes_linear_pipeline():
    
    # STEP 1: Load diabetes dataset.
    
    data = load_diabetes()
    X = data.data
    y = data.target

    # STEP 2: Split into train and test (80-20).
    #         Use random_state=42.

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
#    STEP 3: Standardize features using StandardScaler.
#             IMPORTANT:
#             - Fit scaler only on X_train
#             - Transform both X_train and X_test

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # Fit only on train
    X_test_scaled = scaler.transform(X_test)         # Transform test

            
    # STEP 4: Train LinearRegression model.
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
             
    '''STEP 5: Compute:
            - train_mse
            - test_mse
            - train_r2
            - test_r2'''

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
                    # Metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # STEP 6: Identify indices of top 3 features
    #         with largest absolute coefficients.

    coef_abs = np.abs(model.coef_)
    top_3_feature_indices = np.argsort(coef_abs)[-3:].tolist()

#    RETURN:
#         train_mse,
#         test_mse,
#         train_r2,
#         test_r2,
#         top_3_feature_indices (list length 3)
    
    # -------------------------------------------------------
    # COMMENT:
    # Overfitting occurs if train R² >> test R².
    # If both are similar → model generalizes well.
    #
    # Feature scaling is important because:
    # - Features may have different magnitudes.
    # - Scaling ensures fair coefficient comparison.
    # - Improves numerical stability.
    # -------------------------------------------------------

    return train_mse, test_mse, train_r2, test_r2, top_3_feature_indices



# =========================================================
# QUESTION 2 – Cross-Validation (Linear Regression)
# =========================================================

def diabetes_cross_validation():
    
    # STEP 1: Load diabetes dataset.
    
    data = load_diabetes()
    X = data.data
    y = data.target
    
    # STEP 2: Standardize entire dataset (after splitting is NOT needed for CV,
    #         but use pipeline logic manually).
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    '''STEP 3: Perform 5-fold cross-validation
            using LinearRegression.
            Use scoring='r2'.'''

    model = LinearRegression()
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')



    '''STEP 4: Compute:
            - mean_r2
            - std_r2'''
            
    mean_r2 = np.mean(scores)
    std_r2 = np.std(scores)

    # -------------------------------------------------------
    # COMMENT:
    # Standard deviation represents variability of performance
    # across folds.
    #
    # Cross-validation reduces variance risk by:
    # - Using multiple train/test splits
    # - Giving more reliable estimate of generalization
    # -------------------------------------------------------

#    RETURN:
#         mean_r2,
#         std_r2
    

    return mean_r2, std_r2


# =========================================================
# QUESTION 3 – Logistic Regression Pipeline (Cancer)
# =========================================================

def cancer_logistic_pipeline():
    
    # STEP 1: Load breast cancer dataset.
    
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # STEP 2: Split into train-test (80-20).
        #  Use random_state=42.
            
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # STEP 3: Standardize features.
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # STEP 4: Train LogisticRegression(max_iter=5000).

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    '''STEP 5: Compute:
            - train_accuracy
            - test_accuracy
            - precision
            - recall
            - f1
            - confusion matrix (optional to compute but not return)'''

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    cm = confusion_matrix(y_test, y_test_pred)

    # In comments:
    #     Explain what a False Negative represents medically.

    # -------------------------------------------------------
    # COMMENT:
    # False Negative (FN) in medical context:
    # A patient actually has cancer,
    # but the model predicts "no cancer".
    #
    # This is dangerous because:
    # - Disease remains untreated
    # - Condition may worsen
    # -------------------------------------------------------

    # RETURN:
    #     train_accuracy,
    #     test_accuracy,
    #     precision,
    #     recall,
    #     f1
    

    return train_accuracy, test_accuracy, precision, recall, f1



# =========================================================
# QUESTION 4 – Logistic Regularization Path
# =========================================================

def cancer_logistic_regularization():
    
    # STEP 1: Load breast cancer dataset.

    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # STEP 2: Split into train-test (80-20).

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # STEP 3: Standardize features.

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}


    # STEP 4: For C in [0.01, 0.1, 1, 10, 100]:
            # - Train LogisticRegression(max_iter=5000, C=value)
            # - Compute train accuracy
            # - Compute test accuracy

    for C_value in [0.01, 0.1, 1, 10, 100]:

        model = LogisticRegression(max_iter=5000, C=C_value)
        model.fit(X_train_scaled, y_train)

        train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
        test_acc = accuracy_score(y_test, model.predict(X_test_scaled))

    # STEP 5: Store results in dictionary:
            # {
            #     C_value: (train_accuracy, test_accuracy)
            # }

        results[C_value] = (train_acc, test_acc)


    '''In comments:
        - What happens when C is very small?
        - What happens when C is very large?
        - Which case causes overfitting?'''

    # -------------------------------------------------------
    # COMMENT:
    # Very small C:
    # - Strong regularization
    # - Simpler model
    # - Risk of underfitting
    #
    # Very large C:
    # - Weak regularization
    # - Complex model
    # - Risk of overfitting
    #
    # Overfitting usually happens when C is very large.
    # -------------------------------------------------------

    # RETURN:
    #     results_dictionary
    

    return results


# =========================================================
# QUESTION 5 – Cross-Validation (Logistic Regression)
# =========================================================

def cancer_cross_validation():
    
    # STEP 1: Load breast cancer dataset.

    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # STEP 2: Standardize entire dataset.

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # STEP 3: Perform 5-fold cross-validation
            # using LogisticRegression(C=1, max_iter=5000).
            # Use scoring='accuracy'.

    model = LogisticRegression(C=1, max_iter=5000)

    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')

   

    # STEP 4: Compute:
            # - mean_accuracy
            # - std_accuracy

    mean_accuracy = np.mean(scores)
    std_accuracy = np.std(scores)

    '''In comments:
        Explain why cross-validation is especially
        important in medical diagnosis problems.'''

        
    # -------------------------------------------------------
    # COMMENT:
    # Cross-validation is critical in medical diagnosis because:
    # - Medical decisions affect human lives.
    # - We need reliable and stable performance estimates.
    # - It reduces risk of relying on one lucky split.
    # -------------------------------------------------------

    # RETURN:
    #     mean_accuracy,
    #     std_accuracy


    return mean_accuracy, std_accuracy
