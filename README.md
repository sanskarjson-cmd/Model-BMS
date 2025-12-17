# NSCLC Survival Analysis: Gradient Boosting vs Cox Model

## Overview

This project implements a **Gradient Boosting Survival Model** as an alternative to traditional Cox Proportional Hazards regression for predicting overall survival in Non-Small Cell Lung Cancer (NSCLC) patients.

## Problem Statement

Predicting patient survival time is critical for:
- Treatment planning and clinical decision-making
- Patient risk stratification
- Clinical trial design
- Resource allocation in healthcare

Traditional Cox models make strong assumptions (proportional hazards) that are often violated in real-world data. This project explores machine learning alternatives.

## Dataset

**Source**: NSCLC patient cohort (N=1,289 patients)
- **Events (deaths)**: 708 (54.9%)
- **Censored**: 581 (45.1%)
- **Follow-up period**: Up to January 1, 2025

**Features**:
- Demographics: Age, sex, race
- Clinical: ECOG performance status, cancer stage, metastatic status, histology
- Treatment: Surgery status
- Risk factors: Smoking history

## Models Compared

### 1. Cox Proportional Hazards (Baseline)
- **Type**: Parametric survival regression
- **C-index**: 0.7200
- **Limitations**: 
  - Proportional hazards assumption violated for 4 variables
  - Linear relationships only
  - No automatic feature interactions

### 2. Gradient Boosting Regressor (Proposed Alternative)
- **Type**: Tree-based machine learning
- **C-index**: 0.7247 (+0.66% improvement)
- **Advantages**:
  - ✅ No proportional hazards assumption needed
  - ✅ Captures non-linear relationships automatically
  - ✅ Learns feature interactions (age × stage, surgery × metastatic)
  - ✅ Robust to outliers (Huber loss)
  - ✅ Weighted learning (events weighted 1.0, censored 0.5)

## Key Results

### Model Performance
```
Model                    C-index    Type
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cox Proportional Hazards  0.7200    Parametric regression
Gradient Boosting          0.7247    Tree-based ML (sklearn)
```

### Risk Stratification
The Gradient Boosting model successfully stratifies patients into three risk groups:

| Risk Group | Median Survival | Death Rate |
|------------|-----------------|------------|
| **High Risk** | 8.6 months | 72.1% |
| **Medium Risk** | 17.1 months | 57.0% |
| **Low Risk** | 41.3 months | 36.0% |

### Feature Importance
Top predictors of survival:
1. **Surgery** (29.8%) - Most important factor
2. **Age** (16.9%) - Strong predictor
3. **Stage IA** (12.2%) - Early detection matters
4. **Stage IV** (10.5%) - Advanced disease indicator
5. **Race (White)** (5.0%)

## Technical Implementation

### Dependencies
```python
pandas==2.x
numpy==1.x
scikit-learn==1.3+
lifelines==0.27+
matplotlib==3.x
```

### Model Architecture
```python
GradientBoostingRegressor(
    n_estimators=100,          # 100 decision trees
    learning_rate=0.05,        # Slow learning for generalization
    max_depth=4,               # Shallow trees (prevent overfitting)
    min_samples_split=20,      # Minimum samples to split
    min_samples_leaf=10,       # Minimum samples in leaf
    subsample=0.8,             # 80% data per tree
    loss='huber',              # Robust to outliers
    random_state=42
)
```

### Handling Censored Data
- **Sample Weights**: Events (deaths) = 1.0, Censored = 0.5
- **Approach**: Treats censored times as lower bounds for survival
- **Evaluation**: Concordance index (C-index) for ranking predictions

## Usage

### 1. Data Preprocessing
```python
# Load and merge datasets
cohort = prepare_nsclc_cohort(df_nsclc, df_mortality, df_demographics, df_ecog)

# Calculate survival time
cohort["os_time_days"] = (cohort["end_date"] - cohort["start_date"]).dt.days

# Encode categorical variables
model_df = pd.get_dummies(cohort, columns=categorical_cols, drop_first=True)
```

### 2. Train Model
```python
# Prepare features and target
X_train = train_df.drop(columns=["os_time_days", "event"])
y_train = train_df["os_time_days"]

# Create sample weights
train_weights = np.where(train_df["event"] == 1, 1.0, 0.5)

# Train Gradient Boosting
gb_model = GradientBoostingRegressor(...)
gb_model.fit(X_train, y_train, sample_weight=train_weights)
```

### 3. Make Predictions
```python
# Predict survival time
predicted_survival = gb_model.predict(X_test)

# Calculate C-index
c_index = concordance_index(y_test, predicted_survival, event_indicator)
```

## Model Evaluation

### Concordance Index (C-index)
- **Interpretation**: Probability that model correctly ranks patient pairs by survival time
- **Range**: 0.5 (random) to 1.0 (perfect)
- **Result**: 0.7247 indicates good discriminative ability

### Prediction Accuracy (for observed deaths)
- **MAE**: 499 days (16.6 months)
- **RMSE**: 703 days (23.4 months)

## Limitations

1. **Not a Native Survival Model**: Unlike Cox, GB doesn't have a built-in survival objective
2. **Censoring Approximation**: Treats censored observations as lower bounds (not optimal)
3. **Less Interpretable**: Tree-based models harder to explain than Cox coefficients
4. **No Confidence Intervals**: Predictions lack uncertainty quantification

## Improvements Over Cox

| Aspect | Cox Model | Gradient Boosting |
|--------|-----------|-------------------|
| **Proportional Hazards** | Required (4 violations found) | Not needed ✅ |
| **Non-linear Effects** | Manual feature engineering | Automatic ✅ |
| **Feature Interactions** | Manual specification | Automatic ✅ |
| **Outlier Robustness** | Sensitive | Robust (Huber loss) ✅ |
| **Interpretability** | High (coefficients) | Lower (tree splits) |
| **Prediction Type** | Risk score | Survival time ✅ |

## Future Work

### 1. Enhanced Models
- **XGBoost Survival**: Native survival objective (`survival:cox`)
- **Random Survival Forest**: Ensemble of survival trees
- **Deep Learning**: DeepSurv neural network
- **Ensemble**: Combine Cox + GB for robust predictions

### 2. Additional Features
- Biomarkers: PD-L1, EGFR, ALK status
- Treatment history: Chemotherapy, immunotherapy, radiation
- Comorbidities: Heart disease, diabetes, COPD
- Lab values: Hemoglobin, albumin, LDH

### 3. Model Improvements
- Hyperparameter tuning (GridSearchCV)
- Cross-validation (5-fold CV)
- Calibration analysis (predicted vs actual survival curves)
- External validation cohort

### 4. Clinical Deployment
- Web application for risk prediction
- Integration with EHR systems
- Real-time survival updates
- Explainability (SHAP values)

## References

1. **Cox Proportional Hazards**: Cox, D. R. (1972). Regression models and life-tables. *Journal of the Royal Statistical Society*
2. **Gradient Boosting**: Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*
3. **Survival Analysis**: Klein, J. P., & Moeschberger, M. L. (2003). *Survival Analysis: Techniques for Censored and Truncated Data*
4. **C-index**: Harrell, F. E., et al. (1982). Evaluating the yield of medical tests. *JAMA*

## License

MIT License

## Contact

For questions or collaboration:
- **GitHub**: [Your GitHub Username]
- **Email**: [Your Email]

## Acknowledgments

- Original Cox model implementation by [Friend's Name]
- NSCLC dataset provided by [Institution/Source]
- Built with scikit-learn and lifelines libraries

---

**Note**: This model is for research purposes only and should not be used for clinical decision-making without proper validation and regulatory approval.