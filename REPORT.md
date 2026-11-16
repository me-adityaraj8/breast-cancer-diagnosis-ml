# Project Report
## Breast Cancer Tumour Diagnosis using Supervised and Unsupervised Machine Learning

**Course:** Machine Learning Lab, Small Project
**Team:** K Aditya Raj Kumar Dora (240101230) · Atyadik Kumar Anand (241010251) · Naman Singh (240101219)

---

### 1. Introduction
Breast cancer is among the most common cancers worldwide, and early, accurate diagnosis greatly improves survival.
A Fine Needle Aspirate (FNA) of a breast lump is imaged, and the cell nuclei are measured. This project uses those
measurements to (1) **predict** whether a tumour is malignant or benign with **supervised learning**, and (2) check
whether the data **naturally separates** into those two groups without any labels, using **unsupervised learning**.

### 2. Objectives
1. Explore and prepare the dataset (quality checks, EDA, stratified train/test split).
2. **Program 1:** build, tune and evaluate supervised classifiers, giving priority to **not missing cancers** (high recall).
3. **Program 2:** apply PCA and four clustering algorithms, choose the number of clusters without labels, and measure how well the clusters match the true diagnosis.
4. Compare the two approaches and interpret the results in medical terms.

### 3. Dataset
**Breast Cancer Wisconsin (Diagnostic)**, UCI Machine Learning Repository (Wolberg, Street & Mangasarian, 1995).

| Property | Value |
|---|---|
| Samples | 569 patients |
| Classes | Benign 357 (62.7%), Malignant 212 (37.3%) |
| Features | 30 numeric = 10 nucleus measurements × {mean, standard error, worst} |
| Measurements | radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension |
| Missing / duplicate values | none |

The label was re-encoded so that **1 = Malignant** (the disease is the positive class).

### 4. Exploratory Data Analysis (K Aditya Raj Kumar Dora)
* **Data quality:** 0 missing, 0 duplicate, 0 negative values, so no imputation was needed.
* **Scale:** features range from about 0.05 (smoothness) to about 2500 (area), so standardisation is required for distance-based methods.
* **Class balance:** 63/37, mildly imbalanced, so we use stratified splits and report recall, F1 and ROC-AUC alongside accuracy.
* **Multicollinearity:** 21 feature pairs have |r| > 0.9. Radius, perimeter and area are almost identical (r ≈ 0.99), which motivates regularisation and PCA.
* **Strongest single predictors of malignancy:** worst concave points (r = 0.79), worst perimeter (0.78), mean concave points (0.78), worst radius (0.78).
* **Outliers:** 171 rows contain at least one IQR outlier, and 67% of them are malignant. They are real large tumours, so they were **kept**.
* **Split:** 80/20 stratified (455 train / 114 test), `random_state = 42`. The test set was used only once, for the final evaluation.

Figures: `figures/01_*.png`

### 5. Program 1: Supervised Learning (Atyadik Kumar Anand)

**5.1 Method.** Every model is a `Pipeline(StandardScaler → classifier)`, so scaling is learned only on training folds and nothing leaks.
Seven algorithms were compared with 5-fold stratified cross-validation:

| Model | CV Accuracy | CV Recall | CV F1 | CV ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.974 | 0.953 | 0.964 | 0.996 |
| SVM (RBF) | 0.971 | 0.947 | 0.961 | 0.995 |
| Gradient Boosting | 0.967 | 0.947 | 0.955 | 0.991 |
| Random Forest | 0.965 | 0.941 | 0.952 | 0.989 |
| KNN | 0.965 | 0.918 | 0.951 | 0.987 |
| Naive Bayes | 0.938 | 0.912 | 0.917 | 0.987 |
| Decision Tree | 0.923 | 0.894 | 0.896 | 0.917 |

**5.2 Hyperparameter tuning** (`GridSearchCV`, scoring = F1):

| Model | Baseline CV F1 | Tuned CV F1 | Best parameters |
|---|---|---|---|
| **SVM (RBF)** | 0.961 | **0.976** | C = 3, gamma = 0.01, class_weight = balanced |
| Logistic Regression | 0.964 | 0.967 | C = 0.3, L2, class_weight = balanced |
| Random Forest | 0.952 | 0.956 | 200 trees, max_depth = 10, max_features = log2 |

The SVM had the best tuned CV F1 and was chosen as the **final model before the test set was used**.

**5.3 Threshold tuning.** Missing a cancer (false negative) is far worse than a false alarm. Using out-of-fold
probabilities on the training set, we chose the highest threshold that still reaches ≥ 98% recall: **0.34** instead of 0.50.

**5.4 Test-set results (114 unseen patients)**

| Final SVM | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | FN | FP |
|---|---|---|---|---|---|---|---|---|
| threshold 0.50 | 0.982 | 1.000 | 0.952 | 1.000 | 0.976 | 0.997 | 2 | 0 |
| **threshold 0.34** | **0.982** | **0.976** | **0.976** | **0.986** | **0.976** | **0.997** | **1** | **1** |

Lowering the threshold traded one false alarm for one fewer missed cancer, with no change in accuracy.

**5.5 Interpretation.** Permutation importance, Logistic Regression coefficients and Random Forest importance all
rank the **"worst" features** highest (worst texture, radius, concave points, concavity, perimeter, area): enlarged,
irregular nuclei, which agrees with clinical knowledge. The learning curve shows a train–validation F1 gap of only
0.001, so the model neither overfits nor underfits.

Figures: `figures/02_*.png`

### 6. Program 2: Unsupervised Learning (Naman Singh)

**6.1 PCA.** On standardised data, PC1 explains 44.3% and PC2 19.0% of the variance; **7 components keep 90%**.
PC1 loads on concave points, concavity, compactness and perimeter, an overall "size and irregularity" axis.
Clustering was done on the 7-component representation.

**6.2 Choosing k without labels.** The silhouette score for K-Means was highest at **k = 2** (0.371) and decreased after that.
The Ward dendrogram shows its largest merge distance at the final merge, which also supports 2 clusters.

**6.3 Results** (labels revealed only for evaluation):

| Method | Silhouette ↑ | Davies–Bouldin ↓ | Calinski–Harabasz ↑ | ARI ↑ | NMI ↑ | Purity ↑ |
|---|---|---|---|---|---|---|
| **K-Means** | **0.371** | **1.216** | **308.4** | **0.671** | **0.555** | **0.910** |
| Agglomerative (Ward) | 0.347 | 1.290 | 282.8 | 0.666 | 0.543 | 0.909 |
| GMM (2, full cov.) | 0.310 | 1.398 | 241.3 | 0.485 | 0.375 | 0.849 |
| DBSCAN | — (1 cluster) | — | — | 0.049 | 0.038 | 0.610 |

* **K-Means** agrees with the true diagnosis for **91.0%** of patients (518 of 569).
* **Cluster profile:** the malignant-like cluster has 1.4× the radius, 2.0× the area, 4.1× the concavity and 3.6× the concave points of the benign-like cluster, which makes medical sense.
* **GMM** gives soft probabilities; 52 patients have < 80% confidence, and these are the borderline cases.
* **DBSCAN** finds no low-density gap between the classes, so it returns one dense cluster plus 36 noise points. 72% of those noise points are malignant, so it works as an **anomaly detector** rather than a partitioner.

Figures: `figures/03_*.png`

### 7. Comparison: Supervised vs Unsupervised
| Approach | Accuracy | Malignant recall | Malignant precision |
|---|---|---|---|
| K-Means, no labels | 91.0% | 82.5% | 92.6% |
| Ward, no labels | 90.9% | 88.7% | 87.0% |
| **SVM, supervised (test set)** | **98.2%** | **97.6%** | **97.6%** |

Unsupervised learning recovers most of the structure with **zero labels**, which is useful when labelled data is
expensive (each label needs a pathologist). Supervised learning is clearly better at the **boundary cases** where the
two groups overlap, which are exactly the patients where a correct decision matters most.

### 8. Conclusion
* A tuned RBF-SVM with a recall-oriented threshold diagnoses unseen patients with **98.2% accuracy, 97.6% recall and ROC-AUC 0.997**, missing 1 of 42 cancers.
* Without any labels, K-Means on PCA features recovers the diagnosis for **91%** of patients, which shows the measurements carry a strong natural signal.
* Good methodology mattered as much as model choice: leakage-free pipelines, stratified CV, a held-out test set used once, and a medically appropriate metric (recall).

### 9. Limitations and future work
* Small, single-centre dataset (569 patients). External validation on other hospitals' data would be needed before any real use.
* Only tabular features. A CNN on the raw FNA images could learn its own features.
* Cost-sensitive learning or calibrated probabilities could make the recall/precision trade-off explicit.
* Semi-supervised learning (a few labels plus clustering) could combine the strengths of both programs.

### 10. References
1. W. H. Wolberg, W. N. Street, O. L. Mangasarian. *Breast Cancer Wisconsin (Diagnostic)*. UCI Machine Learning Repository, 1995.
2. W. N. Street, W. H. Wolberg, O. L. Mangasarian. "Nuclear feature extraction for breast tumor diagnosis." *IS&T/SPIE Electronic Imaging*, 1993.
3. F. Pedregosa et al. "Scikit-learn: Machine Learning in Python." *JMLR* 12, 2011.
