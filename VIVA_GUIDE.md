# Viva Guide: Who Explains What

The brief says each member must explain **their own part**. Below: what to show, a ~60-second script, and likely
questions with short answers. Rehearse by running your notebook top to bottom and explaining each plot aloud.

---

## K Aditya Raj Kumar Dora: Dataset, EDA and Train/Test Split (`01_data_and_eda.ipynb`)

**60-second script**
> "We used the Breast Cancer Wisconsin dataset: 569 patients and 30 features measured on cell nuclei from a biopsy image.
> There are 10 measurements, each recorded as mean, standard error and worst. The data is clean, with no missing or duplicate values.
> It's 63% benign and 37% malignant, so we stratified the split and don't rely on accuracy alone. The features have very
> different scales, from 0.05 to 2500, so every model standardises first. Radius, perimeter and area are 99% correlated,
> which is why we use regularisation and PCA later. Concave points and size are the strongest indicators of cancer.
> We kept the outliers because two-thirds of them are malignant tumours. We made one 80/20 stratified split that both programs use."

**Likely questions**
| Question | Answer |
|---|---|
| Why stratify? | So train and test keep the same 63/37 class ratio; otherwise one set could get too few malignant cases. |
| Why not remove outliers? | They are genuine large tumours (67% malignant), not errors. Removing them would delete the signal we need. |
| What is the "worst" feature? | The mean of the 3 largest values of that measurement in the image, i.e. the most abnormal nuclei. |
| Why scaling? | KNN, SVM, K-Means use distances; without scaling, `area` (~1000) would dominate `smoothness` (~0.1). |
| What is multicollinearity and why care? | Features that are strongly correlated (radius vs area r = 0.99). It makes linear-model coefficients unstable, so we use regularisation and PCA. |
| Why keep the test set aside? | To get an honest estimate on unseen data. Using it for tuning would be **data leakage**. |
| Why is the positive class Malignant? | Recall and precision should measure how well we detect **cancer**. |

---

## Atyadik Kumar Anand: Program 1, Supervised Learning (`02_supervised_learning.ipynb`)

**60-second script**
> "We compared 7 classifiers with 5-fold stratified cross-validation, each in a pipeline with StandardScaler to avoid leakage.
> Logistic Regression and SVM were best, and the single Decision Tree was worst because it overfits. We tuned the top 3 with GridSearchCV
> on F1. The RBF-SVM with C = 3 and gamma = 0.01 had the best CV F1 of 0.976, so we chose it before touching the test set.
> Missing a cancer is the costliest error, so we lowered the decision threshold from 0.5 to 0.34, chosen on training folds only,
> to reach 98% recall. On 114 unseen patients: 98.2% accuracy, 97.6% recall, ROC-AUC 0.997, and only 1 of 42 cancers missed.
> The learning curve gap is 0.001, so there's no overfitting. The important features are the 'worst' size and shape features."

**Likely questions**
| Question | Answer |
|---|---|
| Why is recall the key metric? | A false negative (missed cancer) delays treatment. A false positive only means an extra test. |
| Precision vs recall? | Precision = of those predicted malignant, how many are. Recall = of real malignant, how many we caught. |
| What is ROC-AUC? | The probability that the model ranks a random malignant case above a random benign one. 0.997 ≈ near-perfect ranking. |
| What does SVM do? | Finds the boundary with the **maximum margin** between classes. The RBF kernel maps data to a higher-dimensional space so the boundary can curve. |
| C and gamma? | C: penalty for misclassification (high C → fewer training errors, risk of overfitting). gamma: reach of each point (high gamma → wiggly boundary). Small gamma = 0.01 gave a smooth boundary. |
| `class_weight='balanced'`? | Weights errors on the minority (malignant) class more, in proportion to class frequency. |
| Why a Pipeline? | The scaler is fitted only on training folds inside CV, so there is no leakage from validation data. |
| Why cross-validation, not one split? | 5 different validation sets give a more reliable estimate and a standard deviation. |
| How was the threshold chosen without cheating? | `cross_val_predict` gives out-of-fold probabilities on the **training** set; we picked the highest threshold with ≥ 98% recall there, then applied it once to the test set. |
| Why did the Decision Tree do worst? | A single deep tree has high variance (overfits). Random Forest averages many trees (bagging) to reduce it. |
| Bagging vs boosting? | Bagging (RF) trains trees in parallel on bootstrap samples, which reduces variance. Boosting (GB) trains trees one after another, each fixing the previous errors, which reduces bias. |
| Why did Naive Bayes underperform? | It assumes features are independent, which is badly broken here (r ≈ 0.99 between features). |
| Why are permutation importances small? | The features are highly correlated: shuffling one leaves its "twins" (e.g. radius ↔ perimeter ↔ area) to carry the same information. |
| Overfitting check? | Learning curve: train and validation F1 end 0.001 apart and both high, so good generalisation. |

---

## Naman Singh: Program 2, Unsupervised Learning (`03_unsupervised_learning.ipynb`)

**60-second script**
> "We hid the labels completely. After scaling, PCA showed 7 of 30 components keep 90% of the variance, with PC1 alone at 44%,
> which is a size and irregularity axis. We clustered on those 7 components. The silhouette score picked k = 2 without any labels.
> K-Means, Ward hierarchical clustering and a Gaussian Mixture all found two groups. When we revealed the labels, K-Means matched
> the true diagnosis for 91% of patients, with ARI 0.67. The malignant-like cluster has twice the area and four times the
> concavity. DBSCAN failed to split the classes because there's no density gap, but its noise points were 72% malignant,
> so it works as an outlier detector. Supervised learning is still better at 98%, mainly on borderline cases."

**Likely questions**
| Question | Answer |
|---|---|
| What is PCA? | Projects data onto orthogonal directions of maximum variance (eigenvectors of the covariance matrix). It removes redundancy and allows 2-D plots. |
| Why cluster on PCA-reduced data? | Removes correlated or noisy dimensions; GMM and DBSCAN struggle in high dimensions (curse of dimensionality). |
| K-Means algorithm? | Pick k centroids → assign each point to the nearest → recompute centroids as the means → repeat until stable. It minimises within-cluster sum of squares (inertia). |
| Why `n_init`? | K-Means depends on the starting centroids; running it many times and keeping the best inertia avoids bad local minima. |
| Elbow vs silhouette? | Elbow: where inertia stops dropping fast (ambiguous here). Silhouette: cohesion vs separation per point, −1…1; it clearly peaked at k = 2. |
| Silhouette formula? | s = (b − a) / max(a, b); a = mean distance to own cluster, b = mean distance to nearest other cluster. |
| Ward linkage? | Merges the pair of clusters that increases total within-cluster variance the least. |
| How do you read a dendrogram? | Height = merge distance. Cut where the vertical gap is largest; here that is the last merge, giving 2 clusters. |
| GMM vs K-Means? | GMM fits Gaussians with EM and allows elliptical clusters plus **soft** probabilities; K-Means makes hard, spherical assignments. |
| What is BIC? | Log-likelihood penalised by the number of parameters; lower is better. It guards against too many components. |
| DBSCAN parameters? | `eps` (neighbourhood radius) and `min_samples`. Core points have ≥ min_samples neighbours within eps; points not reachable from a core point are noise. eps was chosen from the k-distance plot. |
| Why did DBSCAN fail? | The benign and malignant groups touch with no low-density gap, so DBSCAN sees one dense region plus sparse outliers. |
| ARI and NMI? | External metrics comparing clusters with the true labels. ARI is chance-corrected (0 = random, 1 = perfect); NMI is shared information (0–1). |
| Did you use labels for clustering? | No. Labels were used only at the end to evaluate (ARI, NMI, purity) and to name the clusters. |
| Why is unsupervised useful if supervised is better? | Labels are expensive (they need a pathologist). Clustering explores data, finds suspicious outliers, and can pre-screen or support semi-supervised learning. |

---

## Common questions (anyone may be asked)
* **Supervised vs unsupervised?** Supervised learns a mapping from labelled examples; unsupervised finds structure in unlabelled data.
* **Why this dataset?** It is real medical data, clean, and has a clear binary label, so both programs can use the same data and be compared directly.
* **Biggest takeaway?** The signal is strong enough that clustering alone gets 91%. With labels and careful methodology (no leakage, recall-oriented threshold) we reach 98% while missing only 1 of 42 cancers.
* **Limitations?** Only 569 patients from one centre; this would need external validation and should only ever support a doctor's decision, not replace it.
