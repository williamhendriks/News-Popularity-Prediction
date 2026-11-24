# Predicting Online News Popularity

Group project for the course Machine Learning (University of Amsterdam). In collaboration with Xanthos Kekkos.

Can the popularity of a news article be predicted from information available before it is published? This project classifies articles as popular or unpopular, uses clustering to look for natural groups of articles, and explains the best model with SHAP.

## Results

An article counts as popular if it has more than the median number of shares (1,400), which gives two similarly sized classes.

| Model | Mean CV AUC | Test AUC | Test accuracy |
|---|---|---|---|
| Random forest | 0.73 | 0.66 | 0.66 |
| Linear SVC | 0.71 | 0.65 | 0.65 |
| Logistic regression | 0.71 | 0.65 | 0.65 |
| k-nearest neighbors | 0.69 | 0.63 | 0.63 |

- Keeping the 13 top-ranked features (out of 59) lowered the mean CV AUC only from 0.73 to 0.72.
- Outlier handling and engineered features changed the metrics negligibly.
- Clusters were interpretable, but barely related to popularity (adjusted Rand index of about 0.015 at best).
- In the SHAP analysis, weekend publication, keyword-related features, and shares of linked articles had the most influence.

## What is in this repository

| File | Contents |
|---|---|
| `classification.ipynb` | Scaling comparison, hyperparameter tuning, random forest, feature selection, outlier detection, feature engineering, SHAP, and a function to test what-if articles |
| `clustering.ipynb` | K-means and agglomerative clustering, PCA, and cluster analysis |
| `report.pdf` | The full written report |

## Data

The dataset is not included. Download the Online News Popularity dataset from the UCI Machine Learning Repository and place `OnlineNewsPopularity.csv` in the same folder as the notebooks.

## Running the notebooks

1. Install the libraries: `pandas numpy scikit-learn matplotlib seaborn plotly scipy shap tqdm joblib`.
2. Open a notebook in Jupyter and run the cells in order. The grid searches are slow.

## Reference

Fernandes, K., Vinagre, P., and Cortez, P. (2015). A proactive intelligent decision support system for predicting the popularity of online news. In Portuguese Conference on Artificial Intelligence, pages 535-546. Springer.
