import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2
from statsmodels.stats.multitest import multipletests

file_path = "C:/Users/sugam patel/Downloads/filtered_gene_expression_counts.csv"

df = pd.read_csv(file_path)

# Separate the features (gene expression data) and the target variable (class labels)
X = df.iloc[:, :-1]  # All columns except the last are features (genes)
y = df.iloc[:, -1]   # The last column is the target variable


chi2_values, p_values = chi2(X, y)


# Apply FDR correction using the Benjamini-Hochberg procedure
fdr_adjusted_p_values = multipletests(p_values, alpha=0.05, method='fdr_bh')[1]

# Create a DataFrame to display the results
results = pd.DataFrame({
    'Feature': X.columns,
    'Chi2 Statistic': chi2_values,
    'p-value': p_values,
    'FDR-adjusted p-value': fdr_adjusted_p_values
})

# Display the results
print(results.sort_values(by='FDR-adjusted p-value'))


