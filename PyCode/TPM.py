import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = "C:/Users/sugam patel/Downloads/filtered_gene_expression_counts.csv"
# Example: Load data from a CSV file
df = pd.read_csv(file_path)

column_sums = df.sum(axis=0)

print(column_sums)


# Boxplot of gene expression values across samples
sns.boxplot(data=df)
plt.show()

# check through the violonplot
sns.violinplot(data=df)
plt.show()


