# import library
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

file_data = "C:/Users/sugam patel/Downloads/filtered_gene_expression_counts.csv"
data = pd.read_csv(file_data)

# Perform PCA
pca = PCA(n_components = 2)
pca_result = pca.fit_transform(data)

# Convert pca result to dataframe
pca_df = pd.DataFrame(pca_result,columns = ['PC1','PC2'])

# Plot PCA result

plt.figure(figsize=(10,6))
sns.scatterplot(x = 'PC1', y = 'PC2', data = pca_df)
plt.title("PCA of Count Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
