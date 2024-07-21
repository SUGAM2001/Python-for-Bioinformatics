import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "C:/Users/sugam patel/Downloads/filtered_gene_expression_counts.csv" 

data = pd.read_csv(file_path)
data.head()

# Create Density Plot
plt.figure(figsize=(10, 6))
for sample in data.columns:
    sns.kdeplot(data[sample], label=sample)
plt.title('Density Plot of Gene Expression Levels')
plt.xlabel('Expression Level')
plt.ylabel('Density')
plt.legend()
plt.show()
