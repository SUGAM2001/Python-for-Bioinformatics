import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "C:/Users/sugam patel/Downloads/filtered_gene_expression_counts.csv" 

data = pd.read_csv(file_path)
data.head()

# Melt the DataFrame to 'long-form' or 'tidy' format for Seaborn
melted_data = data.melt(var_name='Sample', value_name='Expression')

# Add a column to differentiate between control and treatment samples
melted_data['Type'] = melted_data['Sample'].apply(lambda x: 'Control' if 'C' in x else 'Treatment')

# Display the melted DataFrame
print(melted_data.head())

plt.figure(figsize=(10, 6))
sns.boxplot(x='Sample', y='Expression', hue='Type', data=melted_data, palette=['blue', 'orange'])

# Customize the plot
plt.title('Gene Expression Box Plot')
plt.xlabel('Sample')
plt.ylabel('Expression Level')
plt.xticks(rotation=45)

