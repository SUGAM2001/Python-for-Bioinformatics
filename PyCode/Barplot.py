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

# Calculate mean and standard deviation
summary_df = melted_data.groupby(['Sample', 'Type'])['Expression'].agg(['mean', 'std']).reset_index()

# Create Bar Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Sample', y='mean', hue='Type', data=summary_df, palette=['blue', 'orange'], capsize=0.1)
plt.errorbar(x=summary_df['Sample'], y=summary_df['mean'], yerr=summary_df['std'], fmt='none', c='black', capsize=5)
plt.title('Bar Plot of Mean Gene Expression Levels')
plt.xlabel('Sample')
plt.ylabel('Mean Expression Level')
plt.xticks(rotation=45)
plt.show()

