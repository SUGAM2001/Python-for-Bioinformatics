# Import important library
import pandas as pd
file_path = "C:/Users/sugam patel/Downloads/GSE268894_1_genes_fpkm_expression.txt/GSE268894_1_genes_fpkm_expression.txt"

exp = pd.read_table(file_path)

# print the header of the data
exp.head()
# check the shape of the file
print(exp.shape)
# Display basic information about the DataFrame
print(exp.info())

# To check the col name 
print(exp.columns)
# To check the column shape
print(exp.columns.shape)

# In this file there are multiple columns with the fpkm values, so i only selct the expression column

count_columns = ['count.C1', 'count.C2', 'count.C3', 'count.T1', 'count.T2', 'count.T3']
exp = exp[count_columns]
print(exp)

# Display summary statistics for the numerical columns
print(exp.describe())

# Check for missing values
print(exp.isna().sum())


# Save the DataFrame to a CSV file
exp_count.to_csv('filtered_gene_expression_counts.csv', index=False)
