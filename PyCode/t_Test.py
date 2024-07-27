# import library
import pandas as pd
import numpy as np
import scipy.stats as stats

file_path = "C:/Users/sugam patel/Downloads/filtered_gene_expression_counts.csv"

data = pd.read_csv(file_path)
data

# Separating the data into different groups
control = data.loc[:,'count.C1':'count.C3']
Treatment = data.loc[:,'count.T1':'count.T3']


# convet dataframe to Numpy array
control_val = control.values
Treatment_val = Treatment.values

# compute t-test for each row 
t_stats,p_values = stats.ttest_ind(control_val,Treatment_val, axis = 1)

# combine results into Dataframe
t_test_result = pd.DataFrame({'t-test':t_stats,'p-values':p_values}, index = data.index)

t_test_result
