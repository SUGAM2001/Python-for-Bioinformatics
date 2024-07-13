# import library
from sklearn.preprocessing import StandardScaler

exp_count = exp_count
count_columns = count_columns

# Create an instance of StandardScaler
scaler = StandardScaler()

# Fit and transform the data
exp_scaled = scaler.fit_transform(exp_count)

# Create a DataFrame with the scaled data and specified column names
count_scaled = pd.DataFrame(exp_scaled, columns=count_columns)

# Print the first few rows of the DataFrame
print(count_scaled.head())
