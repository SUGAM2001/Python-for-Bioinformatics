import pandas as pd
from sklearn.preprocessing import MinMaxScaler

file_path = ""
data = pd.read_csv(file_path)

# Select the count col
count_col = ['count.C1', 'count.C2', 'count.C3', 'count.T1', 'count.T2', 'count.T3']

#Extract the count data
count_data = data[count_col]

scaler = MinMaxScaler()
scaled_count_data = scaler.fit_transform(count_data)

# To save it in DataFrame
save_df = pd.DataFrame(scaled_count_data, columns = count_col)

# If you want to replace your original data
data[count_col] = scaled_count_data






