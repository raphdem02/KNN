import pandas as pd 
df = pd.read_csv("data2/data.txt",header = None, delimiter=";")

df = df.iloc[:,:-1]
df = df.select_dtypes(include=['float64'])

df.to_csv("data2/data_modified.txt", header = None, index = None, sep = ";", float_format = "%.6f")