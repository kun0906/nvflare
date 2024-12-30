import pandas as pd

# Sample Data
data = {'Region': ['East', 'East', 'West', 'West', 'North'],
        'Product': ['A', 'B', 'A', 'B', 'A'],
        'Sales': [100, 200, 150, 250, 130],
        'Profit': [10, 20, 15, 25, 13]}

df = pd.DataFrame(data)

df1 = df.groupby(['Region', 'Product']).agg({'Sales': 'sum'})

# # Grouping by 'Region' and 'Product', summing 'Sales'
grouped = df.groupby(['Region', 'Product']).agg({'Sales': 'sum', 'Sales': 'count'}).reset_index()
#
# print(grouped)

result = df.groupby(['Region', 'Product']).agg({
    'Sales': ['sum', 'mean', 'count'],
    'Profit': ['sum', 'mean']
})

print(result.columns.values)



