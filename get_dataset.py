import pandas as pd

csv_file = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data',header=None) 

df = csv_file[[0,len(csv_file.columns)-1]]
df.columns = ['label', 'value']
grouped_by_state = df.groupby('label')
aa = grouped_by_state['value'].sum()/grouped_by_state['value'].count()
final_frame = pd.DataFrame([aa.index,aa.values],).transpose()
final_frame.columns=['label','value']

