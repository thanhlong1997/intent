import pandas as pd
df=pd.read_csv('C:\\Users\\Luong Thanh Long\\PycharmProjects\\Feature - Copy\\data.csv')
tags={1:'search'}
training_data = {}
for index in tags:
    training_data[tags[index]]=[]

for item in range(2250):
    training_data[tags[df['label'][item]]].append(df['text'][item])

