import pandas as pd
import numpy as np
tags={1:'search',0:'diff'}

def load_excel_training_data(excel_file,excel_sheet):
    df = pd.read_csv(excel_file, encoding="UTF-8")
    training_data={}

    for index in tags:
        training_data[tags[index]]=[]
    for item in range(2150):
        training_data[tags[df['label'][item]]].append(df['text'][item])
    for item in range(2151,4068):
        training_data['diff'].append(df['text'][item])
    return training_data

def read_excel(excel_file,excel_sheet= "Sheet1"):
    df=pd.read_excel(excel_file, encoding="UTF-8")
    return df

def read_special(excel_file):
    df = read_excel(excel_file)
    list2 = []
    for item in df.index:
        list2.append(df['append'][item])
    return list2

def split_data(training_data, ratio):
    testing_data = {}
    for tag in training_data:
        training_data[tag] = np.asarray(training_data[tag])
        train_numbers = np.random.choice(training_data[tag].shape[0], round(training_data[tag].shape[0] * ratio),
                                         replace=False)
        test_numbers = np.array(list(set(range(training_data[tag].shape[0])) - set(train_numbers)))
        testing_data[tag] = list(training_data[tag][test_numbers])
        training_data[tag] = list(training_data[tag][train_numbers])
    return training_data, testing_data