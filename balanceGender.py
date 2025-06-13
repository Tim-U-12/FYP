from dataset import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample

filepaths, ages, genders, races = load_or_process_data()

test_data = pd.DataFrame(filepaths, columns=['filepaths'])
test_data['age'] = ages
test_data['gender'] = genders
test_data['race'] = races

print(test_data['gender'].value_counts())

maleClass = test_data[test_data.gender == 0]
femaleClass = test_data[test_data.gender == 1]

maleUndersampled = resample(maleClass, replace=False, n_samples=len(femaleClass))

balanced_data = pd.concat([maleUndersampled, femaleClass])

print(balanced_data)

balanced_data.to_csv('balancedGender.csv', index=False)