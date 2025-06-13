from dataset import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample

filepaths, ages, genders, races = load_or_process_data()

test_data = pd.DataFrame(filepaths, columns=['filepaths'])
test_data['age'] = ages
test_data['gender'] = genders
test_data['race'] = races

class0 = test_data[test_data.race == 0] # 10078
class1 = test_data[test_data.race == 1] # 4526
class2 = test_data[test_data.race == 2] # 3434
class3 = test_data[test_data.race == 3] # 3975
class4 = test_data[test_data.race == 4] # 1692

zeroUndersampled = resample(class0, replace=False, n_samples=len(class2))
oneUndersampled = resample(class1, replace=False, n_samples=len(class2))
class2
threeUndersampled = resample(class3, replace=False, n_samples=len(class2))
fourOversampled = resample(class4, replace=True, n_samples=len(class2))

balanced_data = pd.concat([zeroUndersampled, oneUndersampled, class2, threeUndersampled, fourOversampled])

print(balanced_data)

balanced_data.to_csv('balancedRace.csv', index=False)