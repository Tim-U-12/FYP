from dataset import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample

filepaths, ages, genders, races = load_or_process_data()

test_data = pd.DataFrame(filepaths, columns=['filepaths'])
test_data['age'] = ages
test_data['gender'] = genders
test_data['race'] = races

print(test_data['age'].value_counts())

class0 = test_data[test_data.age == 0] # 3062
class1 = test_data[test_data.age == 1] # 1531
class2 = test_data[test_data.age == 2] # 7344
class3 = test_data[test_data.age == 3] # 4536
class4 = test_data[test_data.age == 4] # 2245
class5 = test_data[test_data.age == 5] # 2299
class6 = test_data[test_data.age == 6] # 1316
class7 = test_data[test_data.age == 7] # 699 [OLD]
class8 = test_data[test_data.age == 8] # 504 [OLD]
class9 = test_data[test_data.age == 9] # 137 [OLD]
class10 = test_data[test_data.age == 10] # 21 [OLD]
class11 = test_data[test_data.age == 11] # 11 [OLD]
classOld = test_data[test_data.age >= 7] # 1372 [OLD]

print(classOld.__len__()) # 1372

classOld.replace(8, 7, inplace=True)
classOld.replace(9, 7, inplace=True)
classOld.replace(10, 7, inplace=True)
classOld.replace(11, 7, inplace=True)

zeroUndersampled = resample(class0, replace=False, n_samples=len(class5))
oneOversampled = resample(class1, replace=True, n_samples=len(class5))
twoUndersampled = resample(class2, replace=False, n_samples=len(class5))
threeUndersampled = resample(class3, replace=False, n_samples=len(class5))
fourOversampled = resample(class4, replace=True, n_samples=len(class5))
class5
sixOversampled = resample(class6, replace=True, n_samples=len(class5))
oldOversampled = resample(classOld, replace=True, n_samples=len(class5))

balanced_data = pd.concat([zeroUndersampled, oneOversampled, twoUndersampled, threeUndersampled, fourOversampled,
                           class5, sixOversampled, oldOversampled])

print(balanced_data.__len__())

balanced_data.to_csv('balancedAge.csv', index=False)