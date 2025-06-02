from dataset import extractCSVLabels
from utils import UTKLabelType

if __name__ == "__main__":
    filepath = "../data/Balanced Datasets/balancedAge.csv"
    csvPaths, csvLabels = extractCSVLabels(filepath, UTKLabelType.GENDER)
    print(len(csvPaths), len(csvLabels))
