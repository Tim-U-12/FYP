from dataset import extractCSVLabels
from utils import UTKLabelType

if __name__ == "__main__":
    labelType = UTKLabelType.GENDER
    filepath = f"data/BalancedDatasets/balanced{labelType}.csv"
    csvPaths, csvLabels = extractCSVLabels(filepath, labelType)
    
    print(csvLabels)
    print(len(csvPaths), len(set(csvLabels)))
