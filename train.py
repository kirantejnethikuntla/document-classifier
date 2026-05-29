import pandas as pd

data = pd.read_csv("data/spam_updated.csv", encoding="latin-1")

data = data.iloc[:, :2]
data.columns = ['label', 'message']

data['label'] = data['label'].replace({
    'ham': 'not spam',
    'spam': 'spam'
})

data.to_csv("data/spam_updated.csv", index=False)

print("Dataset updated successfully!")