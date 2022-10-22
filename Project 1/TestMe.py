import sys
import joblib
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, classification_report


def getOutliersFromAverage(data, gain):
    outliers = []

    for feature in data.columns:
        average = data[feature].mean()
        std = data[feature].std()

        outliers += (
            data[feature][data[feature] <= (average - std * gain)].index.tolist()
            + data[feature][data[feature] >= (average + std * gain)].index.tolist()
        )

    return list(dict.fromkeys(outliers))


def removeLine(data, indexes):
    return data.drop(indexes)


# read csv file
df = pd.read_csv(sys.argv[1])

for feature in df.columns:
    df = df[df[feature].notna()]

df_features = df.drop(["Date", "Time", "Persons"], axis=1)

df_output = df["Persons"]

# z score
for feature in df_features.columns:
    if feature == "PIR1" or feature == "PIR2":
        continue

    df_features[feature] = (df_features[feature] - df_features[feature].mean()) / df_features[feature].std(ddof=0)

# remove outliers
outliers = getOutliersFromAverage(df_features, 6)
df_features = removeLine(df_features, outliers)
df_output = removeLine(df_output, outliers)

print("Removed outliers: " + str(len(outliers)))
print()

# moving average
for feature in df_features.columns:
    if feature == "PIR1" or feature == "PIR2":
        continue

    df_features[feature] = df_features[feature].rolling(75).mean()

for feature in df_features.columns:
    df_output = df_output[df_features[feature].notna()]
    df_features = df_features[df_features[feature].notna()]

# normalization with min-max
for feature in df_features.columns:
    if feature == "PIR1" or feature == "PIR2":
        continue

    df_features[feature] = (df_features[feature] - df_features[feature].min()) / (
        df_features[feature].max() - df_features[feature].min()
    )

# predict and confusion matrix
X = df_features.to_numpy()
Y = df_output.to_numpy()

clf = joblib.load("trained_mlp.sav")

y_pred = clf.predict(X)

print(classification_report(Y, y_pred))  # type: ignore

ConfusionMatrixDisplay.from_estimator(clf, X, Y, cmap=mpl.cm.Blues)  # type: ignore
plt.title("MLP")
plt.show()
