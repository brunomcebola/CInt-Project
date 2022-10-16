import sys
import joblib
import numpy as np
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


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
        if y_actual[i] == y_hat[i] == 0:
            TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            FN += 1

    return (TP, FP, TN, FN)


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

print("Removes outliers: " + str(len(outliers)))
print()

for i, row in df_output.items():
    if row == 3:
        df_output[i] = 1
    else:
        df_output[i] = 0

df_features["S1Light"] = (df_features["S1Light"] - df_features["S1Light"].min()) / (
    df_features["S1Light"].max() - df_features["S1Light"].min()
)
df_features["S2Light"] = (df_features["S2Light"] - df_features["S2Light"].min()) / (
    df_features["S2Light"].max() - df_features["S2Light"].min()
)
df_features["S3Light"] = (df_features["S3Light"] - df_features["S3Light"].min()) / (
    df_features["S3Light"].max() - df_features["S3Light"].min()
)

# predict and confusion matrix
X = df_features.to_numpy()
X = np.concatenate((X[:, 3:6], X[:, 7:9]), axis=1)
Y = df_output.to_numpy()


clf = joblib.load("trained_mlp_fuzzy.sav")
lo_sim = joblib.load("lo_sim.sav")
po_sim = joblib.load("po_sim.sav")
output_sim = joblib.load("output_sim.sav")

TP = TN = FP = FN = 0

y_pred = []

# test
for id, x in enumerate(X):
    # compute total light
    lo_sim.input["ls1"] = x[0]
    lo_sim.input["ls2"] = x[1]
    lo_sim.input["ls3"] = x[2]

    lo_sim.compute()
    lo_mix_in = lo_sim.output["lo"]

    # compute total motion
    po_sim.input["p1"] = x[3]
    po_sim.input["p2"] = x[4]

    po_sim.compute()
    po_mix_in = po_sim.output["po"]

    # compute mix
    output_sim.input["lo_mix"] = lo_mix_in
    output_sim.input["po_mix"] = po_mix_in

    output_sim.compute()

    correct = Y[id]
    predicted = 0 if output_sim.output["output"] < 0.5 else 1

    y_pred.append(predicted)

    if predicted == 1 and correct == 1:
        TP += 1
    elif predicted == 0 and correct == 0:
        TN += 1
    elif predicted == 1 and correct == 0:
        FP += 1
    elif predicted == 0 and correct == 1:
        FN += 1

print("Fuzzy")
print(f"Precision: {TP/(TP+FP)}")
print(f"Recall: {TP/(TP+FN)}")
print(f"F1: {2*TP/(2*TP+FP+FN)}")
print()

ConfusionMatrixDisplay.from_predictions(Y, y_pred, cmap=mpl.cm.Blues)  # type: ignore
plt.title("Fuzzy")
plt.show(block=False)

# MLP to compare

y_pred = clf.predict(X)

TP, FP, TN, FN = perf_measure(Y, y_pred)

print("MLP")
print(f"Precision: {TP/(TP+FP)}")
print(f"Recall: {TP/(TP+FN)}")
print(f"F1: {2*TP/(2*TP+FP+FN)}")

ConfusionMatrixDisplay.from_estimator(clf, X, Y, cmap=mpl.cm.Blues)  # type: ignore
plt.title("MLP")
plt.show()
