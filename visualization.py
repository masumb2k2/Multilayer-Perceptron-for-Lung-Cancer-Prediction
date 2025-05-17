import matplotlib
matplotlib.use("TkAgg")
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv('survey lung cancer.csv')
df.columns = [col.strip() for col in df.columns]

# Encode target variable for plotting
df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

# Encode 'GENDER' for plotting
df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 0})

# Show head
print(df.head())

# Histogram for all features
df.hist(figsize=(15, 12))
plt.tight_layout()
plt.show()

# KDE (density) plots for each feature, split by cancer status
features = [col for col in df.columns if col != 'LUNG_CANCER']

plt.subplots(4, 4, figsize=(24, 18))
for idx, col in enumerate(features):
    ax = plt.subplot(4, 4, idx + 1)
    try:
        sns.kdeplot(df[df['LUNG_CANCER'] == 0][col], label="No Cancer", linestyle='-', color='black')
        sns.kdeplot(df[df['LUNG_CANCER'] == 1][col], label="Cancer", linestyle='--', color='black')
    except Exception:
        sns.histplot(df, x=col, hue='LUNG_CANCER', multiple="stack", ax=ax)
    ax.set_title(col)
    ax.yaxis.set_ticklabels([])
    if idx == 0:
        ax.legend()
# Hide the 16th subplot (bottom right) since we have only 15 features
plt.subplot(4,4,16).set_visible(False)
plt.tight_layout()
plt.show()
