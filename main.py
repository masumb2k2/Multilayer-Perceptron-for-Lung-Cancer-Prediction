import matplotlib
matplotlib.use("TkAgg")
from utils import preprocess
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(16)

try:
    df = pd.read_csv('survey lung cancer.csv')
except:
    print("""
      Dataset not found in your computer.
      Please ensure 'survey lung cancer.csv' is in the same directory as this script.
      """)
    quit()

# Preprocess data (replace 0s with mean except for gender, standardize, etc.)
df = preprocess(df)

# Split the data into training and testing sets
X = df.loc[:, df.columns != 'LUNG_CANCER']
y = df.loc[:, 'LUNG_CANCER']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build neural network in Keras
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=200, verbose=False)

# Results - Accuracy
scores = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: %.2f%%\n" % (scores[1]*100))
scores = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy: %.2f%%\n" % (scores[1]*100))

# Results - Confusion Matrix
y_test_pred = (model.predict(X_test) > 0.5).astype("int32")
c_matrix = confusion_matrix(y_test, y_test_pred)
ax = sns.heatmap(c_matrix, annot=True, xticklabels=['No Cancer', 'Cancer'], yticklabels=['No Cancer', 'Cancer'], cbar=False, cmap='Blues')
ax.set_xlabel("Prediction")
ax.set_ylabel("Actual")
plt.show()
plt.clf()

# Results - ROC Curve
y_test_pred_probs = model.predict(X_test)
FPR, TPR, _ = roc_curve(y_test, y_test_pred_probs)
plt.plot(FPR, TPR)
plt.plot([0,1],[0,1],'--', color='black') # diagonal line
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
plt.clf()
