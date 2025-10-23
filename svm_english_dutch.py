from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score,confusion_matrix
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning

import re
"""
labels[1 = english, -1 = dutch]

Creates 2 separate .csv file with different features from english and dutch  
1. For training set (uni declaration) 
2. For dev/test set (40 translations from additional text)

Feature extractions:
1. Char patterns : 
    a. Dutch : ij, sch, aa, oo, ee, ijk, cht
    b. English : th, sh, ing, ion, ough
2. Whole words :
    a. Dutch: van, de, en
    b. English: the, and, has
3. Average Word Length : Dutch longer words than English

"""
with open("universal-declaration/english.txt", "r", encoding="utf-8") as f:
    english_test_text = [line.strip() for line in f if line.strip()]

with open("universal-declaration/dutch.txt", "r", encoding="utf-8") as f:
    dutch_test_text = [line.strip() for line in f if line.strip()]

with open("other-translation/english.txt", "r", encoding="utf-8") as f:
    english_other_text = [line.strip() for line in f if line.strip()]

with open("other-translation/dutch.txt", "r", encoding="utf-8") as f:
    dutch_other_text = [line.strip() for line in f if line.strip()]

#print(dutch_other_text)

def extract_features(text):
    text = text.lower()
    features = {
            "count_ij": text.count("ij"),
            "count_sch": text.count("sch"),
            "count_th": text.count("th"),
            "count_oo": text.count("oo"),
            "count_aa": text.count("aa"),
            "count_uu": text.count("uu"),
            "count_the": len(re.findall(r"\bthe\b", text)),
            "count_de": len(re.findall(r"\bde\b", text)),
            "count_en": len(re.findall(r"\ben\b", text)),
            "count_van": len(re.findall(r"\bvan\b", text)),
            "count_and": len(re.findall(r"\band\b", text)),
            "count_has": len(re.findall(r"\bhas\b", text)),
            "avg_word_len": np.mean([len(w) for w in text.split()]) if text.split() else 0
         }
    return features

X_train = []
y_train = []
X_other = []
y_other = []
for text in english_test_text:
    features = extract_features(text)
    X_train.append(list(features.values()))
    y_train.append([1])
    
for text in dutch_test_text:
    features = extract_features(text)
    X_train.append(list(features.values()))
    y_train.append([-1])


for text in english_other_text:
    features = extract_features(text)
    X_other.append(list(features.values()))
    y_other.append([1])

for text in dutch_other_text:
    features = extract_features(text)
    X_other.append(list(features.values()))
    y_other.append([-1])

X_train = np.array(X_train, dtype=float)
y_train = np.array(y_train, dtype=float).ravel()

X_other = np.array(X_other, dtype=float)
y_other = np.array(y_other, dtype=float).ravel()



#shuffle 
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    shuffled_indices = np.random.permutation(len(a))
    return a[shuffled_indices], b[shuffled_indices]

X_train, y_train = unison_shuffled_copies(X_train,y_train)
X_other, y_other = unison_shuffled_copies(X_other,y_other)

#split other into dev and test sets
X_dev = X_other[:20]
y_dev = y_other[:20]

X_test = X_other[20:40]
y_test = y_other[20:40]

print(y_dev.shape)

warnings.filterwarnings("ignore", category=ConvergenceWarning)


print("=========================================================")
print()


n_epochs_list = [int(x.strip()) for x in input("Enter epochs to train on (e.g. 1,2,3,4,5,...):\n").split(',')]
c_list = [int(x.strip()) for x in input
          ("Enter positive C values to train on (e.g. 1,2,3,4,5,...).\n" \
            "C values correspond with epoch values based on their index, ensure # of C values == # of epoch values:\n").split(',')]

print()
print("=========================================================")
print("Now printing results...")
print("=========================================================")
print()

best_acc = -1
best_params = (float('inf'),float('inf'))
best_model = None
for epoch, C in zip(n_epochs_list, c_list):
    svm = LinearSVC(C=C, max_iter =epoch)
    svm.fit(X_train, y_train)
    print(f"Finished training with epoch: {epoch} and C: {C}")
    y_pred = svm.predict(X_dev) #predict label on dev set
    acc = accuracy_score(y_dev, y_pred) * 100
    cm = confusion_matrix(y_dev, y_pred)
    print(f"The accuracy with epoch {epoch} and C {C} using the dev set is {acc}%")
    print(f"Confusion Matrix (total count of each) with epoch {epoch} and C {C} on dev set:\nTN: {cm[0][0]:.2f}, FP: {cm[0][1]:.2f}, FN: {cm[1][0]:.2f}, TP: {cm[1][1]:.2f}")
    print()
    print("--------------------------------------------------------------------------")
    print()

    if acc > best_acc or (acc == best_acc and epoch < best_params[1]):
        best_acc = acc
        best_params = (C, epoch)
        best_model = svm

y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test,y_pred) * 100
cm = confusion_matrix(y_test, y_pred)
print("======================================================")
print("Now printing results on test set...")
print("======================================================")
print(f"Using epoch: {best_params[1]} and C: {best_params[0]} on test set")
print(f"The accuracy: {acc}%")
print(f"Confusion Matrix (total count of each):\nTN: {cm[0][0]:.2f}, FP: {cm[0][1]:.2f}, FN: {cm[1][0]:.2f}, TP: {cm[1][1]:.2f}")
print()