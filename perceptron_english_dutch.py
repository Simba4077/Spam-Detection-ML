import numpy as np
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
y_train = np.array(y_train, dtype=float)

X_other = np.array(X_other, dtype=float)
y_other = np.array(y_other, dtype=float)

# print(test_matrix.shape)
# print(other.shape)
# print(test_matrix[:5])

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

#function to classify a single example
def classify(weights, bias, x):
    #linear combo + bias
    total = weights.dot(x) + bias
    if total > 0:
        return 1
    else:
        return -1
    

#function to train perceptron
def train_perceptron(X, y, n_epochs):
    _, n_features = X.shape #divide rows and columns -> samples, features
    weights = np.zeros(n_features) #initialize weights to 0
    bias = 0 #initialize bias to 0

    for _ in range(n_epochs):
        for index, x in enumerate(X):
            output = classify(weights, bias, x)
            #update rule
            if y[index] * output <= 0:
                weights +=  y[index] * x
                bias += y[index]
    return weights, bias

def confusion_matrix(X, y, weights, bias):
    """
                Predicted
    
     Actual     |   1  1  |   -1  1  |
                |   1 -1  |   -1 -1  |
     
    -------------------------------
    TP | FP
    FN | TN

    """
    TP = FP = TN = FN = 0
    for index, x in enumerate(X):
        output = classify(weights, bias, x)
        if y[index] == 1 and output == 1:
            TP += 1
        elif y[index] == 1 and output == -1:
            FN += 1
        elif y[index] == -1 and output == 1:
            FP += 1
        elif y[index] == -1 and output == -1:
            TN += 1
    return TP, FP, TN, FN

#train perceptron on spam dataset
epoch_weights = {}
epoch_biases = {}
n_epochs_list = [int(x.strip()) for x in input("Enter different epochs to train on (e.g. 1,2,3,4,5,...): ").split(',')]
for epoch in n_epochs_list:
    weights, bias = train_perceptron(X_train, y_train, int(epoch))
    epoch_weights[epoch] = weights
    epoch_biases[epoch] = bias
print("Found corresponding weights and bias for each epoch")

accuracies = {}
#evaluate on dev set
for n_epochs in epoch_weights.keys():
    weights = epoch_weights[n_epochs]
    bias = epoch_biases[n_epochs]
    print(f"Evaluating on development set with weights and bias from {n_epochs} epochs of training...")
    TP, FP, TN, FN = confusion_matrix(X_dev, y_dev, weights, bias)
    print(f"Confusion Matrix (total count of each):\nTP: {TP:.2f}, FP: {FP:.2f}, TN: {TN:.2f}, FN: {FN:.2f}")
    accuracy = (TP + TN) / (len(y_dev)) * 100
    print(f"Accuracy: {accuracy}%")
    print("--------------------------------------------------------")
    accuracies[n_epochs] = accuracy

#select best epoch based on dev set accuracy
best_epoch = max(accuracies, key=accuracies.get)

#trained perceptron with best epoch, evaluate on test set
print("--------------------------------------------------------")
print(f"Evaluating on test set with weights and bias from {best_epoch} epochs of training...")
weights = epoch_weights[best_epoch]
bias = epoch_biases[best_epoch]
TP, FP, TN, FN = confusion_matrix(X_test, y_test, weights, bias)
print(f"Confusion Matrix (total count of each):\nTP: {TP:.2f}, FP: {FP:.2f}, TN: {TN:.2f}, FN: {FN:.2f}")
accuracy = (TP + TN) / (len(y_test)) * 100
print(f"Accuracy: {accuracy}%")