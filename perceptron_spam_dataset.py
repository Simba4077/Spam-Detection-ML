import numpy as np
from ucimlrepo import fetch_ucirepo  
# fetch dataset 
spambase = fetch_ucirepo(id=94) 
  
# data convert to numpy 
X = spambase.data.features.to_numpy() 
y = spambase.data.targets.to_numpy().ravel()

#from https://stackoverflow.com/a/4602224 
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    shuffled_indices = np.random.permutation(len(a))
    return a[shuffled_indices], b[shuffled_indices]

X, y = unison_shuffled_copies(X,y)
y = np.array([-1 if yval == 0 else 1 for yval in y.tolist()])

rows, _ = X.shape
test_count = rows // 10      # 10%
dev_count = rows // 10       # 10%
train_count = rows - test_count - dev_count  # 80%

X_train = X[:train_count]
y_train = y[:train_count]

X_dev = X[train_count:train_count + dev_count]
y_dev = y[train_count:train_count + dev_count]

X_test = X[train_count + dev_count:]
y_test = y[train_count + dev_count:]

#quick sanity check
    # print(f"Training set size: {X_train.shape[0]} samples") #should be 80%, 3681
    # print(f"Development set size: {X_dev.shape[0]} samples") #should be 10%, 460
    # print(f"Test set size: {X_test.shape[0]} samples") #should be 10%, 460
    # print(X_test[459])


#function to classify a single example
def classify(weights, bias, x):
    #linear combo + bias
    total = weights.dot(x) + bias
    if total > 0:
        return 1
    else:
        return -1


#testing classify function above, should return -1
    # t = [2,-3,-4,1]   
    # w = np.array([1,1,1,1])
    # print(classify(w, 1, t))


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


#confusion matrix
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
print()
n_epochs_list = [int(x.strip()) for x in input("Enter different epochs to train on (e.g. 1,2,3,4,5,...): ").split(',')]
for epoch in n_epochs_list:
    weights, bias = train_perceptron(X_train, y_train, int(epoch))
    epoch_weights[epoch] = weights
    epoch_biases[epoch] = bias
print("Found corresponding weights and bias for each epoch!")
print("-----------------------------------------------------")

accuracies = {}
#evaluate on dev set
for n_epochs in epoch_weights.keys():
    weights = epoch_weights[n_epochs]
    bias = epoch_biases[n_epochs]
    print(f"Evaluating on development set with weights and bias from {n_epochs} epochs of training...")
    TP, FP, TN, FN = confusion_matrix(X_dev, y_dev, weights, bias)
    print(f"Confusion Matrix (total count of each):\nTP: {TP:.2f}, FP: {FP:.2f}, TN: {TN:.2f}, FN: {FN:.2f}")
    accuracy = (TP + TN) / (len(y_dev)) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    print("--------------------------------------------------------")
    accuracies[n_epochs] = accuracy

#select best epoch based on dev set accuracy and lowest epochs
best_epoch = None
best_acc = -1

for epoch, acc in accuracies.items():
    if acc > best_acc or (acc == best_acc and (best_epoch is None or epoch < best_epoch)):
        best_acc = acc
        best_epoch = epoch


#trained perceptron with best epoch, evaluate on test set
print("--------------------------------------------------------")
print(f"Evaluating on test set with weights and bias from {best_epoch} epochs of training...")
weights = epoch_weights[best_epoch]
bias = epoch_biases[best_epoch]
TP, FP, TN, FN = confusion_matrix(X_test, y_test, weights, bias)
print(f"Confusion Matrix (total count of each):\nTP: {TP:.2f}, FP: {FP:.2f}, TN: {TN:.2f}, FN: {FN:.2f}")
accuracy = (TP + TN) / (len(y_test)) * 100
print(f"Accuracy: {accuracy:.2f}%")
print()








