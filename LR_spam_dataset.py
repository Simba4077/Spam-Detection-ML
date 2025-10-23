from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
import numpy as np
from ucimlrepo import fetch_ucirepo  
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

"""
    Underlying computation for SVMs:
    Do this for n = epochs iterations
    1. Classify points y= wx + b
    2. Find margin = y(w*x+b)
    3. Check if point is correctly classified and outside margin:
        a. If y(w*x+b)>=1, then the point is correctly classified and outside the margin, can regularize to shrink weights by doing w=w-lr*w
        b. If y(w*x+b)<1, then the point is misclassified or outside origin --> must do updates for weights and bias
            updates:
                w = w + lr * C * y * x
                b = b + lr * C * y 

"""
#dataset pre-processing

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


#training
print("=========================================================")
print()


n_epochs_list = [int(x.strip()) for x in input("Enter epochs to train on (e.g. 1,2,3,4,5,...): ").split(',')]
c_list = [int(x.strip()) for x in input
          ("Enter positive C values to train on (e.g. 1,2,3,4,5,...).\n" \
            "C values correspond with epoch values based on their index, ensure # of C values == # of epoch values: ").split(',')]

print()
print("=========================================================")
print("Now printing results...")
print("=========================================================")
print()

best_acc = -1
best_params = (float('inf'),float('inf'))
best_model = None
for epoch, C in zip(n_epochs_list, c_list):
    LR = LogisticRegression(C=C, max_iter =epoch)
    LR.fit(X_train, y_train)
    print(f"Finished training with epoch: {epoch} and C: {C}")
    y_pred = LR.predict(X_dev) #predict label on dev set
    acc = accuracy_score(y_dev, y_pred) * 100
    cm = confusion_matrix(y_dev, y_pred)
    print(f"The accuracy with epoch {epoch} and C {C} using the dev set is {acc:.2f}%")
    print(f"Confusion Matrix (total count of each) with epoch {epoch} and C {C} on dev set:\nTN: {cm[0][0]:.2f}, FP: {cm[0][1]:.2f}, FN: {cm[1][0]:.2f}, TP: {cm[1][1]:.2f}")
    print()
    print("--------------------------------------------------------------------------")
    print()

    if acc > best_acc or (acc == best_acc and epoch < best_params[1]):
        best_acc = acc
        best_params = (C, epoch)
        best_model = LR

y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test,y_pred) * 100
cm = confusion_matrix(y_test, y_pred)
print("======================================================")
print("Now printing results on test set...")
print("======================================================")
print(f"Using epoch: {best_params[1]} and C: {best_params[0]} on test set")
print(f"The accuracy: {acc:.2f}%")
print(f"Confusion Matrix (total count of each):\nTN: {cm[0][0]:.2f}, FP: {cm[0][1]:.2f}, FN: {cm[1][0]:.2f}, TP: {cm[1][1]:.2f}")
print()










