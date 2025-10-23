# __Spam Detection using Perceptron, SVM, and Logistic Regression ML Algs__

A simple machine learning project using three algorithms (Perceptron, Support Vector Machine (SVM), and Logistic Regression) to classify emails as spam or non-spam based on the UCI Spambase dataset.  

Project Overview:  
This project compares three classic binary classifiers on the same dataset.  



Install requirements: `pip install numpy pandas scikit-learn matplotlib`
Run model script directly using `python [script_name]`

Program will prompt you for hyperparameters:
Logisitic Regression - epochs and C values to train and test on
SVM - epochs and C values to train and test on
Perceptron - epochs to train and test on

ML pipeline will use your inputted hyperparams to train on a split of the spam dataset (~80%), then run alg with the found parameters and their respective hyperparamters, and lastly test on the test set with the hyperparams that yielded the highest accuracy. A confusion matrix will also be printed for each hyperparameter(s) that is used for the dev and test set.


