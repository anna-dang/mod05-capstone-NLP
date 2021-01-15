""" This file contains functions to for NLP and modeling for my capstone project. 01/14/2021 """

from sklean.metrics import classification_report, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



def evaluate_model(model, X, labels, return_preds=False, norm_type='true'):
    """ Given model and predictors/labels, returns model performance evaluation
        as Classification Report Table and two confusion matrices (one with 
        prediction class distribution and one normalized (default: 'true').
        Also returns a ROC curve with AUC measurement.

        Optional: return predicted labels."""

    # Get binary predictions by round predicted probabilities
    y_hat = np.concatenate(model.predict(X).round())
   
    # Print classification report
    print("---"*20)
    print("Classification Report for Test Data: \n")
    print(metrics.classification_report(labels, y_hat))
    print("---"*20)

    # Print model performance
    print("Loss of the model is - " , model.evaluate(X, labels)[0])
    print("Accuracy of the model is - " , model.evaluate(X, labels)[1]*100 , "%")
    print("---"*20)

    # Print ratio of predictions
    correct = np.sum(y_hat == test_labels)
    incorrect = np.sum(y_hat != test_labels)
    print(f"Correct: {correct}, {round((correct/len(test_labels)*100), 2)}%") 
    print(f"Incorrect: {incorrect}, {round((incorrect/len(test_labels)*100), 2)}%")
    print("---"*20)

    # Build confusion matrix
    print("Model Prediction Results for Test Data:")
    cm = metrics.confusion_matrix(labels, y_hat, normalize='all')
    cm_true = metrics.confusion_matrix(labels, y_hat, normalize='true')

    # Set figure
    fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(16,4) )

    # Plot quantity distribution confusion matrix
    sns.heatmap(cm, annot=True, fmt='.0%', cmap="BuPu", square=True, ax=ax1);
    ax1.set(title='Distribution of Predictions',ylabel='True Class', xlabel='Predicted Class')

    # Plot normalized matrix (to ROWS, true values)
    sns.heatmap(cm_true, annot=True, fmt='.0%', cmap="BuPu", square=True, ax=ax2);
    ax2.set(title='Normalized to True Class',ylabel='True Class', xlabel='Predicted Class')

    # Calculate 'false-positive rate', 'true-positive rate' and 'area under curve' (AUC)
    fpr , tpr , thresholds = metrics.roc_curve(labels, y_hat)
    auc_score = metrics.roc_auc_score(labels, y_hat)

    ax3.plot(fpr,tpr, color='indigo'); 
    ax3.axis([0,1,0,1]) 
    ax3.set(xlabel ='False Positive Rate', ylabel ='True Positive Rate', 
            title = f"ROC Curve, AUC = {auc_score:.2f}")
    #ax3.title(f"ROC Curve, AUC: {round(auc_score), 3}", size=15)
    ax3.grid(color='whitesmoke', zorder=0)
    plt.show()

    # If selected, return predicted labels
    if return_preds == True:
        return y_hat



def plot_feature_importance():