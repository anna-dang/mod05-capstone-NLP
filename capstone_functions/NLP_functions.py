""" This file contains functions to for NLP and modeling for my capstone project. 01/14/2021 """

import numpy as np
import pandas as pd
import seaborn as sns
import unidecode as ud
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from PIL import Image

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer, FunctionTransformer
from sklearn.metrics import auc, accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.utils import class_weight

import nltk
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer



def preprocess_review(text, pattern=None, stopwords=None):
    """Lowercase and tokenize a given text via WordNet or Regex (if pattern provided).
    Removes stopwords if provided. Removes digits. Lemmatizes tokens to their roots.

    Args:
        text (str): Text to be processed
        pattern (str, optional): ReGex pattern to replace. Defaults to None.
        stopwords (list of str, optional): List of words to remove from text.
                                             Defaults to None.

    Returns:
        [list of str]: List of tokens processed according to arguments.
    """
    # Lowercase and tokenize
    if pattern:
        
        tokeniser = RegexpTokenizer(pattern)
        tokens = tokeniser.tokenize(text.lower())
        
    else:
        
        tokens = word_tokenize(text.lower())
    
    if stopwords:
        
        tokens = [w for w in tokens if w not in stopwords]

    # Remove numbers
    tokens = [c for c in tokens if not c.isdigit()]

    # Lemmatize 
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token, pos='v') for token in tokens]
    
    return lemmas



def plot_word_frequencies(df, n=10, pattern=None, stopwords=None):
    """From a given DataFrame, create a unique coprus per rating. Tokenize each of the 
    per rating corpora and plot the 'n' most frequent words for each ratings (1 - 5).

    Args:
        df (DataFrame): Must be columns ['Review', 'Rating'] where 'Review' is a str and
                        'Rating' is a corresponding int label 1 (worst) - 5 (best).
        n (int, optional): Number of words to return. Defaults to 10.
        pattern (str, optional): ReGex pattern for tokenizing. Defaults to None.
        stopwords (list of str, optional): Words to remove from corpus during tokenizing. 
                                            Defaults to None.
    """
    fig, axes = plt.subplots(3, 2, figsize=(25, 30))

    for rating, ax in zip(list(range(6, 0, -1)), axes.flatten()):

        if 1 <= rating - 1 <= 5:
            
            reviews = df.loc[df['Rating'] == (rating - 1)]['Review']

            # Make corporus
            text = " ".join(review for review in reviews)

            # Tokenize
            review_tokens = preprocess_review(text, pattern, stopwords)

            # Get freq.
            freq_n = nltk.FreqDist(review_tokens)

            freq_x = [i[0] for i in freq_n.most_common(n)]
            freq_y = [i[1] for i in freq_n.most_common(n)]

            # Plot word frequency
            ax.barh(freq_x, freq_y, color = 'grey')
            ax.set_title(f"{rating-1} Star Reviews", size=25, fontweight='bold')
            plt.xlabel('Occurances', size=10)
            plt.ylabel("Word", size=10)
            ax.set_ylabel("Frequency")
            ax.tick_params(axis='x', labelrotation = 0)
            
        else:

            # Leave 6th axis blank
            ax.axis('off')




def make_cloud(df, rating=int, pattern=None, stopwords=None, mask=None):
    """Generate a word cloud of most frequent words for specified rating corpus. Defaults 
    to generating for entire corpus (all classes).

    Args:
        df (DataFrame): Must be columns ['Review', 'Rating'] where 'Review' is a str and
                        'Rating' is a corresponding int label 1 (worst) - 5 (best).
        rating (int, optional): Which class to plot, 1 - 5. 
                                Defaults to int, resulting in full corpus cloud.
        pattern (str, optional): ReGex pattern for tokenizing. Defaults to None.
        stopwords (list of str, optional): Words to remove from corpus during tokenizing. 
                                            Defaults to None.
        mask (np.array, optional): Outline shape desired for cloud. Defaults to rectangle.

    Returns:
        wordcloud object: image information to plot a word cloud
    """

    # If given rating in range, generate cloud for that rating corpus
    if 1 <= rating <= 5:
        
        reviews = df.loc[df['Rating'] == rating]['Review']
        text = " ".join(review for review in reviews)

    # If no rating/outside range, generate cloud for entire corpus
    else:
        
        reviews = df['Review']
        text = " ".join(review for review in reviews)
    
    # Generate cloud
    wordcloud = WordCloud(random_state=619, colormap='bone_r', 
                          collocations=False, regexp=pattern,
                          width=1600, height=800,
                          min_font_size= 20,
                          font_path="./driver/LEMONMILK-Regular.otf",
                          background_color="white",
                          stopwords=stopwords,
                          mask=mask).generate(text)
    
    return wordcloud


def plot_cloud(wordcloud):
    """Plots a given word cloud via image display.

    Args:
        wordcloud w(ordcloud object) : image information to plot a word cloud
    """
    plt.figure(figsize=(20,10))
    plt.imshow(wordcloud, interpolation='bilinear', aspect='auto') 
    plt.axis("off");



def plot_clouds_per_rating(df, pattern=None, stopwords=None, mask=None):
    """Generate a word cloud of most the frequent words for each rating corpus, plotted
    individually.

    Args:
        df (DataFrame): Must be columns ['Review', 'Rating'] where 'Review' is a str and
                        'Rating' is a corresponding int label 1 (worst) - 5 (best).
        rating (int, optional): Which class to plot, 1 - 5. 
                                Defaults to int, resulting in full corpus cloud.
        pattern (str, optional): ReGex pattern for tokenizing. Defaults to None.
        stopwords (list of str, optional): Words to remove from corpus during tokenizing. 
                                            Defaults to None.
        mask (np.array, optional): Outline shape desired for cloud. Defaults to rectangle.
    """
    fig, axes = plt.subplots(3, 2, figsize=(25, 20))
    plt.tight_layout()

    for i, ax in zip(list(range(6, 0, -1)), axes.flatten()):

        if 1 <= i-1 <= 5:

            # Generate and plot clouds
            ax.imshow(make_cloud(df, 
                                rating=(i-1), 
                                pattern=pattern, 
                                stopwords=stopwords,
                                mask=mask), 
                                interpolation="bilinear")
            ax.axis("off")
            ax.set_title(f"{i-1} Star Reviews", size=25, fontweight='bold')

        else:

            # Sixth axis turned off
            ax.axis('off')



def transform_format(val):
    """Transforms a PNG file edge values to 255.

    Args:
        val (np.array): PNG file represented as RGB values 0 - 255

    Returns:
        np.array: Transformed PNG file.
    """
    if val == 0:
        return 255
    else:
        return val



def make_cloud_mask(image_path):
    """From a PNG file create an outline shape, or "mask", compatible 
    with the make_wordcloud() function.

    Args:
        image_path (str): local path to PNG file to mask

    Returns:
        np.array: Transformed RGB values for border of mask.
    """
    mask_array = np.array(Image.open(image_path))
    
    transformed_mask = np.ndarray((mask_array.shape[0], mask_array.shape[1]), np.int32)
    
    for i in range(len(mask_array)):
        transformed_mask[i] = list(map(transform_format, mask_array[i]))
        
    return transformed_mask



def set_weights(y_train, ret_class=False):
    """Compute class weights for a given target column. Use class weights to
    calculate sample weights for use in Bayesian modeling with Sci-Kit Learn.

    Prints a preview of the generated weights and corresponding classes.

    Args:
        y_train (list or Series): The target/class data for model.
        ret_class (bool, optional): Whether to return class weights also. Defaults to False.

    Returns:
        np.array: Array of sample weights, one per sample in y_train.

    """
    # Set training weights to balance classes
    class_weights = class_weight.compute_class_weight(class_weight = 'balanced', 
                                                    classes = np.unique(y_train), 
                                                    y = y_train)

    c_weights_dict = dict(zip(np.unique(y_train), class_weights))
    print("Class Weights:", len(class_weights), "classes")
    display(c_weights_dict)

    # For MultiNomial Naive Bayes - sample weights are required
    sample_weights = class_weight.compute_sample_weight(c_weights_dict, y = y_train)
    print("Sample Weights:", len(sample_weights), "samples")
    display(sample_weights)

    if ret_class:

        return class_weights, sample_weights
    
    else:

        return sample_weights



def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    """Compute the multi-class 'Area Under Curve' from a ROC Curve by
    binarizing labels and predictions - reformating each class score
    as 'One vs Rest' probability.

    Args:
        y_test (series or list): Test labels
        y_pred (series of list): Model predicted labels
        average (str, optional): Scoring type to build ROC curve. 
                                Defaults to "macro".

    Returns:
        float: calculated AUC
    """
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)



def cross_val_model(model, X, y, cv=5, scoring='recall_weighted'):
    """Computes specified cross validation score for a given model with
    a labeled set of X and y. Performs 'k' number of cross validations.

    Prints the score for each for and a weighed average for all folds.

    'Weighted' accounts for class imbalance by computing the average of binary 
    metrics in which each class’s score is weighted by its presence in the true data sample.
    
    Args:
        model (sklearn model): Trained model.
        X (predictors): pd.DF or Series encoded to model training specifications.
        y (list or Series): Labels for given X.
        cv (int, optional): 'k', number of folds for K-Folds cross validation. Defaults to 5.
        scoring (str, optional): Type of scoring to evaluate folds. Defaults to 'recall_weighted'.
    """
    cv_scores = cross_val_score(model, X, y, cv=cv, n_jobs=1, scoring=scoring)
    print(cv_scores)
    print("Recall: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))



def evaluate_model(model, X, labels, return_preds=False, norm_type='true'):
    """Given model and predictors/labels, returns model performance evaluation.
    
    Args:
        model (sklearn model): Trained model.
        X (predictors): pd.DF or Series encoded to model training specifications.
        labels (list or Series): Labels for given X.
        return_preds (bool, optional): Return the predicted values. Defaults to False.
        norm_type (str, optional): Type of normalization for the confusion matrix. 
                                    Defaults to 'true', normed across rows.

    Prints:
    - Confusion matrix (default: 'normalized true')
    - Classification Report Table 
    - ROC AUC score
    - Percent and count of correct vs incorrect predictions
    - Table of model predictions:   'actual_count': actual number of class in sample 
                                    'pred_count': number of predicted per class by model
                                    'diff' : difference in count (neg = under, pos = over)
                                    '%_actual': percent of class in sample
                                    '%_pred': percent predicted

    Returns:
        y_hat: Optional. Series of predictions of model for X.
    """

    # Get binary predictions by round predicted probabilities
    y_hat = model.predict(X)
   
    # Print classification report
    report = classification_report(labels, y_hat, output_dict=True)
    print("Classification Report: \n")
    print(classification_report(labels, y_hat))

    # Print ratio of predictions
    correct = np.sum(y_hat == labels)
    incorrect = np.sum(y_hat != labels)
    print(f"Correct: {correct}, {round((correct/len(labels)*100), 2)}%") 
    print(f"Incorrect: {incorrect}, {round((incorrect/len(labels)*100), 2)}%")
    
    # ROC AUC score
    roc_auc = multiclass_roc_auc_score(labels, y_hat, average='macro')
    print(f"ROC/AUC: {round(roc_auc, 3)}")

    # Print prediction breakdown table
    classes, pred_counts = np.unique(y_hat, return_counts=True)
    
    actual_counts = []
    for key in classes:
        actual_counts.append(report[str(key)]['support'])
    actual_counts = np.array(actual_counts)
        
    preds = {'actual_count': actual_counts, 
            'pred_count': pred_counts,
            'diff' : pred_counts - np.array(actual_counts),
            '%_actual': np.round(actual_counts/sum(actual_counts) * 100, decimals=2),
            '%_pred': np.round(pred_counts/sum(pred_counts) * 100, decimals=2)}

    pred_df = pd.DataFrame(preds, index=[str(i) for i in classes]).rename_axis('class')
    display(pred_df)

    # Build confusion matrix
    cm_true = confusion_matrix(labels, y_hat, normalize='true')

    # Plot matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm_true, annot=True, fmt='.0%', cmap="bone_r", square=True,
                xticklabels=classes, yticklabels=classes);
    sns.set(font_scale=1.5)
    plt.title('Normalized to True Class')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.ylim(len(classes), 0)
    plt.show()

    # If selected, return predicted labels
    if return_preds == True:
        return y_hat



def print_explainer(explainer, model, X_test, y_test, y_2_test, n=10, idx=0):
    """Explain the feature importance of word tokens from a single text entry for the 
    given text classification model using LIME Tex Explainer. 

    Args:
        explainer (LimeTextExplainer instance): LIME Explainer instantiated with class names
        model (sklearn model): Trained to X_train
        X_test (list or series): Text predictors
        y_test ([type]): Corresponding 5 class labels
        y_2_test ([type]): Assigned binary label
        n (int, optional): Number of tokens to explain. Defaults to 10.
        idx (int, optional): Index number of which entry to explain. Defaults to 0.

    Returns:
        LimeTextExplainer explained instance: Generated from one prediction for 'n' number of 
                                                features explained.
    """
    exp = explainer.explain_instance(X_test.iloc[idx], model.predict_proba, num_features=n)
    print('Review id: %d' % idx)
    print('User Rating:', y_test.iloc[idx])
    print('True class: %s' % y_2_test.iloc[idx])
    print('Probability of Flag =', model.predict_proba([X_test.iloc[idx]])[0,0])
    print('Probability of Pass =', model.predict_proba([X_test.iloc[idx]])[0,1])
    print("Prediction: Correct ✓ " if (y_2_test.iloc[idx] == model.predict([X_test.iloc[idx]])[0]) 
          else "Prediction: Incorrect ✗")

    return exp