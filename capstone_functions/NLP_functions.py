""" This file contains functions to for NLP and modeling for my capstone project. 01/14/2021 """

from numpy.core.overrides import ArgSpec
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer, FunctionTransformer
from sklearn.metrics import auc, accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import unidecode as ud
import nltk
from PIL import Image
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer



def preprocess_review(text, pattern=None, stopwords=None):
    
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

    # Lemmatise 
    lemmatiser = WordNetLemmatizer()
    lemmas = [lemmatiser.lemmatize(token, pos='v') for token in tokens]
    
    return lemmas



def preprocess_review(text, pattern=None, stopwords=False):
    
    # Tokenise words while ignoring punctuation
    tokeniser = RegexpTokenizer(r'\w+')
    tokens = tokeniser.tokenize(text)
    
    # Lowercase and lemmatise 
    lemmatiser = WordNetLemmatizer()
    lemmas = [lemmatiser.lemmatize(token.lower(), pos='v') for token in tokens]
    
    if stopwords:
        # Remove stop words
        keywords = [lemma for lemma in lemmas if lemma not in stopwords]
        return keywords

    else:    
        return lemmas



def plot_word_frequencies(df, n=10, pattern=None, stopwords=None):
    
    fig, axes = plt.subplots(3, 2, figsize=(25, 30))

    for rating, ax in zip(list(range(6, 0, -1)), axes.flatten()):

        if 1 <= rating - 1 <= 5:
            
            reviews = df.loc[df['Rating'] == (rating - 1)]['Review']

            # make corporus
            text = " ".join(review for review in reviews)

            review_tokens = preprocess_review(text, pattern, stopwords)

            freq_n = nltk.FreqDist(review_tokens)

            freq_x = [i[0] for i in freq_n.most_common(n)]
            freq_y = [i[1] for i in freq_n.most_common(n)]

            ax.bar(freq_x, freq_y)
            ax.set_title(f"{rating-1} Star Reviews", size=25, fontweight='bold')
            ax.set_ylabel("Frequency")
            ax.tick_params(axis='x', labelrotation = -35)
            
        else:
            ax.axis('off')




def make_cloud(df, rating=int, pattern=None, stopwords=None, mask=None):
    
    if 1 <= rating <= 5:
        
        reviews = df.loc[df['Rating'] == rating]['Review']
        text = " ".join(review for review in reviews)

    else:
        
        reviews = df['Review']
        text = " ".join(review for review in reviews)
    
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

    plt.figure(figsize=(20,10))
    plt.imshow(wordcloud, interpolation='bilinear', aspect='auto') 
    plt.axis("off");



def plot_clouds_per_rating(df, pattern=None, stopwords=None, mask=None):

    fig, axes = plt.subplots(3, 2, figsize=(25, 20))
    plt.tight_layout()

    for i, ax in zip(list(range(6, 0, -1)), axes.flatten()):

        if 1 <= i-1 <= 5:
            ax.imshow(make_cloud(df, 
                                rating=(i-1), 
                                pattern=pattern, 
                                stopwords=stopwords,
                                mask=mask), 
                                interpolation="bilinear")
            ax.axis("off")
            ax.set_title(f"{i-1} Star Reviews", size=25, fontweight='bold')

        else:
            ax.axis('off')



def transform_format(val):
    if val == 0:
        return 255
    else:
        return val



def make_cloud_mask(image_path):
    mask_array = np.array(Image.open(image_path))
    
    transformed_mask = np.ndarray((mask_array.shape[0], mask_array.shape[1]), np.int32)
    
    for i in range(len(mask_array)):
        transformed_mask[i] = list(map(transform_format, mask_array[i]))
        
    return transformed_mask



def multiclass_roc_auc_score(y_test, y_pred, average="macro"):

    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)


def cross_val_model(model, X, y, cv=5, scoring='recall_weighted'):
    """"weighted" accounts for class imbalance by computing the average of binary 
    metrics in which each class’s score is weighted by its presence in the true data sample."""
    cv_scores = cross_val_score(model, X, y, cv=cv, n_jobs=1, scoring=scoring)
    print(cv_scores)
    print("Recall: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))


def evaluate_model(model, X, labels, return_preds=False, norm_type='true'):
    """ Given model and predictors/labels, returns model performance evaluation
        as Classification Report Table and two confusion matrices (one with 
        prediction class distribution and one normalized (default: 'true').
        Also returns a ROC curve with AUC measurement.

        Optional: return predicted labels."""

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

    # Print prediction breakdown
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

    # Plot normalized matrix (to ROWS, true values)
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



def print_explainer(explainer, model, X_train, y_train, y_2_train, n=10, idx=0):
    exp = explainer.explain_instance(X_train.iloc[idx], model.predict_proba, num_features=n)
    print('Review id: %d' % idx)
    print('User Rating:', y_train.iloc[idx])
    print('True class: %s' % y_2_train.iloc[idx])
    print('Probability of Flag =', model.predict_proba([X_train.iloc[idx]])[0,0])
    print('Probability of Pass =', model.predict_proba([X_train.iloc[idx]])[0,1])
    print("Prediction: Correct :)" if (y_2_train.iloc[idx] == model.predict([X_train.iloc[idx]])[0]) 
          else "Prediction: Incorrect :(")
    return exp







def plot_feature_importance():
    pass