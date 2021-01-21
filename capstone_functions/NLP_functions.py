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
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer


#https://gist.github.com/jiahao87/d57a2535c2ed7315390920ea9296d79f

def process_review(review):
    
    review = review.lower()
    # tokens = nltk.word_tokenize(review)
    # stopwords_removed = [token.lower() for token in tokens if token.lower() not in stopwords_list]
    
    # remove special character
    review = re.sub("([^\x00-\x7F])+"," ", review)

    # Convert accents/convert utf-8 to normal text
    review = ud.unidecode(review)

    return review

# map to data (data is form list of articles as strings...)
#processed_data = list(map(process_review, data))

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



def make_transformer(function, active=True):
    
    """Maps a singular function to series and returns a series compatibile with sklearn pipelines. 
    A hacky way to bypass building tranformer via OOP and inheritance. Works for now!
    Source: https://ryan-cranfill.github.io/sentiment-pipeline-sklearn-3/ """
    
    def map_function_to_list(list_or_series, active=True):
        
        if active:
            
            return [function(i) for i in list_or_series]
        
        else: # if it's not active, just pass it right back
            
            return list_or_series
    
    return FunctionTransformer(map_function_to_list, validate=False, kw_args={'active':active})




def make_cloud(df, rating=int, pattern=None, stopwords=None,width=1600, height=800):
    
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
                          stopwords=stopwords).generate(text)
    
    return wordcloud


def plot_cloud(wordcloud):

    plt.figure(figsize=(20,10))
    plt.imshow(wordcloud, interpolation='bilinear', aspect='auto') 
    plt.axis("off");



def plot_clouds_per_rating(df, pattern=None, stopwords=None):

    fig, axes = plt.subplots(3, 2, figsize=(25, 20))
    plt.tight_layout()

    for i, ax in zip(list(range(6, 0, -1)), axes.flatten()):

        if 1 <= i-1 <= 5:
            ax.imshow(make_cloud(df, 
                                rating=(i-1), 
                                pattern=pattern, 
                                stopwords=stopwords), 
                                interpolation="bilinear")
            ax.axis("off")
            ax.set_title(f"{i-1} Star Reviews", size=25, fontweight='bold')

        else:
            ax.axis('off')



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



def plot_feature_importance():
    pass