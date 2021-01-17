""" This file contains functions to for NLP and modeling for my capstone project. 01/14/2021 """

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import auc, accuracy_score, confusion_matrix, classification_report, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud

# # from medium
# def preprocess_text(text):
#     # Tokenise words while ignoring punctuation
#     tokeniser = RegexpTokenizer(r'\w+')
#     tokens = tokeniser.tokenize(text)
    
#     # Lowercase and lemmatise 
#     lemmatiser = WordNetLemmatizer()
#     lemmas = [lemmatiser.lemmatize(token.lower(), pos='v') for token in tokens]
    
#     # Remove stop words
#     keywords= [lemma for lemma in lemmas if lemma not in stopwords.words('english')]
#     return keywords
# # Create an instance of TfidfVectorizer
# vectoriser = TfidfVectorizer(analyzer=preprocess_text)
# # Fit to the data and transform to feature matrix
# X_train_tfidf = vectoriser.fit_transform(X_train)
# X_train_tfidf.shape



def process_review(review):
    pass
    # tokens = nltk.word_tokenize(review)
    # stopwords_removed = [token.lower() for token in tokens if token.lower() not in stopwords_list]
    # return stopwords_removed

# map to data (data is form list of articles as strings...)
#processed_data = list(map(process_review, data))



def make_cloud(df, rating=int, pattern=None, stopwords=None):
    
    if 1 <= rating <= 5:
        
        reviews = df.loc[df['Rating'] == rating]['Review']
        text = " ".join(review for review in reviews)

    else:
        
        reviews = df['Review']
        text = " ".join(review for review in reviews)
    
    wordcloud = WordCloud(random_state=619, colormap='plasma', 
                          collocations=False, regexp=pattern,
                          width=400, height=300,
                          font_path="./driver/tommy_font.otf",
                          background_color="white",
                          stopwords=stopwords).generate(text)
    
    return wordcloud



def plot_cloud(wordcloud):
    plt.figure(figsize=(15,10))
    plt.imshow(wordcloud, interpolation='bilinear', aspect='auto') 
    plt.axis("off");



def plot_clouds_per_rating(df, pattern=None, stopwords=None):

    fig, axes = plt.subplots(3, 2, figsize=(25, 30))
    fig.tight_layout(pad=0.2, w_pad=0.5, h_pad=1.2)

    for i, ax in zip(list(range(6, 0, -1)), axes.flatten()):

        if 1 <= i-1 <= 5:
            ax.imshow(make_cloud(df, rating=(i-1), pattern=pattern, stopwords=stopwords), 
                                interpolation="bilinear")
            ax.axis("off")
            ax.set_title(f"- - - {i-1} Stars - - -", size=25, fontweight='bold')

        else:
            ax.axis('off')



def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)



def evaluate_model(model, X, labels, return_preds=False, norm_type='true'):
    """ Given model and predictors/labels, returns model performance evaluation
        as Classification Report Table and two confusion matrices (one with 
        prediction class distribution and one normalized (default: 'true').
        Also returns a ROC curve with AUC measurement.

        Optional: return predicted labels."""

    # Get binary predictions by round predicted probabilities
    y_hat = model.predict(X)
   
    # Print classification report
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

    # Build confusion matrix
    cm = confusion_matrix(labels, y_hat, normalize='all')
    cm_true = confusion_matrix(labels, y_hat, normalize='true')

    # Set figure
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(25, 10))

    # Plot quantity distribution confusion matrix
    sns.heatmap(cm, annot=True, fmt='.0%', cmap="BuPu", ax=ax1);
    ax1.set(title='Distribution of Predictions',ylabel='True Class', xlabel='Predicted Class')
    ax1.set_ylim(5.0, 0)

    # Plot normalized matrix (to ROWS, true values)
    sns.heatmap(cm_true, annot=True, fmt='.0%', cmap="BuPu", ax=ax2);
    ax2.set(title='Normalized to True Class',ylabel='True Class', xlabel='Predicted Class')
    ax2.set_ylim(5.0, 0)
    
    plt.show()

    # If selected, return predicted labels
    if return_preds == True:
        return y_hat



def plot_feature_importance():
    pass