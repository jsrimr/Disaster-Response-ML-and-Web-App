import sys
import os
import re
import joblib
import pandas as pd
from sqlalchemy import create_engine

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
nltk.download(['punkt', 'wordnet', 'stopwords'])

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report


from sqlalchemy import create_engine


def load_data(database_filepath):
    """
    Load the clean data from the SQL database
    Args:
    database_filepath
    Returns:
    X: features dataframe
    Y: target dataframe
    names : Name of classified classes
    """
    
    engine = create_engine(f'sqlite:///{database_filepath}')
    table_name = os.path.basename(database_filepath).split('.')[0]
    df = pd.read_sql_table(table_name,con=engine)
    
    X = df['message']
    Y = df.iloc[:, 3:]
    names = Y.columns.tolist()

    return X, Y, names


def tokenize(text):
    # check if there are urls within the text
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex,text)
    for url in detected_urls:
        text = text.replace(url,"urlplaceholder")
    
    # remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]"," ",text)
    
    # tokenize the text
    tokens = word_tokenize(text)
    
    # remove stop words
    tokens = [tok for tok in tokens if tok not in stopwords.words("english")]
    
    # lemmatization
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


import os
def build_model(model_filepath="classifier.pkl", X_train=None, Y_train=None):
    """
    Build and train an end-to-end ML pipeline including feature transformation
    and estimation.
    
    Args:
        model_filepath: If the filepath exists, this function loads the model in the filepath. 
                        Otherwise, build the model from scratch, then train, and return the model.
        X_train: train input data
        Y_train: train label data
    Returns:
        pipeline: end-to-end ML pipeline
    """
    if os.path.exists(model_filepath):
        print("loaded model")
        pipeline = joblib.load(model_filepath)

    else:
        pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
        params = {'clf__estimator__n_estimators':[100,200],'clf__estimator__max_depth':[5]}
        pipeline = GridSearchCV(pipeline, param_grid=params, cv=3, verbose=3)

        print('Training model...')
        pipeline.fit(X_train, Y_train)

    return pipeline



def evaluate_model(model, X_test, Y_test, category_names):
    """
    Compute F1 score and Accuracy on test_data given model
    INPUT: 
        model : estimator
        X_test : test input data
        Y_test : test label data
        category_names : test label name
    OUTPUT: Stdout - F1 score and Accuracy per label(class)
    """
    # predict
    y_pred = model.predict(X_test)

    # classification report
    print(classification_report(Y_test.values, y_pred, target_names=category_names))

    # accuracy score
    accuracy = (y_pred == Y_test.values).mean()
    print('The model accuracy score is {:.3f}'.format(accuracy))



def save_model(model, model_filepath="classifier.pkl"):
    """
    Save the model as a .pkl file
    Args:
        model:          model to be saved
        model_filepath: path the model is saved to
    """
    joblib.dump(model, model_filepath)
    return


def main():

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model("classifier.pkl", X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
