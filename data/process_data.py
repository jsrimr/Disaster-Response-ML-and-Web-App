import os
import sys
import pandas as pd


def load_data(messages_filepath, categories_filepath):
    """
    Load the messages and categories datasets and merge them into a dataframe
    INPUT: messages_filepath (.csv file), catgeories_filepath (.csv file)
    OUTPUT: a merged df
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id")

    return df


def clean_data(df):
    """
    Split the categories into 36 columns each represents a category.
    Each meassage receives a value of 1 for the category its belong to, and 0 for others
    INPUT: merged df
    TASK:
        1. Split categories into seperate category columns
        2. Convert category values to just numbers 0 or 1
        3. Replace categories column in df with new category columns
        4. Remove duplicates
    OUTPUT: a dataframe in which each unique meassage is labeled with a category
    """
    categories = df["categories"].str.split(";", expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
        
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        categories[column] = categories[column].apply(lambda x: 1 if x > 0 else 0)

    df = df.drop(['categories','original'], axis=1)
    df = pd.concat([df, categories], axis=1)

    df.drop_duplicates(inplace=True)

    return df

from sqlalchemy import create_engine
def save_data(df, database_filename):
    engine = create_engine(f'sqlite:///{database_filename}')
    table_name = os.path.basename(database_filename).split('.')[0]
    df.to_sql(table_name, engine, index=False,if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()