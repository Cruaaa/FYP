import pandas as pd
import numpy as np
import contractions
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from gensim.parsing.preprocessing import STOPWORDS
import unicodedata

# download current version of NLTK stopwords list
#nltk.download('stopwords')

# reading in of file
dataset = pd.read_csv("tripadvisor_hotel_reviews.csv")

# removal of rows with a null value
dataset = dataset.dropna()

# main function to iterate through and preprocess entire dataset
def main():
    for index, row in dataset.iterrows():
        current_review = row['review']

        current_review = preprocess(current_review)

        dataset.loc[index, 'review'] = current_review

    # print(dataset)

    dataset.to_csv('processed_tripadvisor_dataset.csv', index=False)


current_string = "LoWerCAsE haven't can't won't  Url                       = http://youtube.com #Hashtag 456 @Username this sentence, it!  has. some. <br/> \npunctuation&!                        emoji time 😂 💀🚣"


def preprocess(string):
    print("starting string : " + string)

    # lowercase removal
    string = lowercase_change(string)

    print("lowercase removed: " + string)

    # expand contractions, also removes unnecessary whitespace
    string = expand_contractions(string)

    print("contractions expanded: " + string)

    # removal of emojis
    string = emoji_removal(string)

    print("emojis removed: " + string)

    # removal of URL, hashtag and username
    string = web_value_removal(string)

    print("web values removed: " + string)

    # removal of escape sequences
    string = remove_escape_sequences(string)

    print("scape sequence removal" + string)

    # stopword removal
    string = remove_stopwords(string)

    print("stopword removal: " + str(string))

    # stemming
    string = stem_review(string)

    print("stemming processed: " + str(string))

    # removing punctuation and numbers
    string = punctuation_and_num_removal(string)

    print("punctuation and number removal: " + string)

    # removing additional whitespace
    string = remove_whitespace(string)

    print("Whitespace removed: " + string)

    print("preprocessed string: \n " + str(string))

    return string


# changing all uppercase characters to lowercase
def lowercase_change(string):
    # use of built-in .lower() function
    string = string.lower()

    # print("Uppercases to lowercases: " + string)

    return string


# expansion of any contractions
def expand_contractions(string):
    # empty list to store contractions after expansion
    expanded_words = []
    for word in string.split():
        # using contractions.fix to expand the shortened words
        expanded_words.append(contractions.fix(word))

    # print("expanded contractions: " + string)

    # replacing string with newly modified string
    string = ' '.join(expanded_words)
    return string


# URL, hashtag and username removal
def web_value_removal(string):
    # Remove URLs
    string = re.sub(r'http\S+', '', string)
    # Remove hashtags
    string = re.sub(r'#\w+', '', string)
    # Remove usernames
    string = re.sub(r'@\w+', '', string)
    return string

# remove all punctuation characters and numbers
def punctuation_and_num_removal(string):
    string = string.translate(str.maketrans('', '', '''"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~0123456789'''))
    return string

# removal of any emojis from the string
def emoji_removal(string):
    # list of all values allocated to current and future emojis
    standard_emoji_list = [chr(i) for i in range(0x2600, 0x26FF)]
    # sublist of all non-standard values allocated to very specific emojis
    extended_emoji_list = [chr(i) for i in range(0x1F000, 0x1FFFF)]
    # loop checking if each character is an emoji hex value
    new_string = ''
    for char in string:
        if not any(char in emoji for emoji in standard_emoji_list) and not any(
                char in emoji for emoji in extended_emoji_list):
            new_string += char
    # changing string to modified string
    string = new_string
    return string

# remove of all standard unicode escape sequences
def remove_escape_sequences(string):
    string = string.encode('unicode_escape').decode()

    # print(string)

    return string

# removal of words present in stopword list
def remove_stopwords(string):
    # import of standard NLTK Stopword list and addition of specified words to stopword list
    stopwordlist = STOPWORDS.union(set(['<br/>', 'br/','br']))
    # Breaking string up into tokens
    tokenized_string = word_tokenize(string)
    # Removal of any tokens that match any entry on the stopword list
    tokenized_string = [word for word in tokenized_string if not word in stopwordlist]
    return tokenized_string

# removal of all additional whitespace
def remove_whitespace(string):
    string = ' '.join(string.split())
    return string

# function to stem tokenised words
def stem_review(string):
    # creation of stemmer objecy
    stemmer = PorterStemmer()
    # stemming of all tokens
    stemmed_text = [stemmer.stem(token) for token in string]
    # recomposing stemmed tokens into string object
    stemmed_text = " ".join(stemmed_text)
    return str(stemmed_text)

# standard main call
if __name__ == "__main__":
    preprocess(current_string)

    # print(dataset)

    # main()
