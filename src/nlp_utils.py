import re


def preprocessor(text):
    """
    text preprocessor
    :param text: an input string
    :return:
    """
    # remove everything between '<' and '>', can be used to remove HTML tags
    # text = re.sub('<[^>]*>', '', text)
    # temporarily store all emoticons
    emoticons = re.findall('[:;=](?:-)?[)(DP|/\\]', text)
    # lower case, remove all non-word characters, add emoticons to the end of the string, remove the "nose" '-'
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    # remove non-English unicode characters (not from the ASCII character set)
    text = re.sub('[^\x00-\x7F]', '', text)
    # strip leading and trailing whitespaces, replace multiple spaces with a single
    text = re.sub('\s+', ' ', text)
    return text

# from nltk.tokenize import word_tokenize
# tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet) # remove URLs
# tweet = word_tokenize(tweet) # remove repeated characters (helloooooooo into hello)
