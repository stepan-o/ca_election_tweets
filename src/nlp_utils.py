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
    emoticons = re.findall('[:;=](?:-)?[)(DP|]', text)
    # lower case, remove all non-word characters, add emoticons to the end of the string, remove the "nose" '-'
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    return text
