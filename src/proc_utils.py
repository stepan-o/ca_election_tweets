from collections import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def string_concat(ser, string_name="", display_sym=500,
                  input_type='strings'):
    """
    a function to concatenate a single string
    out of a series of text documents
    """
    con_string = []

    if input_type == 'strings':
        for text in ser:
            con_string.append(text)

    elif input_type == 'lists':
        for str_list in ser:
            con_string.append(" ".join(str_list))
    else:
        print("'input_type' must be 'strings' or 'lists'.")

    con_string = pd.Series(con_string).str.cat(sep=' ')

    print("First {0} symbols in the {1} string:\n"
          .format(display_sym, string_name))
    print(con_string[:display_sym])

    return con_string


def duplicate_check(df, subsets_to_check: dict = None, cols_to_drop=None):
    """
     function to perform the duplicate check

     takes as input a DataFrame and one (or both) of the optional parameters:

     'subsets_to_check' is a dict that will be used for duplicate check criteria
     its keys represent the name of the subset of columns and its values represent
     the lists of columns to be used as criteria

     'subset_to_drop' is a list of columns to be excluded one by one from criteria
     to perform duplicate checks without these columns
     if 'subset_to_drop' is empty, this step is skipped

    :param df: pd.DataFrame -- DataFrame on which to perform the duplicate check

    :param subsets_to_check: dict -- dict, where keys are column subset names and
                                     values are subsets of columns to use

    :param cols_to_drop: Iterable -- an iterable containing names of columns to be
                                     excluded one by one from detection criteria

    :return dup_result_df: pd.DataFrame -- DataFrame with results of the duplicate check
    """

    def perform_duplicate_check(result_key, col_subset):
        """
        function to perform the actual duplicate check
        and record its results

        :param result_key: string -- key to be used to record results of the check
        :param col_subset: list -- subset that will be used to perform the check

        :return: None, updates nonlocal dict 'dup_results'
        """
        # variables from the outer scope
        nonlocal df, dup_results
        # create a new entry in results dictionary 'dup_results'
        dup_results[result_key] = dict()
        # determine the number of duplicates
        dup_results[result_key]['num_duplicates'] = df.duplicated(subset=col_subset).sum()
        dup_results[result_key]['num_total'] = len(df)
        dup_results[result_key]['percentage'] = dup_results[result_key]['num_duplicates'] \
                                                / dup_results[result_key]['num_total'] * 100
        print("Subset '{0}': {1:,} ({2:.2f}% of total {3:,}) records are detected as duplicated."
              .format(result_key,
                      dup_results[result_key]['num_duplicates'],
                      dup_results[result_key]['percentage'],
                      dup_results[result_key]['num_total']))

    # create a dict to store results of the duplicate check
    dup_results = dict()

    # duplicate check using all columns as criteria
    key = 'all_columns'
    subset = df.columns
    perform_duplicate_check(key, subset)

    # duplicate check using 'cols_to_drop' -- each test takes all columns minus one
    if isinstance(cols_to_drop, Iterable):
        for col in cols_to_drop:
            subset = df.columns
            subset = subset.drop(col)
            perform_duplicate_check(col, subset)

    if isinstance(subsets_to_check, dict):
        for k, v in subsets_to_check.items():
            perform_duplicate_check(k, v)

    dup_results_df = pd.DataFrame(dup_results)

    # plot results of the check
    # create figure and axis
    f, ax = plt.subplots(1, figsize=(6, 6))

    # plot results
    dup_results_df.T['num_duplicates'] \
        .sort_values() \
        .plot(kind='barh', color='gray')

    # set axis parameters
    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 16}

    ax.set_title("Results of the duplicate check", fontdict=font)
    ax.set_ylabel("Subset used as detection criteria", fontdict=font)
    ax.set_xlabel("Number of duplicate records", fontdict=font)

    ax.tick_params(axis='both', labelsize=14)

    return dup_results_df


def tfm_2class(df, label_col, label_vals, text_col,
               class_names=('neg_tf', 'pos_tf'),
               stop_words='english',
               min_df=0.01, max_df=0.9,
               return_type='tfm'):
    """
    a function to create a Term Frequency Matrix
    from the corpus of documents found in
    column 'text_col' of DataFrame 'df'

    this function is designed to work with 2 classes
    of target variable found in column 'label_col' of 'df',
    values of classes (e.g., -1, 1)
    need to be supplied as a list in parameter 'label_vals'

    class names for the return DataFrame
    can be specified via parameter 'class_names'

    :param df: pandas.DataFrame
    DataFrame that contains the corpus to be summarized

    :param text_col: string
    the name of the column in df containing corpus of documents to be summarized

    :param label_col: string
    name of the column in 'df' containing label (target) information

    :param label_vals: list
    list of values that 'label_col' can take (e.g., [1, -1]) only 2 values supported

    :param class_names: tuple (string, string)
    names of target classes
    default=('neg_tf', 'pos_tf')

    :param min_df
    min document frequency used by CountVectorizer to filter tokens when creating vocabulary of the corpus
    default=0.01

    :param max_df
    max document frequency used by CountVectorizer to filter tokens when creating vocabulary of the corpus
    default=0.9

    :param stop_words: string
    stop words to by used by CountVectorizer

    :param return_type: string
    type of return:
        'tfm': for Term Frequency Matrix, a DataFrame with the term frequency matrix of the corpus is returned
        'dtr': for difference / total ratio, a float representing the ration of sum of all absolute differences for
               each token (how much more occurrences class 1 has vs class 2 for this token),
               divided by the total number of occurrences by all tokens

    :returns: tfm_df: pandas.DataFrame
    term frequency matrix of the corpus
    """
    # initialize CountVectorizer from Scikit-learn
    vectorizer = CountVectorizer(strip_accents='unicode',
                                 stop_words=stop_words,
                                 max_df=max_df,
                                 min_df=min_df)

    # fit vectorizer to corpus in 'text_col' of 'df'
    vectors_f = vectorizer.fit(df[text_col])

    # create a subset of 'df' with all records of class 1
    class1_subset = df \
        .loc[df[label_col] == label_vals[0], text_col]
    # vectorize subset into a sparse matrix
    class1_doc_matrix = vectors_f \
        .transform(class1_subset)

    # create a subset of 'df' with all records of class 2
    class2_subset = df \
        .loc[df[label_col] == label_vals[1], text_col]
    # vectorize subset into a sparse matrix
    class2_doc_matrix = vectors_f \
        .transform(class2_subset)

    # sum occurrences of each token
    class1_tf = np.sum(class1_doc_matrix, axis=0)
    class2_tf = np.sum(class2_doc_matrix, axis=0)

    # remove single-dimensional entries from the shape of the arrays
    class1 = np.squeeze(np.asarray(class1_tf))
    class2 = np.squeeze(np.asarray(class2_tf))

    # create a DataFrame with token frequencies by class
    tfm_df = pd.DataFrame([class1, class2],
                          columns=vectors_f.get_feature_names()) \
        .transpose()

    # change column names
    tfm_df.columns = class_names

    # create a new column with total token frequency
    tfm_df['total'] = tfm_df[class_names[0]] \
                      + tfm_df[class_names[1]]

    # create a new column with difference between classes
    tfm_df['abs_diff'] = \
        abs(tfm_df[class_names[0]] \
            - tfm_df[class_names[1]])

    if return_type == 'tfm':
        # return Term Frequency Matrix for the corpus
        return tfm_df

    elif return_type == 'dtr':
        # return sum of abs of all diff by total sum
        return tfm_df['abs_diff'].sum() / tfm_df['total'].sum()
    else:
        print("'return_type' must be either 'tfm' " +
              "for Term Frequency Matrix")
        print("or 'dtr' for AbsDiff / Total ratio.")
