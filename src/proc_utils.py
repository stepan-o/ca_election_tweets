from collections import Iterable

import matplotlib.pyplot as plt
import pandas as pd


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
            key = col
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
            'color': 'darkred',
            'weight': 'normal',
            'size': 16}

    ax.set_title("Results of the duplicate check", fontdict=font)
    ax.set_ylabel("Subset used as detection criteria", fontdict=font)
    ax.set_xlabel("Number of duplicate records", fontdict=font)

    ax.tick_params(axis='both', labelsize=14)

    return dup_results_df
