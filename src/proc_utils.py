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
