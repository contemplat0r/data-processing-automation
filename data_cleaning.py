import numpy as np
import pandas as pd

import collections

column_description = collections.namedtuple('column_description', ['dtype', 'value_counts', 'nans_count', 'problem', 'decision'])



def get_df_extended_info(df):

    df_shape = df.shape
    rows_num = df_shape[0]
    columns = df.columns
    df_dtypes = df.dtypes
    
    columns_info = {}
    for column in columns.tolist():
        column_series = df[column]
        print("\nvalues_count len:", len(column_series.value_counts()), '\n')
        columns_info[column] = column_description(
            column_series.dtype,
            column_series.value_counts(),
            len(column_series[column_series.isna()]),
            '',
            ''
        )
     
    return df_shape, columns_info

def detect_problems_column(df_extended_info):
    df_shape, columns_info = df_extended_info
    for column_name, description in columns_info.items():
        if len(description.value_counts) == 1:
            columns_info[column_name] = description._replace(problem='one_value')
        if description.nans_count > 0:
            columns_info[column_name] = description._replace(problem='isna')
    return df_shape, columns_info


if __name__ == '__main__':

    df = pd.read_csv('train_users_2.csv')

    info = get_df_extended_info(df)
    print("shape")
    print(info[0])
    print('\ninfo')
    #print(info[1])
    '''
    for column_name, column_info in info[1].items():
        print('\n', column_name, '\n')
        print(column_info)
        print('\n' * 2)
    '''
    detect_problems_info = detect_problems_column(info)
    for column_name, column_info in detect_problems_info[1].items():
        print('\n', column_name, '\n')
        print(column_info)
        print('\n' * 2)

