import numpy as np
import pandas as pd

import collections

column_description = collections.namedtuple(
        'column_description',
        #['name', 'dtype', 'value_counts', 'nans_count', 'problem', 'way_fix_problem']
        ['name', 'num_of_rows', 'dtype', 'value_counts', 'nans_count', 'problem', 'fix_problem_way']
    )


def get_df_extended_info(df):

    df_shape = df.shape
    columns = df.columns
    
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


    
def create_column_description(column_series):
    return column_description(
            column_series.name,
            column_series.shape[0],
            column_series.dtype,
            column_series.value_counts(),
            len(column_series[column_series.isna()]),
            '',
            ''
        )

def detect_column_problems(column_description):
    if len(column_description.value_counts) == 1:
        column_description = column_description._replace(problem='one_value')
    if column_description.nans_count > 0:
        column_description = column_description._replace(problem='isna')
    return column_description

def select_problem_fix_way(column_description):
    return

def data_clean_processor(df, column_processors):

    columns = df.columns
    columns_description = {}
    for column in columns.tolist():
        column_description = create_column_description(df[column])
        for processor in column_processors:
            column_description = processor(column_description)
        columns_description[column] = column_description

    return {'df_shape': df.shape, 'columns_description': columns_description}


if __name__ == '__main__':

    df = pd.read_csv('train_users_2.csv')

    '''
    info = get_df_extended_info(df)
    print("shape")
    print(info[0])
    print('\ninfo')
    #print(info[1])
    '''

    '''
    for column_name, column_info in info[1].items():
        print('\n', column_name, '\n')
        print(column_info)
        print('\n' * 2)
    '''

    '''
    detect_problems_info = detect_problems_column(info)
    for column_name, column_info in detect_problems_info[1].items():
        print('\n', column_name, '\n')
        print(column_info)
        print('\n' * 2)
    '''

    data_description = data_clean_processor(df, [detect_column_problems])
    print(data_description['df_shape'])
    
    columns_description = data_description['columns_description']

    for column_name, column_description in columns_description.items():
        print(column_name)
        print(column_description)

