import os
import sys
from data import Data
from mmd import select_criticism_regularized, greedy_select_protos
import pandas as pd
import numpy as np
from pprint import pprint

print('')

data_directory = 'sleep_stage_data'
data_records_metadata = [
    {
        'file_name': '1106016-1_EEG_features.csv',
        'start_time': 600,
        'end_time': 7770,
    },
    {
        'file_name': '1006251-1_EEG_features.csv',
        'start_time': 900,
        'end_time': 8070,
    },
    {
        'file_name': '1209056-1_EEG_features.csv',
        'start_time': 14400,
        'end_time': 21570,
    },
]

examples_selected = {}

for data_record_metadata in data_records_metadata:
    data_record_file_path = os.path.join(data_directory, data_record_metadata['file_name'])
    data_record = pd.read_csv(data_record_file_path)
    # Filter for Wake and N1 only
    data_record = data_record[data_record['sleep_stage'] <= 1]
    data_record = data_record[data_record['epoch_start_in_seconds'] >= data_record_metadata['start_time']]
    data_record = data_record[data_record['epoch_start_in_seconds'] <= data_record_metadata['end_time']]
    data_record = data_record.reset_index()
    feature_columns = [c for c in data_record if c.startswith('machine_')]
    X = data_record[feature_columns].as_matrix()
    y = data_record['sleep_stage'].as_matrix()
    data = Data()
    data.load_data(X, y, gamma=0.026, docalkernel=False, savefile=None, testfile=None, dobin=False)
    # Calculate global kernel
    data.calculate_kernel()
    # Calculate local kernel
    #data.calculate_kernel_individual()

    num_examples_to_select_per_class = 10
    num_examples_to_select = 40

    num_prototypes_to_select = num_examples_to_select
    prototypes_selected = greedy_select_protos(data.kernel, np.array(range(np.shape(data.kernel)[0])), num_prototypes_to_select)
    prototypes_selected_y = data.y[prototypes_selected]
    prototypes_selected_wake = prototypes_selected[prototypes_selected_y == 0][:num_examples_to_select_per_class]
    prototypes_selected_N1 = prototypes_selected[prototypes_selected_y == 1][:num_examples_to_select_per_class]

    num_criticisms_to_select = num_examples_to_select
    criticisms_selected = select_criticism_regularized(data.kernel, prototypes_selected, num_criticisms_to_select, is_K_sparse=False, reg='logdet')
    criticisms_selected_y = data.y[criticisms_selected]
    criticisms_selected_wake = criticisms_selected[criticisms_selected_y == 0][:num_examples_to_select_per_class]
    criticisms_selected_N1 = criticisms_selected[criticisms_selected_y == 1][:num_examples_to_select_per_class]

    examples_selected[data_record_metadata['file_name']] = {
        'prototypes': {
            'Wake': data_record['epoch_start_in_seconds'][prototypes_selected_wake].tolist(),
            'N1': data_record['epoch_start_in_seconds'][prototypes_selected_N1].tolist(),
        },
        'criticisms': {
            'Wake': data_record['epoch_start_in_seconds'][criticisms_selected_wake].tolist(),
            'N1': data_record['epoch_start_in_seconds'][criticisms_selected_N1].tolist(),
        }
    }

    # print(data_record_file_name)
    # print('Prototypes')
    # print(prototypes_selected)
    # print(prototypes_selected_y)
    # print('Wake')
    # print(prototypes_selected_wake)
    # print('Criticisms')
    # print(criticisms_selected)
    # print(criticisms_selected_y)
    # print('')

pprint(examples_selected)
