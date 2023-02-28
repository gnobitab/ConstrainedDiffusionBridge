import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def generate_one_hot_mat(mat):
    upper_bound = np.max(mat)
    mat_one_hot = np.zeros((mat.shape[0], int(upper_bound+1)))
    
    for j in range(mat.shape[0]):
        mat_one_hot[j, int(mat[j])] = 1.
        
    return mat_one_hot

def generate_normalize_numerical_mat(mat):
    mat = (mat - np.min(mat))/(np.max(mat) - np.min(mat))
    #mat = 2 * (mat - 0.5)
    
    return mat
    
def get_training_data_and_info(data, discrete_columns, continuous_constraint_columns):
    ### take the pandas data, get the normalized numpy matrix for training
    from sdmetrics.utils import HyperTransformer
    ht = HyperTransformer()
    data_onehot = ht.fit_transform(data)
    data_np = data_onehot.values
     
    data_info = {}
    data_info['order'] = list(data.columns)
    data_info['discrete_pointer'] = 5
    data_info['continuous_pointer'] = 0
    continuous_ind = 0
    for col in list(data.columns):
        data_info[col] = {}
        data_info[col]['name'] = col
        if col in discrete_columns:
            data_info[col]['length'] = len(np.unique(data[col].values))
            data_info[col]['type'] = 'discrete'
            col_data = pd.DataFrame({'field': data[col]})
            enc = OneHotEncoder(handle_unknown='ignore')
            enc.fit(col_data)
            data_info[col]['enc'] = enc 
        else:
            data_info[col]['type'] = 'continuous'
            if col in continuous_constraint_columns:
                data_info[col]['bound'] = True
            else:
                data_info[col]['bound'] = False
            data_info[col]['max'] = data_np[:, continuous_ind].max()
            data_info[col]['min'] = data_np[:, continuous_ind].min()
            data_np[:, continuous_ind] = (data_np[:, continuous_ind] - data_info[col]['min']) / (data_info[col]['max'] - data_info[col]['min'])
            data_np[:, continuous_ind] = data_np[:, continuous_ind] * 2. - 1.
            print('continuous column after nomralization:', col, data_np[:, continuous_ind].min(), data_np[:, continuous_ind].max())
            continuous_ind += 1

    return data_np, data_info
      
def inverse_transform_numpy_to_pandas(data_np, data_info):
    discrete_pointer = data_info['discrete_pointer']
    continuous_pointer = data_info['continuous_pointer']
    output = {}
    for col in data_info['order']:
        if data_info[col]['type'] == 'discrete':
            output[col] = data_info[col]['enc'].inverse_transform(data_np[:, discrete_pointer:(discrete_pointer+data_info[col]['length'])])
            discrete_pointer += data_info[col]['length']
        elif data_info[col]['type'] == 'continuous':
            output_col = data_np[:, continuous_pointer] #/ 8.
            print('continuous column:', continuous_pointer, col, output_col.min(), output_col.max())
            output_col = (output_col + 1.) / 2.
            output_col = output_col * (data_info[col]['max'] - data_info[col]['min']) + data_info[col]['min']
            output[col] = output_col.astype(np.int32)
            continuous_pointer += 1
        else:
            assert False, 'Types are limited to Discrete and Continuous'
        output[col] = output[col].squeeze()

    return pd.DataFrame(data=output)
