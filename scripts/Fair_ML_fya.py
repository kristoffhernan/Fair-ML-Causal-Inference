
# import libraries
import os
import pickle
from tqdm import tqdm

import numpy as np
import pandas as pd


# probability and modeling
import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from utils import FairKModel
from utils import FairKModelTest
from utils import FairKModelTest2

import multiprocessing

# loading data
base_path = 'Fair-ML-Causal-Inference'
law_path = os.path.join('~', base_path, 'data', 'law_data.csv')
data = pd.read_csv(law_path)

data.head()

data = data.loc[data['region_first'] != 'PO']

# keep features used by researchers
cols_keep = ['race', 'sex', 'LSAT', 'UGPA', 'ZFYA'] 
law_data = data[cols_keep]
law_data.info()

law_data.loc[:,'sex'] = np.where(law_data['sex'] == 1, 'Female', 'Male')

# split the data first to avoid data leakage
train, test = train_test_split(law_data, train_size=0.8, random_state=256)

# explicit categories and their unique values
categories = [('sex', list(law_data['sex'].unique())),
              ('race', list(law_data['race'].unique()))]

# denote column name and the unique categories as variables
ohe_columns = [x[0] for x in categories]
ohe_categories = [x[1] for x in categories]

# initialize OHE
enc = OneHotEncoder(sparse=False, categories=ohe_categories, )

# fit and transform the train
train_trans = pd.DataFrame(
    enc.fit_transform(train[ohe_columns]),
    columns = enc.get_feature_names_out(),
    index = train.index
)
# concatenate transformed cols with non transformed
train_trans = pd.concat([train.drop(ohe_columns, axis=1), train_trans], axis=1).reset_index(drop=True)
# remove the prefix the default encoding gives
train_trans.columns = [col.split('_')[1] if '_' in col else col for col in train_trans.columns]

# apply same transformation to test data
test_trans = pd.DataFrame(
    enc.fit_transform(test[ohe_columns]),
    columns = enc.get_feature_names_out(),
    index = test.index
)
# concetenate trans cols with non trans, reset index
test_trans = pd.concat([test.drop(ohe_columns, axis=1) ,test_trans], axis=1).reset_index(drop=True)
test_trans.columns = [col.split('_')[1] if '_' in col else col for col in test_trans.columns]


# round data to whole numbers to prep it for Poisson (discrete pmf)
train_trans['LSAT'] = train_trans['LSAT'].round()
test_trans['LSAT'] = test_trans['LSAT'].round()

# convert to tensors so its compatible with pytorch
X_train, y_train = torch.tensor(train_trans.drop(['ZFYA'], axis=1).values, dtype=torch.float32), torch.tensor(train_trans['ZFYA'], dtype=torch.float32) 
X_test, y_test = torch.tensor(test_trans.drop(['ZFYA'], axis=1).values, dtype=torch.float32), torch.tensor(test_trans['ZFYA'], dtype=torch.float32) 


# create and train model
data_dir = '../data'
model_dir = '../models'

train_tensor = torch.tensor(train_trans.values, dtype=torch.float32)
test_tensor = torch.tensor(test_trans.values, dtype=torch.float32)

protected_attributes = ['Female', 'Male', 'White', 'Hispanic', 'Asian', 'Black', 'Other', 'Mexican', 'Puertorican', 'Amerindian']

train_unaware = train_trans.drop(protected_attributes, axis=1)
test_unaware = test_trans.drop(protected_attributes, axis=1)


with open(os.path.join(data_dir,'inferred_K_train_100.pkl'), 'rb') as f:
  inferred_K_train = pickle.load(f)
K_list_train = inferred_K_train['K_values']
reestimated_params = inferred_K_train['parameters']
metrics_train = inferred_K_train['metrics']

K_list_test = []
metrics_test = {'ESS_values':[], 'normalized_weights':[]}

def perform_sampling(i, test_tensor, reestimated_params):
    # Your existing sampling logic here
    # Ensure this function is self-contained and does not rely on global state

    conditioned_model = pyro.condition(FairKModelTest2, data={
        'gpa': test_tensor[i, 1], 
        'lsat': test_tensor[i, 0].type(torch.int32)
    })

    # Imporance sampling
    importance = pyro.infer.Importance(conditioned_model, num_samples=100)

    # executes the mcmc process
    importance.run(R=test_tensor[:,5:], S=test_tensor[:,3:5], num_observations=test_tensor.shape[0], reestimated_params=reestimated_params) # samples from P_M(U | x^{(i)}, a^{(i)})
            
    # obtains distribution of sampled values for K
    marginal = pyro.infer.EmpiricalMarginal(importance, sites="K")
    K_list_test.append(marginal.mean)

    return marginal.mean, importance.get_ESS(), importance.get_normalized_weights()


def wrapper_func(i):
    # Wrapper function that calls perform_sampling with all necessary arguments
    return perform_sampling(i, test_tensor, reestimated_params)


def parallel_importance_sampling(test_tensor, reestimated_params, num_processes=8):
    with multiprocessing.Pool(num_processes) as pool:
        results = list(tqdm(pool.imap(wrapper_func, range(test_tensor.shape[0])), total=test_tensor.shape[0]))

    # Unpack results
    K_list_test, ESS_values, normalized_weights = zip(*results)

    # Save results
    with open(os.path.join(data_dir, 'inferred_K_test_100_nofya.pkl'), 'wb') as f:
        pickle.dump({'K_values': K_list_test, 'metrics': {'ESS_values': ESS_values, 'normalized_weights': normalized_weights}}, f)

# Usage
parallel_importance_sampling(test_tensor, reestimated_params)