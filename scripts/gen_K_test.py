#!/usr/bin/env python
# coding: utf-8



# In[1]:


# import libraries
import os
import pickle
from tqdm import tqdm

import numpy as np
import pandas as pd

# probability and modeling
import torch
from torch import nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import seaborn as sns
import matplotlib.pyplot as plt

from utils import FairKModelTest
from utils import FairKModelTest2


# In[2]:


# loading data
base_path = 'Fair-ML-Causal-Inference'
law_path = os.path.join('~', base_path, 'data', 'law_data.csv')
data = pd.read_csv(law_path)

data.head()


# ### Prepping Data
# 
# Remove any problem observations and select features the researchers used

# In[3]:


# remove observations low sample size observations
data = data.loc[data['region_first'] != 'PO']


# In[5]:


# keep features used by researchers
cols_keep = ['race', 'sex', 'LSAT', 'UGPA', 'ZFYA'] 
law_data = data[cols_keep]
law_data.info()


# ### OHE
# 
# One Hot Encode the categorical data so it can be used for regression

# In[6]:


# visualize the distributions for our OHE features

# convert sex to category
law_data.loc[:,'sex'] = np.where(law_data['sex'] == 1, 'Female', 'Male')


# In[7]:


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


# In[8]:


# round data to whole numbers to prep it for Poisson (only accepts integers)
train_trans['LSAT'] = train_trans['LSAT'].round()
test_trans['LSAT'] = test_trans['LSAT'].round()

# convert to tensors so its compatible with pytorch
X_train, y_train = torch.tensor(train_trans.drop(['ZFYA'], axis=1).values, dtype=torch.float32), torch.tensor(train_trans['ZFYA'], dtype=torch.float32) 
X_test, y_test = torch.tensor(test_trans.drop(['ZFYA'], axis=1).values, dtype=torch.float32), torch.tensor(test_trans['ZFYA'], dtype=torch.float32) 


# ### Linear Model

# In[9]:


# object to allow use of dataloader
class Dataset(torch.utils.data.Dataset):
     def __init__(self, dataframe):
          self.dataframe = dataframe
      
     def __len__(self):
          return self.dataframe.shape[0]
     
     def __getitem__(self, idx):
          x = torch.tensor(self.dataframe.drop(['ZFYA'], axis=1).loc[idx,:].values, dtype=torch.float32)
          y = torch.tensor(self.dataframe.loc[idx, 'ZFYA'], dtype=torch.float32)
          return x,y


# In[10]:


# class inherits form a class called nn.Module
class LinearRegressionModel(nn.Module):
    # initialization method for new class
    def __init__(self, input_size, output_size):
        # first thing always do is call initialization method from the parent class, nn.module
        super().__init__()

        # fully connected linear layer
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        # run the linear layer
        output = self.fc1(x)

        return output
    
# evaluate models performance
def evaluate(model, X_train, y_test):
    # Make predictions
    with torch.no_grad(): # disable gradient computation
        predictions = model(X_train).squeeze()

    # Calculate RMSE
    mse = torch.nn.functional.mse_loss(predictions, y_test)
    rmse = np.sqrt(mse.item())

    return rmse

def train(network, train_dataset, test_dataset, file_name_model, n_epochs=10, batch_size = 25):
    assert isinstance(file_name_model, str), "The filename is not a string"
    
    data_loader = torch.utils.data.DataLoader(Dataset(train_dataset), batch_size = batch_size, shuffle=True)
    
    X_test = torch.tensor(test_dataset.drop(['ZFYA'], axis=1).values, dtype=torch.float32)
    y_test = torch.tensor(test_dataset['ZFYA'], dtype=torch.float32)
    
    # Move network to GPU if available
    network = network.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # define optimizer
    optimizer = torch.optim.Adam(network.parameters())

    # best validation score initialization 
    validation_score_best = float('inf')
    train_losses = []
    validation_scores = []

    # train loop
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch in tqdm(data_loader, leave=False):
            # unpack batch
            X, y = batch

            # zero parameter gradients
            optimizer.zero_grad()
            
            # forward pass to get input
            # output is of shape [20,1] but we want of size [20] to compare 
            output = network(X).squeeze()

            # calculate loss
            loss = nn.MSELoss()(output, y)
            epoch_loss += loss.item()
            # root_loss = torch.sqrt(loss)
            
            # backward pass and optimize
            loss.backward()
            optimizer.step() # update model parameters
        
        avg_epoch_loss = epoch_loss / len(data_loader)  # Average loss per epoch
        train_losses.append(avg_epoch_loss)  # Append average epoch loss
        
        validation_score = evaluate(network, X_test, y_test) # evaluation mode
        validation_scores.append(validation_score)
        if epoch % 5 == 0:
            print(f'Epoch {epoch+1}, validation score: {validation_score}')
        network.train() # back to train mode

        if validation_score < validation_score_best:
            validation_score_best = validation_score
            torch.save(network.state_dict(), os.path.join('../models', file_name_model+'.pt')) 
            
    print(f'Best validation score:{validation_score_best}')
    return validation_scores, train_losses


# ### Full Model
# 
# Implements the unfair full model which includes all features

# In[12]:


# create and train model
data_dir = '../data'
model_dir = '../models'
full_model = LinearRegressionModel(train_trans.shape[1]-1, 1)
full_model.load_state_dict(torch.load(os.path.join(model_dir, 'full_model.pt')))
# full_validation_scores, full_train_losses = train(full_model, train_trans, test_trans, 'full_model', n_epochs=15, batch_size=20)


# In[13]:


categories = [item for items in ohe_categories for item in items]
categories


# In[14]:


train_tensor = torch.tensor(train_trans.values, dtype=torch.float32)
test_tensor = torch.tensor(test_trans.values, dtype=torch.float32)


# ### Unaware Model
# 
# Implements the FTU model 

# In[15]:


protected_attributes = ['Female', 'Male', 'White', 'Hispanic', 'Asian', 'Black', 'Other', 'Mexican', 'Puertorican', 'Amerindian']

train_unaware = train_trans.drop(protected_attributes, axis=1)
test_unaware = test_trans.drop(protected_attributes, axis=1)

unaware_model = LinearRegressionModel(train_unaware.shape[1]-1, 1)
unaware_model.load_state_dict(torch.load(os.path.join(model_dir,'unaware_model.pt')))
# unaware_validation_scores, unaware_train_losses = train(unaware_model, train_unaware, test_unaware, 'unaware_model', n_epochs=15, batch_size=20)





# When the sample size is 10, the ESS displays 1 for almost 492 samples. So the ESS value is 1, 99% of the time. This is a strong indicator of poor sampling efficiency. ESS Is a measure of the number of independent samples equivalent ot the correlated samples obtained from the sampling. ESS should be a significant fraction of the total number of samples. Because ESS is close to 1, it suggests the samples are highly autocorrelated and were getting little independent information from the sampling process.  
# 
# Increased sample size to 100. 

# We create a new model so that we can estimate some parameters from the training data and use them in the test phase. This is done to capture dataset-specific nuances. We want to leverage information learned from the training data in the test phase to ensure we account for potential overfitting or biased estimates which may happen when reusing model parameters from the train. 

# In[34]:

with open(os.path.join(data_dir,'inferred_K_train_100.pkl'), 'rb') as f:
  inferred_K_train = pickle.load(f)
reestimated_params = inferred_K_train['parameters']


K_list_test = []
metrics_test = {'ESS_values':[], 'normalized_weights':[]}

for i in tqdm(range(test_tensor.shape[0])):
        # P(GPA,LSAT,FYA∣K,Race,Sex), so gpa, lsat and fya conditioned on observed data from train_tensor
        conditioned_model = pyro.condition(FairKModelTest, data={
                'gpa': test_tensor[i, 1], 
                'lsat': test_tensor[i, 0].type(torch.int32), 
                'fya': test_tensor[i, 2]})

        # Imporance sampling
        importance = pyro.infer.Importance(conditioned_model, num_samples=100)

        # executes the mcmc process
        sampling_results = importance.run(R=test_tensor[:,5:], S=test_tensor[:,3:5], num_observations=test_tensor.shape[0], law_data=law_data, reestimated_params=reestimated_params) # samples from P_M(U | x^{(i)}, a^{(i)})
                
        # obtains distribution of sampled values for K
        marginal = pyro.infer.EmpiricalMarginal(importance, sites="K")
        K_list_test.append(marginal.mean)

        # collect metrics
        metrics_test['ESS_values'].append(importance.get_ESS())
        metrics_test['normalized_weights'].append(importance.get_normalized_weights())

with open(os.path.join(data_dir,'inferred_K_test_100.pkl'), 'wb') as f:
    pickle.dump({'K_values': K_list_test, 'metrics': metrics_test}, f)

