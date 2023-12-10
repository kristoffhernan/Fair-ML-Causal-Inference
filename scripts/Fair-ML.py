#!/usr/bin/env python
# coding: utf-8

# In[1]:

# import libraries
import os
import pickle
from tqdm.notebook import tqdm

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


# In[4]:


# keep features used by researchers
cols_keep = ['race', 'sex', 'LSAT', 'UGPA', 'ZFYA'] 
law_data = data[cols_keep]
law_data.info()


# ### OHE
# 
# One Hot Encode the categorical data so it can be used for regression

# In[5]:


# convert sex to category
law_data.loc[:,'sex'] = np.where(law_data['sex'] == 1, 'Female', 'Male')


# In[6]:


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


# In[7]:


# round data to whole numbers to prep it for Poisson (only accepts integers)
train_trans['LSAT'] = train_trans['LSAT'].round()
test_trans['LSAT'] = test_trans['LSAT'].round()

# convert to tensors so its compatible with pytorch
X_train, y_train = torch.tensor(train_trans.drop(['ZFYA'], axis=1).values, dtype=torch.float32), torch.tensor(train_trans['ZFYA'], dtype=torch.float32) 
X_test, y_test = torch.tensor(test_trans.drop(['ZFYA'], axis=1).values, dtype=torch.float32), torch.tensor(test_trans['ZFYA'], dtype=torch.float32) 


# ### Linear Model

# In[8]:


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


# In[9]:


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
            torch.save(network.state_dict(), file_name_model+'.pt') 
            
    print(f'Best validation score:{validation_score_best}')
    return validation_scores, train_losses


# ### Full Model
# 
# Implements the unfair full model which includes all features

# In[10]:


# create and train model
full_model = LinearRegressionModel(train_trans.shape[1]-1, 1)
full_model.load_state_dict(torch.load('full_model.pt'))
# full_validation_scores, full_train_losses = train(full_model, train_trans, test_trans, 'full_model', n_epochs=15, batch_size=20)


# In[11]:


categories = [item for items in ohe_categories for item in items]
categories


# In[12]:


train_tensor = torch.tensor(train_trans.values, dtype=torch.float32)
test_tensor = torch.tensor(test_trans.values, dtype=torch.float32)

# Look at distribution of FYA on difference races and sex on train set
fit = train_trans.copy()
fit['FYA'] = full_model(X_train).detach().numpy()

fig, (ax1, ax2) = plt.subplots(2)
fig.set_size_inches(9, 6)

for column in ohe_categories[0]:
    data_filtered = fit[fit[column]==1]
    sns.kdeplot(data=data_filtered, x='FYA', label=column, ax=ax1)
ax1.legend()

for column in ohe_categories[1]:
    data_filtered = fit[fit[column]==1]
    sns.kdeplot(data=data_filtered, x='FYA', label=column, ax=ax2)

ax2.legend()

plt.show()


# ### Unaware Model
# 
# Implements the FTU model 

# In[13]:


protected_attributes = ['Female', 'Male', 'White', 'Hispanic', 'Asian', 'Black', 'Other', 'Mexican', 'Puertorican', 'Amerindian']

train_unaware = train_trans.drop(protected_attributes, axis=1)
test_unaware = test_trans.drop(protected_attributes, axis=1)

unaware_model = LinearRegressionModel(train_unaware.shape[1]-1, 1)
unaware_model.load_state_dict(torch.load('unaware_model.pt'))
# unaware_validation_scores, unaware_train_losses = train(unaware_model, train_unaware, test_unaware, 'unaware_model', n_epochs=15, batch_size=20)


# In[14]:


# Look at distribution of FYA on difference races and sex on train set
fit = train_trans.copy()
fit['FYA'] = unaware_model(torch.tensor(train_unaware.drop('ZFYA', axis=1).values, dtype=torch.float32)).detach().numpy()

fig, (ax1, ax2) = plt.subplots(2)
fig.set_size_inches(9, 6)

for column in ohe_categories[0]:
    data_filtered = fit[fit[column]==1]
    sns.kdeplot(data=data_filtered, x='FYA', label=column, ax=ax1)
ax1.legend()

for column in ohe_categories[1]:
    data_filtered = fit[fit[column]==1]
    sns.kdeplot(data=data_filtered, x='FYA', label=column, ax=ax2)

ax2.legend()

plt.show()


# ### Fair K
# 
# Fair K introduces a background latend variables, K, which are not decendants of protected demographic factors. Information about X is passed to $\hat{Y}$ via $P(K|x,a)$
# 
# If we can't calculate P_M(K|x,a) analytically use the following algorithm:
# 
# 
# 
# START Procedure FAIRLEARNING($D,M$)
# 
# 1. For each datapoint $i \in D$, sample $m$ MCMC samples $K_1^{(i)}, \cdots, K_1^{(i)} \sim P_M(K | x^{(i)}, a^{(i)})$
# 2. Let $D'$ be the augmentd dataset where each ponit $(a^{(i)}, x^{(i)}, y^{(i)})$ in $D$ is replaced with the corresponding $m$ points ${ (a^{(i)}, x^{(i)}, y^{(i)}, k_j^{(i)}) }$
# 3. $\hat{\theta} \leftarrow argmin_\theta \sum_{i’ \in D’} l(y^{(i’)}, g_\theta(K^{(i’)}, x^{(i’)}_{\ A}))$
# 
# END procedure
# 
# 
# To solve 1 of the FAIRLEARNING we can use bayes $P(A|B) = P(B|A)P(A) / P(B)$. 
# 
# In our scenario, $P(K∣GPA,LSAT,FYA,Race,Sex)$ is the posterior distribution of $K$. 
# 
# $P(GPA,LSAT,FYA∣K,Race,Sex)$ is the likelihood of observing the data given $K$. 
# 
# $P(K)$ is the prior distribution of $K$, representing our beliefs about $K$ before observing the data. 
# 
# $P(GPA,LSAT,FYA,Race,Sex)$ is the evidence, or the probability of observing the data under all possible values of $K$.
# 
# We can then sample from our posterior using MCMC
# 
# 
# 
# 
# In the FAIRLEARNING algorithm, the augmented dataset {(a(i),x(i),y(i),uj(i))} includes all these variables, but the key aspect is how they are used. The model g_\theta(U(i),xA(i))typically uses the inferred latent variables U(i) (in this case, K values from K_list_train) and the non-protected attributes A(i)​ to predict the outcome Y. The protected attributes A (like race and sex in your model) are not directly used in the prediction to ensure fairness; instead, their effect is mediated through the latent variables. This approach aims to make predictions based on factors like knowledge while accounting for potential biases in the observed data.

# In[29]:


def FairKModel(R, S,num_observations, GPA=None, LSAT=None, FYA=None):
    num_race_cats = len(law_data['race'].unique())
    num_sex_cats = len(law_data['sex'].unique())    

    # 0,1 vectors for the normal distribution
    # R and S are matrices
    r0_vec = torch.zeros(num_race_cats)
    s0_vec = torch.zeros(num_sex_cats)
    r1_vec = torch.ones(num_race_cats)
    s1_vec = torch.ones(num_sex_cats)

    # prior latent variable 'K' (knowledge)
    K = pyro.sample('K', dist.Normal(torch.tensor(0.), torch.tensor(1.)))

    # priors for weights and baselines for GPA, LSAT and FYA
    # GPA ~ N(b_G  + w_G^K K + w_G^R R + w_G^S S, \sigma_G)
    b_G = pyro.sample('b_G', dist.Normal(torch.tensor(0.), torch.tensor(1.)))
    w_G_K = pyro.sample('w_G_K', dist.Normal(torch.tensor(0.), torch.tensor(1.)))
    w_G_R = pyro.sample('w_G_R', dist.Normal(r0_vec,r1_vec)) # outputs a vec of normals of len(race)
    w_G_S = pyro.sample('w_G_S', dist.Normal(s0_vec,s1_vec)) # outputs a vec of normals of len(sex)
    sigma_G_sq = pyro.sample('sigma_G_sq', dist.InverseGamma(torch.tensor(1.), torch.tensor(1.)))

    # LSAT ~ Poisson(exp( b_L +w_L^K K + w_L^R R + w_L^S S))
    b_L = pyro.sample('b_L', dist.Normal(torch.tensor(0.), torch.tensor(1.)))
    w_L_K = pyro.sample('w_L_K', dist.Normal(torch.tensor(0.), torch.tensor(1.)))
    w_L_R = pyro.sample('w_L_R', dist.Normal(r0_vec,r1_vec))
    w_L_S = pyro.sample('w_L_S', dist.Normal(s0_vec,s1_vec))

    # FYA ~ N(w_F^K K + w_F^R R + w_F^S S, 1)
    w_F_K = pyro.sample('w_F_K', dist.Normal(torch.tensor(0.), torch.tensor(1.)))
    w_F_R = pyro.sample('w_F_R', dist.Normal(r0_vec,r1_vec))
    w_F_S = pyro.sample('w_F_S', dist.Normal(s0_vec,s1_vec))

    # Calculate the parameters values of the data generating distributions
    # print(len(b_G))
    # print(len(w_G_K * K))
    # print((torch.matmul(w_G_R, R.transpose(0,1))).shape)
    # # w 1 x num_col_race
    # # R len_race x num_col_race
    # print((torch.matmul(w_G_S, S)).shape)

    mu_G = b_G + w_G_K * K + torch.matmul(R, w_G_R) + torch.matmul(S, w_G_S)
    lambda_L = b_L + w_L_K * K + torch.matmul(R, w_L_R) + torch.matmul(S, w_L_S)
    mu_F = w_F_K * K + torch.matmul(R, w_F_R) + torch.matmul(S, w_F_S)

    # sample observed data
    # pyro.plate is to denote independent observations and for vectorized computation
    # use if dealing with multiple observations
    # num_observations ensures models LH calculations are performed across all datapoints
    with pyro.plate('data', num_observations):
        # gives likelihood of observed data given model parameters
        gpa = pyro.sample('gpa', dist.Normal(mu_G, torch.sqrt(sigma_G_sq)), obs=GPA) # dist.Normal takes mean, sd (not variance)
        lsat = pyro.sample('lsat', dist.Poisson(lambda_L.exp()), obs=LSAT) # obs is observed
        fya = pyro.sample('fya', dist.Normal(mu_F, torch.tensor(1.)), obs=FYA)

    return gpa, lsat, fya


# In[30]:


# structure of FairKModel
# generates a DAG of the model, showing how different rvs are related to each other within the model
model_graph = pyro.render_model(
    FairKModel, 
    model_args=(train_tensor[:,5:], train_tensor[:,3:5], train_tensor.shape[0], train_tensor[:,1], train_tensor[:,0], train_tensor[:,2]),
    render_distributions=True, 
    render_params=True
    )
model_graph


# In[32]:


from tqdm.notebook import tqdm

values_reestimate = ['b_G', 'w_G_R', 'w_G_S', 'w_G_K', 'sigma_G_sq', 'b_L', 'w_L_R', 'w_L_S', 'w_L_K']

K_list_train = []
reestimated_params = {param: [] for param in values_reestimate}
metrics_train = {'ESS_values':[], 'normalized_weights':[]}

for i in tqdm(range(train_tensor.shape[0])):
        # P(GPA,LSAT,FYA∣K,Race,Sex), so gpa, lsat and fya conditioned on observed data from train_tensor
        conditioned_model = pyro.condition(FairKModel, data={
                'gpa': train_tensor[i, 1], 
                'lsat': train_tensor[i, 0].type(torch.int32), 
                'fya': train_tensor[i, 2]})

        # MCMC is too expensive to run
        # initialize No-U-Turn Sampler a type of mcmc method
        # nuts_kernel = pyro.infer.mcmc.NUTS(conditioned_model)
        # sets up mcmc process using nuts kernel. draws 500 samples from the posterior and uses 500 iterations to tune the sampler
        # mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=500, warmup_steps=200, num_chains= os.cpu_count() // 2)

        # Imporance sampling
        importance = pyro.infer.Importance(conditioned_model, num_samples=100)

        # executes the mcmc process
        sampling_results = importance.run(R=train_tensor[:,5:], S=train_tensor[:,3:5], num_observations=train_tensor.shape[0]) # samples from P_M(U | x^{(i)}, a^{(i)})

        # Extract weighted samples and calculate metrics
        weighted_samples = importance.exec_traces
        for estimate in values_reestimate:
                values = torch.stack([trace.nodes[estimate]['value'] for trace in weighted_samples])
                # Compute the mean for each column if it's a matrix
                reestimated_params[estimate].append(values.mean(0) if values.ndim > 1 else values.mean())
                
        # obtains distribution of sampled values for K
        marginal = pyro.infer.EmpiricalMarginal(importance, sites="K")
        K_list_train.append(marginal.mean)

        # collect metrics
        metrics_train['ESS_values'].append(importance.get_ESS())
        metrics_train['normalized_weights'].append(importance.get_normalized_weights())

with open('inferred_K_train_100.pkl', 'wb') as f:
    pickle.dump({'K_values': K_list_train, 'parameters': reestimated_params, 'metrics': metrics_train}, f)


# When the sample size is 10, the ESS displays 1 for almost 492 samples. So the ESS value is 1, 99% of the time. This is a strong indicator of poor sampling efficiency. ESS Is a measure of the number of independent samples equivalent ot the correlated samples obtained from the sampling. ESS should be a significant fraction of the total number of samples. Because ESS is close to 1, it suggests the samples are highly autocorrelated and were getting little independent information from the sampling process.  
# 
# Increased sample size to 100. 

# We create a new model so that we can estimate some parameters from the training data and use them in the test phase. This is done to capture dataset-specific nuances. We want to leverage information learned from the training data in the test phase to ensure we account for potential overfitting or biased estimates which may happen when reusing model parameters from the train. 

# In[33]:


def FairKModelTest(R, S, num_observations, GPA=None, LSAT=None, FYA=None):
    num_race_cats = len(law_data['race'].unique())
    num_sex_cats = len(law_data['sex'].unique())    

    # 0,1 vectors for the normal distribution
    # R and S are matrices
    r0_vec = torch.zeros(num_race_cats)
    s0_vec = torch.zeros(num_sex_cats)
    r1_vec = torch.ones(num_race_cats)
    s1_vec = torch.ones(num_sex_cats)

    # prior latent variable 'K' (knowledge)
    K = pyro.sample('K', dist.Normal(torch.tensor(0.), torch.tensor(1.)))

    # priors for weights and baselines for GPA, LSAT and FYA
    # GPA ~ N(b_G  + w_G^K K + w_G^R R + w_G^S S, \sigma_G)
    b_G = torch.stack(reestimated_params['b_G']).mean()
    w_G_K = torch.stack(reestimated_params['w_G_K']).mean()
    w_G_R = torch.stack(reestimated_params['w_G_R']).mean(0)
    w_G_S = torch.stack(reestimated_params['w_G_S']).mean(0)
    sigma_G_sq = torch.stack(reestimated_params['sigma_G_sq']).mean()

    # LSAT ~ Poisson(exp( b_L +w_L^K K + w_L^R R + w_L^S S))
    b_L =torch.stack( reestimated_params['b_L']).mean()
    w_L_K = torch.stack(reestimated_params['w_L_K']).mean()
    w_L_R = torch.stack(reestimated_params['w_L_R']).mean(0)
    w_L_S = torch.stack(reestimated_params['w_L_S']).mean(0)

    # FYA ~ N(w_F^K K + w_F^R R + w_F^S S, 1)
    w_F_K = pyro.sample('w_F_K', dist.Normal(torch.tensor(0.), torch.tensor(1.)))
    w_F_R = pyro.sample('w_F_R', dist.Normal(r0_vec,r1_vec))
    w_F_S = pyro.sample('w_F_S', dist.Normal(s0_vec,s1_vec))

    # Calculate the parameters values of the data generating distributions
    # print(len(b_G))
    # print(len(w_G_K * K))
    # print((torch.matmul(w_G_R, R.transpose(0,1))).shape)
    # # w 1 x num_col_race
    # # R len_race x num_col_race
    # produces 1 x len_race
    # print((torch.matmul(w_G_S, S.transpose(0,1))).shape)
    # print(b_G.shape)
    # print((w_G_K * K).shape)

    mu_G = b_G + w_G_K * K + torch.matmul(R, w_G_R) + torch.matmul(S, w_G_S)
    lambda_L = b_L + w_L_K * K + torch.matmul(R, w_L_R) + torch.matmul(S, w_L_S)
    mu_F = w_F_K * K + torch.matmul(R, w_F_R) + torch.matmul(S, w_F_S)

    # sample observed data
    # pyro.plate is to denote independent observations and for vectorized computation
    # use if dealing with multiple observations
    # num_observations ensures models LH calculations are performed across all datapoints
    with pyro.plate('data', num_observations):
        # gives likelihood of observed data given model parameters
        gpa = pyro.sample('gpa', dist.Normal(mu_G, torch.sqrt(sigma_G_sq)), obs=GPA) # obs is observed
        lsat = pyro.sample('lsat', dist.Poisson(lambda_L.exp()), obs=LSAT)
        fya = pyro.sample('fya', dist.Normal(mu_F, torch.tensor(1.)), obs=FYA)

    return gpa, lsat, fya


# In[34]:


from tqdm.notebook import tqdm

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
        sampling_results = importance.run(R=test_tensor[:,5:], S=test_tensor[:,3:5], num_observations=test_tensor.shape[0]) # samples from P_M(U | x^{(i)}, a^{(i)})
                
        # obtains distribution of sampled values for K
        marginal = pyro.infer.EmpiricalMarginal(importance, sites="K")
        K_list_test.append(marginal.mean)

        # collect metrics
        metrics_test['ESS_values'].append(importance.get_ESS())
        metrics_test['normalized_weights'].append(importance.get_normalized_weights())

with open('inferred_K_test_100.pkl', 'wb') as f:
    pickle.dump({'K_values': K_list_test, 'metrics': metrics_test}, f)


# In[35]:


def FairKModelTest2(R, S, num_observations, GPA=None, LSAT=None, FYA=None):
    # prior latent variable 'K' (knowledge)
    K = pyro.sample('K', dist.Normal(torch.tensor(0.), torch.tensor(1.)))

    # priors for weights and baselines for GPA, LSAT and FYA
    # GPA ~ N(b_G  + w_G^K K + w_G^R R + w_G^S S, \sigma_G)
    b_G = torch.stack(reestimated_params['b_G']).mean()
    w_G_K = torch.stack(reestimated_params['w_G_K']).mean()
    w_G_R = torch.stack(reestimated_params['w_G_R']).mean(0)
    w_G_S = torch.stack(reestimated_params['w_G_S']).mean(0)
    sigma_G_sq = torch.stack(reestimated_params['sigma_G_sq']).mean()

    # LSAT ~ Poisson(exp( b_L +w_L^K K + w_L^R R + w_L^S S))
    b_L =torch.stack( reestimated_params['b_L']).mean()
    w_L_K = torch.stack(reestimated_params['w_L_K']).mean()
    w_L_R = torch.stack(reestimated_params['w_L_R']).mean(0)
    w_L_S = torch.stack(reestimated_params['w_L_S']).mean(0)

    mu_G = b_G + w_G_K * K + torch.matmul(R, w_G_R) + torch.matmul(S, w_G_S)
    lambda_L = b_L + w_L_K * K + torch.matmul(R, w_L_R) + torch.matmul(S, w_L_S)

    # sample observed data
    # pyro.plate is to denote independent observations and for vectorized computation
    # use if dealing with multiple observations
    # num_observations ensures models LH calculations are performed across all datapoints
    with pyro.plate('data', num_observations):
        # gives likelihood of observed data given model parameters
        gpa = pyro.sample('gpa', dist.Normal(mu_G, torch.sqrt(sigma_G_sq)), obs=GPA) # obs is observed
        lsat = pyro.sample('lsat', dist.Poisson(lambda_L.exp()), obs=LSAT)

    return gpa, lsat


# In[20]:


from tqdm.notebook import tqdm

K_list_test = []
metrics_test = {'ESS_values':[], 'normalized_weights':[]}

for i in tqdm(range(test_tensor.shape[0])):
        # P(GPA,LSAT,FYA∣K,Race,Sex), so gpa, lsat and fya conditioned on observed data from train_tensor
        conditioned_model = pyro.condition(FairKModelTest2, data={
                'gpa': test_tensor[i, 1], 
                'lsat': test_tensor[i, 0].type(torch.int32), 
                }
                )

        # Imporance sampling
        importance = pyro.infer.Importance(conditioned_model, num_samples=100)

        # executes the mcmc process
        sampling_results = importance.run(R=test_tensor[:,5:], S=test_tensor[:,3:5], num_observations=test_tensor.shape[0]) # samples from P_M(U | x^{(i)}, a^{(i)})
                
        # obtains distribution of sampled values for K
        marginal = pyro.infer.EmpiricalMarginal(importance, sites="K")
        K_list_test.append(marginal.mean)

        # collect metrics
        metrics_test['ESS_values'].append(importance.get_ESS())
        metrics_test['normalized_weights'].append(importance.get_normalized_weights())

with open('inferred_K_test_100_nofya.pkl', 'wb') as f:
    pickle.dump({'K_values': K_list_test, 'metrics': metrics_test}, f)

