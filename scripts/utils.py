
import torch
import pyro
import pyro.distributions as dist


def FairKModel(R, S, num_observations, law_data, GPA=None, LSAT=None, FYA=None):
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
    # print(w_G_R.shape, R.shape)
    # print((torch.matmul(w_G_R, R.transpose(0,1))).shape)
    # # w 1 x num_col_race
    # # R len_race x num_col_race
    # print((torch.matmul(w_G_S, S)).shape)

    mu_G = b_G + w_G_K * K + torch.matmul(w_G_R, R.transpose(0,1)) + torch.matmul(w_G_S, S.transpose(0,1))
    lambda_L = b_L + w_L_K * K + torch.matmul(w_L_R, R.transpose(0,1)) + torch.matmul(w_L_S, S.transpose(0,1))
    mu_F = w_F_K * K + torch.matmul(w_F_R, R.transpose(0,1)) + torch.matmul(w_F_S, S.transpose(0,1))

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




def FairKModelTest(R, S, num_observations, law_data, reestimated_params, GPA=None, LSAT=None, FYA=None):
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
    mu_G = b_G + w_G_K * K + torch.matmul(w_G_R, R.transpose(0,1)) + torch.matmul(w_G_S, S.transpose(0,1))
    lambda_L = b_L + w_L_K * K + torch.matmul(w_L_R, R.transpose(0,1)) + torch.matmul(w_L_S, S.transpose(0,1))
    mu_F = w_F_K * K + torch.matmul(w_F_R, R.transpose(0,1)) + torch.matmul(w_F_S, S.transpose(0,1))

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






def FairKModelTest2(R, S, num_observations, reestimated_params, GPA=None, LSAT=None, FYA=None):
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

    mu_G = b_G + w_G_K * K + torch.matmul(w_G_R, R.transpose(0,1)) + torch.matmul(w_G_S, S.transpose(0,1))
    lambda_L = b_L + w_L_K * K + torch.matmul(w_L_R, R.transpose(0,1)) + torch.matmul(w_L_S, S.transpose(0,1))

    # sample observed data
    # pyro.plate is to denote independent observations and for vectorized computation
    # use if dealing with multiple observations
    # num_observations ensures models LH calculations are performed across all datapoints
    with pyro.plate('data', num_observations):
        # gives likelihood of observed data given model parameters
        gpa = pyro.sample('gpa', dist.Normal(mu_G, torch.sqrt(sigma_G_sq)), obs=GPA) # obs is observed
        lsat = pyro.sample('lsat', dist.Poisson(lambda_L.exp()), obs=LSAT)

    return gpa, lsat