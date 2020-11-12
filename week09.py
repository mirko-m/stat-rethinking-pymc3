# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %matplotlib notebook

# %%
import pymc3 as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import theano.tensor as tt
import os

# %%
RANDOM_SEED = 93429

# %% [markdown]
# # 1

# %%
df = pd.read_csv(os.path.join('data', 'bangladesh.csv'), sep=';')
# make district variable contiguous
distric_to_idx = {d: i for i, d in enumerate(df['district'].unique())}
df['district_id'] = df.district.apply(lambda x: distric_to_idx[x])
df.head()

# %%
num_districts = df['district'].unique().shape[0]
num_districts

# %%
df['district'].max()

# %%
with pm.Model() as model_1_1:
    district_id = pm.Data('district_id', df['district_id'])
    urban = pm.Data('urban', df['urban'])
    coc_obs = pm.Data('contraception_obs', df['use.contraception'])
    
    a_bar = pm.Normal('a_bar', mu=0, sd=1)
    b_bar = pm.Normal('b_bar', mu=0, sd=0.5)
    
    sd_dist = pm.Exponential.dist(1.0) # Note the .dist to avoid creating a new variable
    chol, corr, sigmas = pm.LKJCholeskyCov('chol_cov', eta=2, n=2,
                                           sd_dist=sd_dist, compute_corr=True)
    coeffs = pm.MvNormal('coeffs', mu=[a_bar, b_bar], chol=chol, shape=(num_districts,2))
    
    p = pm.invlogit(coeffs[district_id,0] + coeffs[district_id,1]*urban)
    coc = pm.Bernoulli('contraception', p, observed=coc_obs)
    
    trace_1_1 = pm.sample(2000, tune=2000, return_inferencedata=True, random_seed=RANDOM_SEED, chains=4,
                          target_accept=0.9)

# %%
az.summary(trace_1_1, var_names=['a_bar', 'b_bar', 'chol_cov_corr', 'chol_cov_stds'], round_to=2)

# %%
mean_coeffs = trace_1_1.posterior.coeffs.mean(dim=('chain', 'draw')).values

fig, ax = plt.subplots(1,1)
fig.set_tight_layout(True)

ax.plot(range(num_districts), mean_coeffs[:,0], label='intercept')
ax.plot(range(num_districts), -mean_coeffs[:,1], label='-(slope)')
ax.set_xlabel('district_id')
ax.set_ylabel('Model Coefficient')
ax.legend(loc=0)

# %% [markdown]
# This looks as if 'slope = - intercept.'

# %%
fig, ax = plt.subplots(1,1)
fig.set_tight_layout(True)

ax.scatter(mean_coeffs[:,0], mean_coeffs[:,1])
ax.set_xlabel('intercept')
ax.set_ylabel('slope')

# %%
with model_1_1:
     ppc_1_1 = pm.sample_posterior_predictive(trace_1_1, var_names=['coeffs', 'contraception'],
                                              random_seed=RANDOM_SEED)


# %%
def sigmoid(x):
    return 1/(1+np.exp(-x))


# %%
fig, ax = plt.subplots(1,1)
fig.set_tight_layout(True)

# make predictions for urban and rural
mu_rural = ppc_1_1['coeffs'][:,:,0]
mu_urban = ppc_1_1['coeffs'][:,:,0] + ppc_1_1['coeffs'][:,:,1]*np.ones(ppc_1_1['coeffs'].shape[:2])

ax.scatter(sigmoid(mu_rural.mean(axis=0)), sigmoid(mu_urban.mean(axis=0)))
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.axhline(0.5, ls='--', c='k')
ax.axvline(0.5, ls='--', c='k')
ax.set_xlabel('rural')
ax.set_ylabel('urban')
ax.set_title('probability of using contraceptives')

# %% [markdown]
# There is a big difference between urban and rural locations. We can reparametrize the model by using two intercepts, one for urban and one for rural instead of an intercept and a slope. The two intercepts then have basically no correlation.

# %%
with pm.Model() as model_1_2:
    district_id = pm.Data('district_id', df['district_id'])
    urban = pm.Data('urban', df['urban'])
    coc_obs = pm.Data('contraception_obs', df['use.contraception'])
    
    a_bar = pm.Normal('a_bar', mu=0, sd=1)
    b_bar = pm.Normal('b_bar', mu=0, sd=0.5)
    
    sd_dist = pm.Exponential.dist(1.0) # Note the .dist to avoid creating a new variable
    chol, corr, sigmas = pm.LKJCholeskyCov('chol_cov', eta=2, n=2,
                                           sd_dist=sd_dist, compute_corr=True)
    coeffs = pm.MvNormal('coeffs', mu=[a_bar, b_bar], chol=chol, shape=(num_districts,2))
    
    p = pm.invlogit(coeffs[district_id,urban])
    coc = pm.Bernoulli('contraception', p, observed=coc_obs)
    
    trace_1_2 = pm.sample(2000, tune=2000, return_inferencedata=True, random_seed=RANDOM_SEED, chains=4,
                          target_accept=0.9)

# %%
az.summary(trace_1_2, var_names=['a_bar', 'b_bar', 'chol_cov_corr', 'chol_cov_stds'], round_to=2)

# %%
summary = az.summary(trace_1_2, round_to=2)

# %%
summary[summary.ess_tail < 2000]

# %% [markdown]
# # 2 Effect of Age and Children

# %%
df['age_sd'] = (df['age.centered'] - df['age.centered'].mean()) / df['age.centered'].std()
df['children_sd'] = (df['living.children'] - df['living.children'].mean()) / df['living.children'].std()

# %% [markdown]
# ## Model Using Age Only

# %%
with pm.Model() as model_2_1:
    district_id = pm.Data('district_id', df['district_id'])
    urban = pm.Data('urban', df['urban'])
    coc_obs = pm.Data('contraception_obs', df['use.contraception'])
    age = pm.Data('age', df['age_sd'])
    
    a_bar = pm.Normal('a_bar', mu=0, sd=1)
    b_bar = pm.Normal('b_bar', mu=0, sd=0.5)
    
    sd_dist = pm.Exponential.dist(1.0) # Note the .dist to avoid creating a new variable
    chol, corr, sigmas = pm.LKJCholeskyCov('chol_cov', eta=2, n=2,
                                           sd_dist=sd_dist, compute_corr=True)
    coeffs = pm.MvNormal('coeffs', mu=[a_bar, b_bar], chol=chol, shape=(num_districts,2))
    
    bA = pm.Normal('bA', mu=0, sd=0.5)
    
    p = pm.invlogit(coeffs[district_id,0] + coeffs[district_id,1]*urban + bA*age)
    coc = pm.Bernoulli('contraception', p, observed=coc_obs)
    
    trace_2_1 = pm.sample(2000, tune=2000, return_inferencedata=True, random_seed=RANDOM_SEED, chains=4,
                          target_accept=0.9)

# %%
az.summary(trace_2_1, var_names=['a_bar', 'b_bar', 'bA', 'chol_cov_corr', 'chol_cov_stds'], round_to=2)

# %%
fig, ax = plt.subplots(1,1)
fig.set_tight_layout(True)
az.plot_forest(trace_2_1, var_names=['a_bar', 'b_bar', 'bA'], combined=True, ax=ax)
ax.axvline(0, ls='--', c='k')

# %% [markdown]
# The total effect of age is small, but quite likely positive.

# %% [markdown]
# ## Model Using Age and Children

# %%
with pm.Model() as model_2_2:
    district_id = pm.Data('district_id', df['district_id'])
    urban = pm.Data('urban', df['urban'])
    coc_obs = pm.Data('contraception_obs', df['use.contraception'])
    age = pm.Data('age', df['age_sd'])
    children = pm.Data('children', df['children_sd'])
    
    a_bar = pm.Normal('a_bar', mu=0, sd=1)
    b_bar = pm.Normal('b_bar', mu=0, sd=0.5)
    
    sd_dist = pm.Exponential.dist(1.0) # Note the .dist to avoid creating a new variable
    chol, corr, sigmas = pm.LKJCholeskyCov('chol_cov', eta=2, n=2,
                                           sd_dist=sd_dist, compute_corr=True)
    coeffs = pm.MvNormal('coeffs', mu=[a_bar, b_bar], chol=chol, shape=(num_districts,2))
    
    bA = pm.Normal('bA', mu=0, sd=0.5)
    bC = pm.Normal('bC', mu=0, sd=0.5)
    
    p = pm.invlogit(coeffs[district_id,0] + coeffs[district_id,1]*urban + bA*age + bC*children)
    coc = pm.Bernoulli('contraception', p, observed=coc_obs)
    
    trace_2_2 = pm.sample(2000, tune=2000, return_inferencedata=True, random_seed=RANDOM_SEED, chains=4,
                          target_accept=0.9)

# %%
az.summary(trace_2_2, var_names=['a_bar', 'b_bar', 'bA', 'bC', 'chol_cov_corr', 'chol_cov_stds'], round_to=2)

# %%
fig, ax = plt.subplots(1,1)
fig.set_tight_layout(True)
az.plot_forest(trace_2_2, var_names=['a_bar', 'b_bar', 'bA', 'bC'], combined=True, ax=ax)
ax.axvline(0, ls='--', c='k')

# %% [markdown]
# The effect of age is larger and it changed sign. The effect of children is large and postive. This makes sense, because families with a lot of children may not want more children. As for age, it is possible that older people are more conservative and therefore opposed to contraception. In the previous model the effect of age was probably positive, because it incorporated the effect of children.

# %% [markdown]
# # 3 Include Children as an Ordered Variable

# %%
# delta = pm.Dirichlet('delta', np.full(NUM_EDU_CATEGORIES-1,2))
#     delta_j = pm.math.concatenate((tt.zeros(1), delta))
#     delta_j_cumsum = tt.cumsum(delta_j)

# %%
MAX_CHILDREN = df['living.children'].max()
MAX_CHILDREN

# %%
df['living.children'].value_counts()

# %% [markdown]
# Note the smallest number of children is 1.
# This means that MAX_CHILDREN is equal to the number of possible categories: 1, 2, 3, 4

# %%
with pm.Model() as model_3_1:
    district_id = pm.Data('district_id', df['district_id'])
    urban = pm.Data('urban', df['urban'])
    coc_obs = pm.Data('contraception_obs', df['use.contraception'])
    age = pm.Data('age', df['age_sd'])
    children = pm.Data('children', df['living.children'])
    
    a_bar = pm.Normal('a_bar', mu=0, sd=1)
    b_bar = pm.Normal('b_bar', mu=0, sd=0.5)
    
    sd_dist = pm.Exponential.dist(1.0) # Note the .dist to avoid creating a new variable
    chol, corr, sigmas = pm.LKJCholeskyCov('chol_cov', eta=2, n=2,
                                           sd_dist=sd_dist, compute_corr=True)
    coeffs = pm.MvNormal('coeffs', mu=[a_bar, b_bar], chol=chol, shape=(num_districts,2))
    
    bA = pm.Normal('bA', mu=0, sd=0.5)
    
    # treat children variable as ordered
    delta = pm.Dirichlet('delta', np.full(MAX_CHILDREN-1,2))
    delta_j = pm.math.concatenate((tt.zeros(1), delta))
    delta_j_cumsum = tt.cumsum(delta_j)
    
    bC = pm.Normal('bC', mu=0, sd=0.5)
    
    p = pm.invlogit(coeffs[district_id,0] + coeffs[district_id,1]*urban + bA*age +\
                    bC*delta_j_cumsum[children -1]) # note the -1, because children starts at 1
    coc = pm.Bernoulli('contraception', p, observed=coc_obs)
    
    trace_3_1 = pm.sample(2000, tune=2000, return_inferencedata=True, random_seed=RANDOM_SEED, chains=4,
                          target_accept=0.9)

# %%
az.summary(trace_3_1, var_names=['a_bar', 'b_bar', 'bA', 'bC', 'delta', 'chol_cov_corr', 'chol_cov_stds'],
           round_to=2)

# %%
fig, ax = plt.subplots(1,1)
fig.set_tight_layout(True)
az.plot_forest(trace_3_1, var_names=['a_bar', 'b_bar', 'bA', 'bC', 'delta'], combined=True, ax=ax)
ax.axvline(0, ls='--', c='k')

# %% [markdown]
# Note that `delta[0]` is corresponds to the change of going from 1 child to 2 children and is larger than `delta[1]` and `delta[2]`. This means that having a second child has the biggest impact on the use of contraception.
#
# Note also that `bC` is larger than for the previous model. This makes sense, because a large `bC` value in the previous model would mean that having 4 children has a very large effect, whereas when children is included as an ordered variable the effect of having a 4th child can be rather small compared to the effect of having a second or third child.

# %%
