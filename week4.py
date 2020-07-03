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
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import arviz as az
import scipy.stats as stats
from scipy.special import logsumexp
import pandas as pd
import os

# %% [markdown]
# # 1 Polynesian Islands

# %%
data_1 = {'birbA': [0.2, 0.8, 0.05],
        'birbB': [0.2, 0.1, 0.15],
        'birbC': [0.2, 0.05, 0.7],
        'birbD': [0.2, 0.025, 0.05],
        'birbE': [0.2, 0.025, 0.05]}

df_1 = pd.DataFrame(data=data_1)
df_1

# %% [markdown]
# ## Compute Entropy

# %%
birb_matrix = df_1[['birbA', 'birbB', 'birbC', 'birbD', 'birbE']].values
entropy = -(birb_matrix * np.log(birb_matrix)).sum(axis=1)
df_1['entropy'] = entropy
df_1


# %% [markdown]
# Island 0 has the largest entropy, because the birb distribution is uniform. Island 1 on the other hand has a smaller entropy, because the birb distribution is peaked at `birbA`. Island 2 is in between.

# %% [markdown]
# ## Compute KL Divergences

# %% [markdown]
# $$
# D_{KL}  = - \sum_{i}  p_i \left ( \log(q_i) - \log(p_i) \right )
# $$

# %% [markdown]
# Here $ p $ is assumed to be the "true" distribution and $q$ is some other distribution. 

# %%
def calc_kl_div(p_index, q_index, mat):
    p = mat[p_index,:]
    q = mat[q_index,:]
    return - np.sum( p* ( np.log(q) - np.log(p) ) )


# %%
dkl = np.zeros((3,3))
for q_index in range(3):
    for p_index in range(3):
        dkl[q_index, p_index] = calc_kl_div(p_index, q_index, birb_matrix)
np.round(dkl, 2)

# %% [markdown]
# The top row is the KL Divergence if one estimates the birb population of Island 1 and 2 using the birb distribution from Island 0. This means that the "true" distribution $p$ is given by Island 1/2 and $q$ is given by Island 0. Since Island 0 has the largest entropy the top row has the smalles KL Divergence values.

# %% [markdown]
# # 2 Marriage Rate Data

# %%
df_2 = pd.read_csv(os.path.join('data','happiness.csv'),sep=';')
df_2 = df_2[df_2['age'] > 17]

# %%
df_2.head()

# %%
# normalize age (hapiness is already normalized)
df_2['age'] = (df_2['age'] - 18) / (65-18)

# %%
with pm.Model() as model_6_9:
    a = pm.Normal('a', mu=0, sigma=1, shape=2)
    bA = pm.Normal('bA', mu=0, sigma=2)
    sigma = pm.Exponential('sigma',lam=1)
    
    mu = pm.Deterministic('mu', a[df_2['married']] + bA*df_2['age'])
    happiness = pm.Normal('happiness', mu=mu, sigma=sigma, observed=df_2['happiness'])
    trace_6_9 = pm.sample(1000, tune=1000)

# %%
_ = az.plot_trace(trace_6_9,figsize=(8,6), var_names=['~mu'], )

# %%
az.summary(trace_6_9, var_names=['~mu'], hdi_prob=0.89).round(2)

# %% [markdown]
# What about the exceptions?
#
# 1. The one about indexing seems to be fine: https://discourse.pymc.io/t/multidimensional-indexing/1611/10
# 2. Presumably the one about `from_pymc3` will be solved in the future? I can also be solved like this:
#
# ```python
# with model_6_9:
#     summary = az.summary(trace_6_9, var_names=['~mu'], hdi_prob=0.89).round(2)
# summary
# ```

# %%
with pm.Model() as model_6_10:
    a = pm.Normal('a', mu=0, sigma=1)
    bA = pm.Normal('bA', mu=0, sigma=2)
    sigma = pm.Exponential('sigma',lam=1)
    
    mu = pm.Deterministic('mu', a + bA*df_2['age'])
    happiness = pm.Normal('happiness', mu=mu, sigma=sigma, observed=df_2['happiness'])
    trace_6_10 = pm.sample(1000, tune=10000)

# %%
_ = az.plot_trace(trace_6_10,figsize=(8,6), var_names=['~mu'], )

# %%
az.summary(trace_6_10, var_names=['~mu'], hdi_prob=0.89).round(2)

# %% [markdown]
# ## Compare WAIC

# %%
with model_6_9:
    waic_6_9 = pm.waic(trace_6_9)
    
with model_6_10:
    waic_6_10 = pm.waic(trace_6_10)

# %%
waic_6_9

# %%
waic_6_10

# %%
df_comp_waic = pm.compare({'model_6_9': trace_6_9, 'model_6_10': trace_6_10}, ic='waic', scale='deviance')
df_comp_waic

# %%
pm.compareplot(df_comp_waic)

# %% [markdown]
# ## Compare with Loo

# %%
with model_6_9:
    loo_6_9 = pm.loo(trace_6_9)
    
with model_6_10:
    loo_6_10 = pm.loo(trace_6_10)

# %%
loo_6_9

# %%
loo_6_10

# %%
df_comp_loo = pm.compare({'model_6_9': trace_6_9, 'model_6_10': trace_6_10}, ic='loo', scale='deviance')
df_comp_loo

# %%
pm.compareplot(df_comp_loo)


# %% [markdown]
# Because the data is simulated we know that model_6_10 is the correct model, but WAIC and LOO tell us that model_6_9 makes better predictions. We cannot use WAIC and LOO to answer questions about causality.

# %% [markdown]
# Why do I get identical results for WAIC and LOO? I would have expected a small difference.

# %% [markdown]
# ## Manually Calculate WAIC

# %%
def calc_waic(trace, observations):
    '''
    helper function to calculate waic manually.
    
    Parameters
    ----------
    trace: pymc3 trace object. Must have parameters `mu` and `sigma`
    observations: array with observed data
    
    Returns
    -------
    waic: (float) Widely Applicable Information Criterion in deviance scale
    '''
    
    n_obs = trace['mu'].shape[1] # number of observations
    n_samples = trace['mu'].shape[0] # number of samples from posterior
    
    # need to repeate sigma along axis=1 to match the shape of mu
    ss = np.repeat(trace['sigma'].reshape(-1,1), n_obs, axis=1)
    logprob = stats.norm(loc=trace['mu'], scale=ss).logpdf(observations)

    # calculate log-pointwise-predictive-density (lppd)
    lppd = logsumexp(logprob, axis=0) - np.log(n_samples)

    # calculate WAIC penalty term
    pwaic = np.var(logprob, axis=0)

    waic = -2*(np.sum(lppd) - np.sum(pwaic))
    return waic


# %%
manual_waic_6_9 = calc_waic(trace_6_9, df_2['happiness'].values)
manual_waic_6_10 = calc_waic(trace_6_10, df_2['happiness'].values)
print('waic for model_6_9 = {:.2f}'.format(manual_waic_6_9))
print('waic for model_6_10 = {:.2f}'.format(manual_waic_6_10))

# %% [markdown]
# This matches the values from the built-in function.

# %% [markdown]
# # 3 Foxes

# %%
df_3 = pd.read_csv('data/foxes.csv', sep=';')
df_3.head()

# %%
df_3['avgfood_sd'] = ( df_3['avgfood'] - df_3['avgfood'].mean() ) / df_3['avgfood'].std()
df_3['groupsize_sd'] = ( df_3['groupsize'] - df_3['groupsize'].mean() ) / df_3['groupsize'].std()
df_3['area_sd'] = ( df_3['area'] - df_3['area'].mean() ) / df_3['area'].std()
df_3['weight_sd'] = ( df_3['weight'] - df_3['weight'].mean() ) / df_3['weight'].std()

# %% [markdown]
# model_f_1: avgfood + groupsize + area
#
# model_f_2: avgfood + groupsize
#
# model_f_3: groupsize + area
#
# model_f_4: avgfood
#
# model_f_5: area

# %%
with pm.Model() as model_f_1:
    
    a = pm.Normal('a', mu=0, sigma=1)
    bF = pm.Normal('bF', mu=0, sigma=2)
    bG = pm.Normal('bG', mu=0, sigma=2)
    bA = pm.Normal('bA', mu=0, sigma=2)
    sigma = pm.Exponential('sigma',lam=1)
    
    mu = pm.Deterministic('mu', a + bF*df_3['avgfood_sd'] + bG*df_3['groupsize_sd'] + bA*df_3['area_sd'])
    weight = pm.Normal('happiness', mu=mu, sigma=sigma, observed=df_3['weight'])
    trace_f_1 = pm.sample(1000, tune=1000)

# %%
with pm.Model() as model_f_2:
    
    a = pm.Normal('a', mu=0, sigma=1)
    bF = pm.Normal('bF', mu=0, sigma=2)
    bG = pm.Normal('bG', mu=0, sigma=2)
    sigma = pm.Exponential('sigma',lam=1)
    
    mu = pm.Deterministic('mu', a + bF*df_3['avgfood_sd'] + bG*df_3['groupsize_sd'])
    weight = pm.Normal('happiness', mu=mu, sigma=sigma, observed=df_3['weight'])
    trace_f_2 = pm.sample(1000, tune=1000)

# %%
with pm.Model() as model_f_3:
    
    a = pm.Normal('a', mu=0, sigma=1)
    bG = pm.Normal('bG', mu=0, sigma=2)
    bA = pm.Normal('bA', mu=0, sigma=2)
    sigma = pm.Exponential('sigma',lam=1)
    
    mu = pm.Deterministic('mu', a + bG*df_3['groupsize_sd'] + bA*df_3['area_sd'])
    weight = pm.Normal('happiness', mu=mu, sigma=sigma, observed=df_3['weight'])
    trace_f_3 = pm.sample(1000, tune=1000)

# %%
with pm.Model() as model_f_4:
    
    a = pm.Normal('a', mu=0, sigma=1)
    bF = pm.Normal('bF', mu=0, sigma=2)
    sigma = pm.Exponential('sigma',lam=1)
    
    mu = pm.Deterministic('mu', a + bF*df_3['avgfood_sd'])
    weight = pm.Normal('happiness', mu=mu, sigma=sigma, observed=df_3['weight'])
    trace_f_4 = pm.sample(1000, tune=1000)

# %%
with pm.Model() as model_f_5:
    
    a = pm.Normal('a', mu=0, sigma=1)
    bA = pm.Normal('bA', mu=0, sigma=2)
    sigma = pm.Exponential('sigma',lam=1)
    
    mu = pm.Deterministic('mu', a + bA*df_3['area_sd'])
    weight = pm.Normal('happiness', mu=mu, sigma=sigma, observed=df_3['weight'])
    trace_f_5 = pm.sample(1000, tune=1000)

# %%
trace_dict = {'model_f_1': trace_f_1,
              'model_f_2': trace_f_2,
              'model_f_3': trace_f_3,
              'model_f_4': trace_f_4,
              'model_f_5': trace_f_5}

# %%
foxes_comp_waic = pm.compare(trace_dict, ic='waic', scale='deviance')
foxes_comp_waic

# %%
pm.compareplot(foxes_comp_waic)

# %%
foxes_comp_loo = pm.compare(trace_dict, ic='loo', scale='deviance')
foxes_comp_loo

# %%
pm.compareplot(foxes_comp_loo)

# %% [markdown]
# Reminder:
# --------------
#
# model_f_1: avgfood + groupsize + area
#
# model_f_2: avgfood + groupsize
#
# model_f_3: groupsize + area
#
# model_f_4: avgfood
#
# model_f_5: area

# %% [markdown]
# The first three models all perform very similar. The reason is that for the sake of prediction `area` and `avgfood` can be interchanged. That is not surprising when looking at the DAG for this data (not shown here, but see week3.pdf)

# %%
