# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
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
# # Marriage Rate Data

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

# %%
