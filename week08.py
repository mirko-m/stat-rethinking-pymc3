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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az
import os

# %%
RANDOM_SEED = 95714


# %%
def sigmoid(x):
    return 1.0/(np.exp(-x) + 1)


# %% [markdown]
# # 1

# %%
df1 = pd.read_csv(os.path.join('data', 'reedfrogs.csv'), sep=';')

# %%
df1.head()

# %%
df1.density.unique()

# %%
N_TANKS = len(df1)

# %% [markdown]
# ## 1.1 Reproduce Model from Textbook

# %%
with pm.Model() as model_1_1:
    pond = pm.Data('pond', df1.index.values.astype('int'))
    n = pm.Data('n', df1.density.values.astype('int'))
    survival_obs = pm.Data('survival_obs', df1.surv)
    
    a_bar = pm.Normal('a_bar', mu=0, sigma=1.5)
    sigma = pm.Exponential('sigma', lam=1)
    
    # This parametrization does not give a warning
    # a = pm.Normal('a', mu=a_bar, sigma=sigma, shape=N_TANKS)
    
    # use non-centered variables for better sampling
    # this gives me a warning, but I don't undertsand why:
    # "The number of effective samples is smaller than 25% for some parameters"
    z = pm.Normal('z', mu=0, sigma=1, shape=N_TANKS)
    a = pm.Deterministic('a', a_bar + z*sigma)
    
    p = pm.invlogit(a[pond])
    survival = pm.Binomial('survival', n=n[pond], p=p, observed=survival_obs)
    
    trace_1_1 = pm.sample(2000, tune=2000, return_inferencedata=True, random_seed=RANDOM_SEED, chains=4)

# %%
az.plot_trace(trace_1_1, var_names=['a_bar', 'sigma'])

# %%
az.summary(trace_1_1, var_names=['a_bar', 'sigma'])

# %%
fig, ax = plt.subplots(1,1)
fig.set_tight_layout(True)

ax.scatter(df1.index, df1.propsurv, label='Data')
ax.axvline(x=15.5, ls='--', c='k')
ax.axvline(x=31.5, ls='--', c='k')

# get the values for a and a_bar from the trace
a = trace_1_1.posterior['a'].mean(dim=['chain', 'draw']).values
a_bar = trace_1_1.posterior['a_bar'].mean(dim=['chain', 'draw']).values

ax.axhline(y=sigmoid(a_bar), ls='--', c='k')

ax.scatter(df1.index, sigmoid(a), marker='x', c='C1', label='Model')
ax.set_xlabel('Pond')
ax.set_ylabel('Survival Proportion')
ax.legend(loc=3)

# %%
