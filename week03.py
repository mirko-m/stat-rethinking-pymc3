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
import pandas as pd
import os

# %% [markdown]
# # Load and Standardize Data

# %%
data = pd.read_csv(os.path.join('data', 'foxes.csv'), delimiter=';')

# %%
data.head()

# %%
data['avgfood_sd'] = (data['avgfood'] - data['avgfood'].mean()) / data['avgfood'].std()
data['groupsize_sd'] = (data['groupsize'] - data['groupsize'].mean()) / data['groupsize'].std()
data['area_sd'] = (data['area'] - data['area'].mean()) / data['area'].std()
data['weight_sd'] = (data['weight'] - data['weight'].mean()) / data['weight'].std()

# %%
len(data)

# %% [markdown]
# # 1.  Total Causal Influence of Area on Weight

# %% [markdown]
# ## Prior Predictive Simulation

# %%
# What is the standard deviation of area and weight?
data['area'].std(), data['weight'].std()

# %%
n_samples = 20
alpha = stats.norm(loc=0, scale=0.02).rvs(size=n_samples)
beta_a = stats.norm(loc=0, scale=0.5).rvs(size=n_samples)
mu = alpha + np.outer(data['area_sd'].values, beta_a)

fig, ax = plt.subplots(1,1)
ax.plot(data['area_sd'], mu)
ax.set_xlabel('area (sd)')
ax.set_ylabel('weight (sd)')

# %% [markdown]
# ## Linear Regression

# %%
with pm.Model() as model_1:
    alpha_1 = pm.Normal('alpha_1', mu=0, sigma=0.02)
    beta_1 = pm.Normal('beta_1', mu=0, sigma=0.5)
    sigma_1 = pm.HalfNormal('sigma_1', sigma=5)
    
    mu = pm.Deterministic('mu', alpha_1 + beta_1*data['area_sd'])
    weight_sd = pm.Normal('weight', mu=mu, sigma=sigma_1, observed=data['weight_sd'])
    trace_1 = pm.sample(1000, tune=1000)

# %%
_ = az.plot_trace(trace_1,figsize=(8,6), var_names=['~mu'])

# %%
az.summary(trace_1, var_names=['~mu'], credible_interval=0.89)

# %%
fig, ax = plt.subplots(1,1)
fig.set_tight_layout(True)
ax.plot(data['area_sd'], data['weight_sd'], 'x')
ax.plot(data['area_sd'], trace_1['mu'].mean(axis=0))
az.plot_hpd(data['area_sd'], trace_1['mu'], credible_interval=0.89, ax=ax)

ax.set_xlabel('area (sd)')
ax.set_ylabel('weight (sd)')

# %% [markdown]
# Note that the plot above shows the credible interval of the mean.

# %% [markdown]
# ## Add Groupsize as a Varibale

# %%
with pm.Model() as model_1_2:
    alpha_1 = pm.Normal('alpha_1', mu=0, sigma=0.02)
    beta_1 = pm.Normal('beta_1', mu=0, sigma=0.5, shape=2)
    sigma_1 = pm.HalfNormal('sigma_1', sigma=5)
    
    mu = pm.Deterministic('mu', alpha_1 + beta_1[0]*data['area_sd'] + beta_1[1]*data['groupsize_sd'])
    weight_sd = pm.Normal('weight', mu=mu, sigma=sigma_1, observed=data['weight_sd'])
    trace_1_2 = pm.sample(1000, tune=1000)

# %%
_ = az.plot_trace(trace_1_2,figsize=(8,6), var_names=['~mu'])

# %%
az.summary(trace_1_2, var_names=['~mu'], credible_interval=0.89)

# %% [markdown]
# Including `groupsize` in the model leads to a positve effect of `area` on weight. Note that the `are` indirectly affects group size through `avgfood`.

# %% [markdown]
# # 2 Causal Impact of Adding Food to a Territory

# %% [markdown]
# ## Prior Predictive Simulation

# %%
# What is the standard deviation of avgfood and weight?
data['avgfood'].std(), data['weight'].std()

# %%
n_samples = 20
alpha = stats.norm(loc=0, scale=0.02).rvs(size=n_samples)
beta = stats.norm(loc=0, scale=0.3).rvs(size=n_samples)
mu = alpha + np.outer(data['avgfood_sd'].values, beta)

fig, ax = plt.subplots(1,1)
ax.plot(data['avgfood_sd'], mu)
ax.set_xlabel('area (sd)')
ax.set_ylabel('weight (sd)')

# %% [markdown]
# ## Linear Regression

# %%
with pm.Model() as model_2:
    alpha_2 = pm.Normal('alpha_2', mu=0, sigma=0.02)
    beta_2 = pm.Normal('beta_2', mu=0, sigma=0.3)
    sigma_2 = pm.HalfNormal('sigma_2', sigma=5)
    
    mu = pm.Deterministic('mu', alpha_2 + beta_2*data['avgfood_sd'])
    weight_sd = pm.Normal('weight', mu=mu, sigma=sigma_2, observed=data['weight_sd'])
    trace_2 = pm.sample(1000, tune=1000)

# %%
_ = az.plot_trace(trace_2,figsize=(8,6), var_names=['~mu'])

# %%
az.summary(trace_2, var_names=['~mu'], credible_interval=0.89)

# %%
fig, ax = plt.subplots(1,1)
fig.set_tight_layout(True)
ax.plot(data['avgfood_sd'], data['weight_sd'], 'x')
ax.plot(data['avgfood_sd'], trace_2['mu'].mean(axis=0))
az.plot_hpd(data['avgfood_sd'], trace_2['mu'], credible_interval=0.89, ax=ax)

ax.set_xlabel('avgfood (sd)')
ax.set_ylabel('weight (sd)')

# %% [markdown]
# # 3 Causal Impact of Group Size

# %% [markdown]
# ## Prior Predictive Simulation

# %%
# What is the standard deviation of avgfood, goupsize and weight?
data['avgfood'].std(), data['groupsize'].std(), data['weight'].std()

# %%
n_samples = 20
alpha = stats.norm(loc=0, scale=0.02).rvs(size=n_samples)
beta = stats.norm(loc=0, scale=0.3).rvs(size=n_samples)
mu = alpha + np.outer(data['groupsize_sd'].values, beta) 

fig, ax = plt.subplots(1,1)
ax.plot(data['groupsize_sd'], mu)
ax.set_xlabel('groupsize (sd)')
ax.set_ylabel('weight (sd)')

# %% [markdown]
# ## Linear Regression

# %%
with pm.Model() as model_3:
    alpha_3 = pm.Normal('alpha_3', mu=0, sigma=0.02)
    beta_3 = pm.Normal('beta_3', mu=0, sigma=0.5, shape=2)
    sigma_3 = pm.HalfNormal('sigma_3', sigma=5)
    
    mu = pm.Deterministic('mu', alpha_3 + beta_3[0]*data['avgfood_sd'] + beta_3[1]*data['groupsize_sd'])
    weight_sd = pm.Normal('weight', mu=mu, sigma=sigma_3, observed=data['weight_sd'])
    trace_3 = pm.sample(1000, tune=1000)

# %%
_ = az.plot_trace(trace_3,figsize=(8,6), var_names=['~mu'])

# %%
az.summary(trace_3, var_names=['~mu'], credible_interval=0.89)

# %%
data['groupsize'].value_counts()

# %%
fig, axes = plt.subplots(4,2, figsize=(8,6), sharex=True, sharey=True)
axes = axes.flatten()
fig.set_tight_layout(True)

xx = np.linspace(-2,2.5,num=100)
for i, gs in enumerate(np.sort(data['groupsize'].unique())):
    fltr = data['groupsize']==gs
    axes[i].plot(data['avgfood_sd'][fltr], data['weight_sd'][fltr], 'xC{:d}'.format(i),
            label='groupsize={:d}'.format(gs))
    
    mu = trace_3['alpha_3'] + np.outer(xx, trace_3['beta_3'][:,0]) +\
         np.outer([gs-data['groupsize'].mean()]*xx.shape[0], trace_3['beta_3'][:,1])
    axes[i].plot(xx, mu.mean(axis=1), '-k')
    
    axes[i].legend(loc=0)
    axes[i].set_xlabel('avgfood (sd)')
    axes[i].set_ylabel('weight (sd)')

# %%
# We can see that avgfood and groupsize are correlated.
fig, ax = plt.subplots(1,1)
fig.set_tight_layout(True)
ax.plot(data['avgfood_sd'], data['groupsize_sd'], 'x')
ax.set_xlabel('avgfood (sd)')
ax.set_ylabel('groupsize (sd)')

# %% [markdown]
# # Summary

# %% [markdown]
# What happens here is that `groupsize` and `avgfood` have opposite effects on `weight`. An increase in `avgfood` causes an increase in `weight`. Conversely an increase in `groupsize` causes a decrease in `weight`. However, `avgfood` also causes an increase in `groupsize` so when `groupsize` is not included in the model the direct effect of `avgfood` and the indirect effect of `avgfood` cancel out. 

# %%
