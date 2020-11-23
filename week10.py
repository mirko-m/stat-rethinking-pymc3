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
# %load_ext watermark

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pymc3 as pm
import arviz as az
import xarray
import os

# %%
# %watermark --iversion

# %%
RANDOM_SEED = 76003

# %% [markdown]
# # 1

# %%
df = pd.read_csv(os.path.join('data', 'Primates301.csv'), sep=';')

# %%
df.head()

# %%
# get rid of NaNs in brain size and body mass
fltr_nan = (~df.brain.isna()) & (~df.body.isna())
df_clean = df[fltr_nan].copy() # copy is to get rid of pandas warning

# %%
# Normalize
df_clean['brain_norm'] = df_clean.brain / df_clean.brain.max()
df_clean['body_norm'] = df_clean.body / df_clean.body.max()

# %%
# Come up with errors
df_clean['brain_se'] = df_clean['brain_norm'] * 0.1
df_clean['body_se'] = df_clean['body_norm'] * 0.1

# %%
with pm.Model() as model_1_1:
    # data
    B = pm.Data('B', df_clean['brain_norm'])
    M = pm.Data('M', df_clean['body_norm'])
    
    # parameters
    a = pm.Normal('a', mu=0, sd=1)
    b = pm.Normal('b', mu=0, sd=1)
    sigma = pm.Exponential('sigma', lam=1)
    
    # model
    mu = pm.Deterministic('mu', a + b*pm.math.log(M))
    B_obs = pm.Lognormal('B_obs', mu=mu, sigma=sigma, observed=B)
    
    # sampling
    trace_1_1 = pm.sample(tune=2000, draws=1000, return_inferencedata=True, random_seed=RANDOM_SEED)
    ppc_1_1 = pm.sample_posterior_predictive(trace_1_1, var_names=['B_obs', 'a', 'b'],
                                             random_seed=RANDOM_SEED)

ppc_1_1 = az.from_pymc3(posterior_predictive=ppc_1_1, model=model_1_1)

# %%
az.summary(trace_1_1, var_names=['~mu'])

# %%
fig, ax = plt.subplots()
fig.set_tight_layout(True)

xx = xarray.DataArray(df_clean['body_norm'].values, dims='B_obs_dim_0')
yy = ppc_1_1.posterior_predictive.B_obs.mean(dim=('chain', 'draw'))
hdi = az.hdi(ppc_1_1.posterior_predictive.B_obs).B_obs

ax.scatter(df_clean['body_norm'], df_clean['brain_norm'], label='data')
ax.plot(xx.sortby(xx), yy.sortby(xx), color='C1', label='model outcome')
ax.fill_between(xx.sortby(xx), hdi.sortby(xx)[:,0], hdi.sortby(xx)[:,1], alpha=0.2, color='C1')

ax.legend(loc=0)
ax.set_xlabel('Body Mass')
ax.set_ylabel('Brain Size')

# %% [markdown]
# Note the kink in the plot close to a Body Mass of 0.3. I can get rid of this as follows:

# %%
fig, ax = plt.subplots()
fig.set_tight_layout(True)

xx = np.linspace(0,1)

aa = ppc_1_1.posterior_predictive.a.values
bb = ppc_1_1.posterior_predictive.b.values
mu = np.exp(aa + bb*np.log(xx[:,None]))
hdi_mu = az.hdi(mu.T) # Note that I need to transpose mu for arviz to interpret the shape correctly

ax.scatter(df_clean['body_norm'], df_clean['brain_norm'], label='data')
ax.plot(xx, mu.mean(axis=1), color='C1', label='model mean')
ax.fill_between(xx, hdi_mu[:,0], hdi_mu[:,1], alpha=0.2, color='C1')

ax.legend(loc=0)
ax.set_xlabel('Body Mass')
ax.set_ylabel('Brain Size')

# %%
num_samples = len(df_clean)

# %%
with pm.Model() as model_1_2:
    # data
    B = pm.Data('B', df_clean['brain_norm'])
    M = pm.Data('M', df_clean['body_norm'])
    B_se = pm.Data('B_se', df_clean['brain_se'])
    M_se = pm.Data('M_se', df_clean['body_se'])
    
    # parameters
    a = pm.Normal('a', mu=0, sd=1)
    b = pm.Normal('b', mu=0, sd=1)
    sigma = pm.Exponential('sigma', lam=1)
    
    # model
    M_true = pm.Normal('M_true', mu=0.5, sd=0.5, shape=num_samples)
    M_obs = pm.Normal('M_obs', mu=M_true, sd=M_se, observed=M)
    B_true = pm.Lognormal('B_true', mu = a + b*pm.math.log(M_true), sd=sigma, shape=num_samples)
    B_obs = pm.Normal('B_obs', mu=B_true, sd=B_se, shape=num_samples, observed=B)
    
    # sampling
    trace_1_2 = pm.sample(tune=1000, draws=1000, return_inferencedata=True,
                          start={'M_true': df_clean['body_norm'].values,
                                 'B_true': df_clean['brain_norm'].values},
                          random_seed=RANDOM_SEED)
    ppc_1_2 = pm.sample_posterior_predictive(trace_1_2,var_names=['B_obs', 'a', 'b'],
                                             random_seed=RANDOM_SEED)
    
ppc_1_2 = az.from_pymc3(posterior_predictive=ppc_1_2, model=model_1_2)

# %%
az.summary(trace_1_2, var_names=['~M_true', '~B_true'])

# %%
fig, ax = plt.subplots()
fig.set_tight_layout(True)

ax.scatter(df_clean['body_norm'], df_clean['brain_norm'], facecolors='none', edgecolors='C0',
           label='observed')
ax.scatter(trace_1_2.posterior.M_true.mean(dim=('chain', 'draw')),
           trace_1_2.posterior.B_true.mean(dim=('chain', 'draw')),
           facecolors='none', edgecolors='C1', label='model')

ax.legend(loc=0)
ax.set_xlabel('Body Mass')
ax.set_ylabel('Brain Size')

# %% [markdown]
# Adding errors to the model only has a very small effect on the "true" values for body mass and brain size.

# %%
fig, axes = plt.subplots(1,2, sharex=True, sharey=True)
fig.set_tight_layout(True)
axes = axes.flatten()

# model_1_1
xx = np.linspace(0,1)

aa_1 = ppc_1_1.posterior_predictive.a.values
bb_1 = ppc_1_1.posterior_predictive.b.values
mu_1 = np.exp(aa_1 + bb_1*np.log(xx[:,None]))
hdi_mu_1 = az.hdi(mu_1.T) # Note that I need to transpose mu for arviz to interpret the shape correctly

axes[0].scatter(df_clean['body_norm'], df_clean['brain_norm'], label='data')
axes[0].plot(xx, mu_1.mean(axis=1), color='C1', label='model mean')
axes[0].fill_between(xx, hdi_mu_1[:,0], hdi_mu_1[:,1], alpha=0.2, color='C1')

axes[0].legend(loc=2)
axes[0].set_xlabel('Body Mass')
axes[0].set_ylabel('Brain Size')


# model_1_2
aa_2 = ppc_1_2.posterior_predictive.a.values
bb_2 = ppc_1_2.posterior_predictive.b.values
mu_2 = np.exp(aa_2 + bb_2*np.log(xx[:,None]))
hdi_mu_2 = az.hdi(mu_2.T) # Note that I need to transpose mu for arviz to interpret the shape correctly

axes[1].scatter(df_clean['body_norm'], df_clean['brain_norm'], label='data')
axes[1].plot(xx, mu_2.mean(axis=1), color='C1', label='model mean')
axes[1].fill_between(xx, hdi_mu_2[:,0], hdi_mu_2[:,1], alpha=0.2, color='C1')

axes[1].legend(loc=2)
axes[1].set_xlabel('Body Mass')
axes[1].set_ylabel('Brain Size')

# %%
