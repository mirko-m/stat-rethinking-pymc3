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
# # 1 Measurement Errors

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

# %% [markdown]
# # 2 Missing Values

# %%
df.isna().sum(axis=0)

# %%
# Only consider species with measured body mass
fltr = df['body'].isna()
df_2 = df[~fltr].copy()
df_2['body_norm'] = df_2['body'] / df_2['body'].max()
df_2['brain_norm'] = df_2['brain'] / df_2['brain'].max()

print('Number of species: {:d}'.format(len(df_2)))
print('Number of missing brain values {:d}'.format(df_2['brain_norm'].isna().sum()))

# %%
fig, ax = plt.subplots(1,1)
fig.set_tight_layout(True)

ax.scatter(df_2['body_norm'], df_2['brain_norm'].isna())
ax.set_xlabel('Body (normalized)')
ax.set_ylabel('Brain Value missing')
ax.set_yticks([0,1])
ax.set_yticklabels(['False', 'True'])

# %% [markdown]
# It appears that the brain values are mainly missing for species with small body mass. Presumably they were not measured for these species.

# %% [markdown]
# ## Omit the missing brain values

# %%
with pm.Model() as model_2_1:
    
    # data
    fltr = df_2['brain_norm'].isna()
    tmp = df_2[~fltr].copy()
    B = pm.Data('B', tmp['brain_norm'])
    M = pm.Data('M', tmp['body_norm'])
    
    # parameters
    a = pm.Normal('a', mu=0, sd=1)
    b = pm.Normal('b', mu=0, sd=1)
    sigma = pm.Exponential('sigma', lam=1)
    
    # model
    B_obs = pm.Lognormal('B_obs', mu = a + b*pm.math.log(M), sd=sigma, observed=B)
    
    # sampling
    trace_2_1 = pm.sample(tune=2000, draws=1000, return_inferencedata=True, random_seed=RANDOM_SEED)
    ppc_2_1 = pm.sample_posterior_predictive(trace_2_1, var_names=['B_obs', 'a', 'b'],
                                             random_seed=RANDOM_SEED)
    
ppc_2_1 = az.from_pymc3(posterior_predictive=ppc_2_1, model=model_2_1)

# %%
az.summary(trace_2_1)

# %% [markdown]
# ## Impute the missing brain values

# %%
num_missing = df_2['brain_norm'].isna().sum()

# %%
with pm.Model() as model_2_2:
    
    # Use a masked array for the missing values. This should cause
    # pymc3 to automatically impute them
    # B = pm.Data('B', np.ma.masked_invalid(df_2['brain_norm'].values)) # Does not work
    B = np.ma.masked_invalid(df_2['brain_norm'].values)
    M = pm.Data('M', df_2['body_norm'])
    
    # parameters
    a = pm.Normal('a', mu=0, sd=1)
    b = pm.Normal('b', mu=0, sd=1)
    sigma = pm.Exponential('sigma', lam=1)
    
    # model
    B_obs = pm.Lognormal('B_obs', mu = a + b*pm.math.log(M), sd=sigma, observed=B)
    
    # sampling
    trace_2_2 = pm.sample(tune=2000, draws=1000,
                          start = {'B_obs_missing': [0.5]*num_missing},
                          return_inferencedata=True, random_seed=RANDOM_SEED,
                          target_accept=0.9)
    ppc_2_2 = pm.sample_posterior_predictive(trace_2_2, var_names=['B_obs', 'B_obs_missing', 'a', 'b',],
                                             random_seed=RANDOM_SEED)
    
ppc_2_2 = az.from_pymc3(posterior_predictive=ppc_2_2, model=model_2_2)

# %%
az.summary(trace_2_2, var_names=['~B_obs_missing'])

# %% [markdown]
# What do the imputed values look like?

# %%
fig, ax = plt.subplots(1,1)
fig.set_tight_layout(True)

fltr = df_2['brain_norm'].isna()

b_missing = ppc_2_2.posterior_predictive.B_obs_missing.mean(dim=('chain', 'draw')).values

# calculate hdi and convert it to work with matplotlibs errorbar
hdi = az.hdi(ppc_2_2.posterior_predictive.B_obs_missing).B_obs_missing.values
y_low = -hdi[:,0] + b_missing
y_high = hdi[:,1] - b_missing
yerr = np.vstack((y_low, y_high))

ax.scatter(df_2['body_norm'][~fltr], df_2['brain_norm'][~fltr], c='C0', label='observed')
ax.errorbar(df_2['body_norm'][fltr], b_missing, yerr=yerr, c='C1', label='missing', ls='none', marker='o')
ax.legend(loc=0)
ax.set_xlabel('Body (norm)')
ax.set_ylabel('Brain (norm)')

# %% [markdown]
# Imputing the values does not make a big difference, as can be seen when comparing the parameters for model_2_1 and model_2_2. This is because most of the missing values occur for small body mass, where we have a lot of data already.

# %%
fig, axes = plt.subplots(2,1, sharex=True)
fig.set_tight_layout(True)
axes = axes.flatten()

az.plot_forest(trace_2_1, var_names=['a', 'b', 'sigma'], combined=True, model_names=['missing'], ax=axes[0])
az.plot_forest(trace_2_2, var_names=['a', 'b', 'sigma'], combined=True, model_names=['imputed'], ax=axes[1])

# %% [markdown]
# Let's also have a look at the model predictions

# %%
fig, axes = plt.subplots(1,2, sharex=True, sharey=True)
fig.set_tight_layout(True)
axes = axes.flatten()

xx = np.linspace(0,1)

# model_2_1: no imputation
aa_1 = ppc_2_1.posterior_predictive.a.values
bb_1 = ppc_2_1.posterior_predictive.b.values
mu_1 = aa_1 + bb_1*np.log(xx[:,None])
yy_1 = np.exp(mu_1)
hdi_1 = az.hdi(yy_1.T) # Note that I need to transpose yy for arviz to interpret the shape correctly

axes[0].scatter(df_2['body_norm'], df_2['brain_norm'], c='C0', label='data')
axes[0].plot(xx, yy_1.mean(axis=1), '-C1', label='model mean')
axes[0].fill_between(xx, hdi_1[:,0], hdi_1[:,1], color='C1', alpha=0.2)

axes[0].set_xlabel('Body (norm)')
axes[0].set_ylabel('Brain (norm)')
axes[0].set_title('No Imputation')
axes[0].legend(loc=0)

# model_2_2: with imputation
aa_2 = ppc_2_2.posterior_predictive.a.values
bb_2 = ppc_2_2.posterior_predictive.b.values
mu_2 = aa_2 + bb_2*np.log(xx[:,None])
yy_2 = np.exp(mu_2)
hdi_2 = az.hdi(yy_2.T) # Note that I need to transpose yy for arviz to interpret the shape correctly

b_missing = ppc_2_2.posterior_predictive.B_obs_missing.mean(dim=('chain', 'draw')).values
fltr = df_2['brain_norm'].isna()

axes[1].scatter(df_2['body_norm'], df_2['brain_norm'], c='C0', label='observed')
axes[1].scatter(df_2['body_norm'][fltr], b_missing, c='C2', label='imputed')
axes[1].plot(xx, yy_2.mean(axis=1), '-C1', label='model mean')

axes[1].fill_between(xx, hdi_2[:,0], hdi_2[:,1], color='C1', alpha=0.2)
axes[1].set_xlabel('Body (norm)')
axes[1].set_ylabel('Brain (norm)')
axes[1].set_title('Imputation')
axes[1].legend(loc=0)
