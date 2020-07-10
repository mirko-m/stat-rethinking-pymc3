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
import theano
import os

# %% [markdown]
# # Load Data

# %%
data = pd.read_csv(os.path.join('data', 'Wines2012.csv'), delimiter=';')
data.head()

# %% [markdown]
# # 1

# %%
# Standardize variables
data['score_std'] = data['score'] / data['score'].max()

# %%
data['judge'].value_counts()

# %%
data['wine'].value_counts()

# %%
# There are 20 wines and 9 judges
num_wines = data['wine'].unique().shape[0]
num_judges = data['judge'].unique().shape[0]
print('There are {:d} wines and {:d} judges'.format(num_wines, num_judges))

# %% [markdown]
# ## Make numerical IDs for judges and wines

# %%
judge_to_idx = {judge: idx for idx, judge in enumerate(data['judge'].unique())}
idx_to_judge = {judge_to_idx[judge]: judge for judge in judge_to_idx}

# %%
wine_to_idx = {wine: idx for idx, wine in enumerate(data['wine'].unique())}
idx_to_wine = {wine_to_idx[wine]: wine for wine in wine_to_idx}

# %%
data['judge_idx'] = data['judge'].apply(lambda x: judge_to_idx[x])
data['wine_idx'] = data['wine'].apply(lambda x: wine_to_idx[x])

# %%
data['judge_idx'].value_counts()

# %%
data['wine_idx'].value_counts()

# %% [markdown]
# ## Define the Model

# %%
with pm.Model() as model_1_1:
    
    aW = pm.Normal('aW', mu=0.5, sigma=0.25, shape=num_wines) 
    aJ = pm.Normal('aJ', mu=0.0, sigma=0.25, shape=num_judges) # Judge bias is centered around zero
    
    mu = pm.Deterministic('mu', aW[data['wine_idx']] + aJ[data['judge_idx']])
    sigma = pm.Exponential('sigma', lam=1)
    
    score = pm.Normal('score', mu=mu, sigma=sigma, observed=data['score_std'])
    
    trace_1_1 = pm.sample(1000, tune=1000)

# %% [markdown]
# Note the warning about the effective sample size. This can also be seen in the summary of the model. The columns starting with ess_ are estimates of the effective sample size:

# %%
pm.summary(trace_1_1, var_names=['~mu'])

# %% [markdown]
# I can get around this issue by scaling the scores in a different way. Maybe the issue is that the scores are not normally distributed when they are scaled by dividing by the maximum.

# %%
data['score_std_2'] = ( data['score'] - data['score'].mean() ) / data['score'].std()

# %%
fig, axes = plt.subplots(1,2,sharey=True)
fig.set_tight_layout(True)

data['score_std'].hist(ax=axes[0])
data['score_std_2'].hist(ax=axes[1])

axes[0].set_xlabel('score_std')
axes[1].set_xlabel('score_std_2')
axes[0].set_ylabel('Counts')

# %%
with pm.Model() as model_1_2:
    
    aW = pm.Normal('aW', mu=0.0, sigma=0.5, shape=num_wines) 
    aJ = pm.Normal('aJ', mu=0.0, sigma=0.5, shape=num_judges)
    
    mu = pm.Deterministic('mu', aW[data['wine_idx']] + aJ[data['judge_idx']])
    sigma = pm.Exponential('sigma', lam=1)
    
    score = pm.Normal('score', mu=mu, sigma=sigma, observed=data['score_std_2'])
    
    trace_1_2 = pm.sample(1000, tune=1000)

# %%
pm.summary(trace_1_2)

# %% [markdown]
# ## Prior Samples

# %%
with model_1_2:
    prior_samples = pm.sample_prior_predictive(var_names=['aW', 'aJ', 'mu'])

# %%
prior_samples['aJ'].shape

# %%
fig, axes = plt.subplots(2,1)
fig.set_tight_layout(True)

axes[0].errorbar(range(num_judges), prior_samples['aJ'].mean(axis=0), yerr=prior_samples['aJ'].std(axis=0),
                 ls='none',marker='o')
axes[0].set_xlabel('Judge ID')
axes[0].set_ylabel('Coefficient')

axes[1].errorbar(range(num_wines), prior_samples['aW'].mean(axis=0), yerr=prior_samples['aW'].std(axis=0),
                 ls='none',marker='o')
axes[1].set_xlabel('Wine ID')
axes[1].set_ylabel('Coefficient')

# %%
prior_samples['mu'].shape

# %%
fig, ax = plt.subplots(1,1)
fig.set_tight_layout(True)

num_samples = prior_samples['aW'].shape[0]
wines_ = np.array(range(num_wines))
judges_ = np.array(range(num_judges))
mu_ = np.zeros((num_samples, num_wines*num_judges))

for i_w in range(num_wines):
    for i_j in range(num_judges):
        # mu_[:,i_w*num_judges + i_j] = prior_samples['aW'][:,i_w] + prior_samples['aJ'][:,i_j]
        mu_[:,i_w + i_j*num_wines] = prior_samples['aW'][:,i_w] + prior_samples['aJ'][:,i_j]

for i in range(10): 
    ax.scatter(range(num_judges * num_wines), mu_[i,:], c='C{:d}'.format(i%10))
    
ax.set_xlabel('Unique Wine Judge Combination')
ax.set_ylabel('Score')

# %%
# test that pymc3 gives the same result
np.sum(prior_samples['mu'] - mu_)

# %% [markdown]
# ## Look at Trace

# %% [markdown]
# I need to set compact=True, because the parameters have such a high dimension

# %%
az.plot_trace(trace_1_2, var_names=['~mu'], compact=True)

# %% [markdown]
# ## Predictions

# %%
az.plot_forest(trace_1_2, var_names=['~mu'],combined=True)

# %%
# Wines only (completely unbiased Judge)
fig, ax = plt.subplots(1,1)
fig.set_tight_layout(True)

ax.errorbar(range(num_wines), trace_1_2['aW'].mean(axis=0), yerr=trace_1_2['aW'].std(axis=0),
            ls='none', marker='o')
ax.set_xlabel('Wine ID')
ax.set_ylabel('Score (Standardized)')

# %% [markdown]
# The curve above will be shifted byt the Judge bias aJ, but the shape will no change in this model.

# %%
# Wine and Judges

fig, axes = plt.subplots(3,3, figsize=(8,8),sharex=True,sharey=True)
fig.set_tight_layout(True)
axes=axes.flatten()

num_samples = trace_1_2['aW'].shape[0]

for i in range(num_judges):
    axes[i].set_title('Judge {:d}'.format(i))
    mu_ = np.zeros((num_samples, num_wines))
    for j in range(num_wines):
        mu_[:,j] = trace_1_2['aW'][:,j] + trace_1_2['aJ'][:,i]
    axes[i].errorbar(range(num_wines), mu_.mean(axis=0), yerr=mu_.std(axis=0), ls='none')
    
    # plot the data
    fltr_judge = data['judge_idx'] == i
    axes[i].scatter(data[fltr_judge]['wine_idx'], data[fltr_judge]['score_std_2'], c='C1')
# ax.scatter(range(num_wines * num_judges), trace_1_2['mu'].mean(axis=0))
axes[7].set_xlabel('Wine ID')
axes[3].set_ylabel('Score (Standardized)')

# %% [markdown]
# # 2

# %% [markdown]
# The variables to consider are:
# - flight
# - wine.amer
# - judge.amer

# %%
data['flight'].unique()

# %%
data['flight_idx'] = data['flight'].apply(lambda x: 0 if x=='red' else 1)
data['flight_idx'].value_counts()

# %% [markdown]
# ## Indicator Variables

# %%
with pm.Model() as model_2_1:
    
    a = pm.Normal('a',mu=0,sigma=0.05)
    bF = pm.Normal('bF',mu=0,sigma=0.5)
    bW = pm.Normal('bW',mu=0,sigma=0.5)
    bJ = pm.Normal('bJ',mu=0,sigma=0.5)
    
    mu = pm.Deterministic('mu', a + bF*data['flight_idx'] + bW*data['wine.amer'] + bJ*data['judge.amer'])
    sigma = pm.Exponential('sigma', lam=1)
    
    score = pm.Normal('score', mu=mu, sigma=sigma, observed=data['score_std_2'])
    
    trace_2_1 = pm.sample(1000, tune=1000)

# %%
_ = az.plot_trace(trace_2_1, var_names=['~mu'])

# %%
pm.summary(trace_2_1, var_names=['~mu']).round(2)

# %% [markdown]
# There are $2^3$ different combinations. We can plot them all.

# %%
fig, ax = plt.subplots(1,1)
fig.set_tight_layout(True)
# axes = axes.flatten()

a_ = trace_2_1['a']
bF_ = trace_2_1['bF']
bW_ = trace_2_1['bW']
bJ_ = trace_2_1['bJ']

mean = []
std = []
combinations = []

count=0
for f, lbl_f in enumerate(['red', 'white']):
    for w in [0,1]:
        for j in [0,1]:
            mu_ = a_ + bF_*f + bW_*w + bJ_*j
            mean.append(mu_.mean(axis=0))
            std.append(mu_.std(axis=0))
            combinations.append((lbl_f,w,j))
            fltr = (data['flight_idx'] == f) & (data['wine.amer'] == w) & (data['judge.amer'] == j)
            tmp = data[fltr]
            ax.scatter([count]*len(tmp), tmp['score_std_2'])
            
            count += 1
            
            
ax.errorbar(range(8), mean, yerr=std, ls='none', marker='o', c='k')
ax.set_xticks(range(8))
ax.set_xticklabels(combinations, rotation=30)
ax.set_ylabel('Score (Standardized)')

# %% [markdown]
# The labels are (flight, wine.amer, judge.amer)

# %% [markdown]
# ## Index Variables

# %%
with pm.Model() as model_2_2:
    
    bF = pm.Normal('bF',mu=0,sigma=0.5, shape=2)
    bW = pm.Normal('bW',mu=0,sigma=0.5, shape=2)
    bJ = pm.Normal('bJ',mu=0,sigma=0.5, shape=2)
    
    mu = pm.Deterministic('mu', bF[data['flight_idx']] + bW[data['wine.amer']] + bJ[data['judge.amer']])
    sigma = pm.Exponential('sigma', lam=1)
    
    score = pm.Normal('score', mu=mu, sigma=sigma, observed=data['score_std_2'])
    
    trace_2_2 = pm.sample(1000, tune=1000)

# %%
_ = az.plot_trace(trace_2_2, var_names=['~mu'])

# %%
pm.summary(trace_2_2, var_names=['~mu']).round(2)

# %%
fig, ax = plt.subplots(1,1)
fig.set_tight_layout(True)
# axes = axes.flatten()

bF_ = trace_2_2['bF']
bW_ = trace_2_2['bW']
bJ_ = trace_2_2['bJ']

mean = []
std = []
combinations = []

count=0
for f, lbl_f in enumerate(['red', 'white']):
    for w in [0,1]:
        for j in [0,1]:
            mu_ = bF_[:,f] + bW_[:,w] + bJ_[:,j]
            mean.append(mu_.mean(axis=0))
            std.append(mu_.std(axis=0))
            combinations.append((lbl_f,w,j))
            fltr = (data['flight_idx'] == f) & (data['wine.amer'] == w) & (data['judge.amer'] == j)
            tmp = data[fltr]
            ax.scatter([count]*len(tmp), tmp['score_std_2'])
            
            count += 1
            
            
ax.errorbar(range(8), mean, yerr=std, ls='none', marker='o', c='k')
ax.set_xticks(range(8))
ax.set_xticklabels(combinations, rotation=30)
ax.set_ylabel('Score (Standardized)')

# %% [markdown]
# The labels are (flight, wine.amer, judge.amer)

# %% [markdown]
# The plot is the same as for the indicator variables.

# %% [markdown]
# # 3

# %% [markdown]
# Adding two-way interactions:
# - flight * wine.amer
# - flight * judge.amer
# - wine.amer * judge.amer

# %% [markdown]
# ## Indicator Variable Model 1

# %% [markdown]
# Using the indicator variable F which is 0 for red wine and 1 for white wines

# %%
F = theano.shared(data['flight_idx'].values)
W = theano.shared(data['wine.amer'].values)
J = theano.shared(data['judge.amer'].values)

# %%
with pm.Model() as model_3_1:
    
    a = pm.Normal('a',mu=0,sigma=0.2)
    bF = pm.Normal('bF',mu=0,sigma=0.5)
    bJ = pm.Normal('bJ',mu=0,sigma=0.5)
    bW = pm.Normal('bW',mu=0,sigma=0.5)
    
    cJF = pm.Normal('cJF',mu=0,sigma=0.25)
    cWF = pm.Normal('cWF',mu=0,sigma=0.25)
    cWJ = pm.Normal('cWJ',mu=0,sigma=0.25)
    
    mu = pm.Deterministic('mu', a + bF*F + bW*W + bJ*J +cWF*F*W + cJF*F*J + cWJ*W*J)
    sigma = pm.Exponential('sigma', lam=1)
    
    score = pm.Normal('score', mu=mu, sigma=sigma, observed=data['score_std_2'])
    
    trace_3_1 = pm.sample(1000, tune=1000)

# %%
pm.summary(trace_3_1, var_names=['~mu']).round(2)

# %% [markdown]
# ## Indicator Variable Model 2

# %% [markdown]
# Using the indicator variable R which is 1 for red wine and 0 for white wines, i.e. the orhtogonal choice to F.

# %%
R = theano.shared((data['flight_idx'] == 0).values.astype('int64'))

# %%
with pm.Model() as model_3_2:
    
    a = pm.Normal('a',mu=0,sigma=0.2)
    bR = pm.Normal('bR',mu=0,sigma=0.5)
    bJ = pm.Normal('bJ',mu=0,sigma=0.5)
    bW = pm.Normal('bW',mu=0,sigma=0.5)
    
    cJR = pm.Normal('cJR',mu=0,sigma=0.25)
    cWR = pm.Normal('cWR',mu=0,sigma=0.25)
    cWJ = pm.Normal('cWJ',mu=0,sigma=0.25)
    
    mu = pm.Deterministic('mu', a + bR*R + bW*W + bJ*J +cWR*R*W + cJR*R*J + cWJ*W*J)
    sigma = pm.Exponential('sigma', lam=1)
    
    score = pm.Normal('score', mu=mu, sigma=sigma, observed=data['score_std_2'])
    
    trace_3_2 = pm.sample(1000, tune=1000)

# %%
pm.summary(trace_3_2, var_names=['~mu']).round(2)

# %% [markdown]
# ## Compare the two models

# %%
# compare predictions
fig, axes = plt.subplots(2,1,sharex=True,sharey=True)

axes[0].scatter(trace_3_1['mu'].mean(axis=0), data['score_std_2'])
axes[0].set_ylabel('Score')

axes[1].scatter(trace_3_2['mu'].mean(axis=0), data['score_std_2'])
axes[1].set_xlabel('Precited Score')
axes[1].set_ylabel('Score')

# %% [markdown]
# The predictions are different

# %%
data.head()

# %%
ff = []
ww = []
jj = []
lbls = []
for j, lbl_j in enumerate(['F', 'A']):
    for f, lbl_f in zip([1,0], ['W', 'R']):
        for w, lbl_w in enumerate(['F', 'A']):
            ff.append(f)
            ww.append(w)
            jj.append(j)
            lbls.append(lbl_w+lbl_j+lbl_f)
rr = [1 if x==0 else 0 for x in ff]
            
print(lbls)
print(ww)
print(jj)
print(ff)
print(rr)

# %%
F.set_value(ff)
R.set_value(rr)
W.set_value(ww)
J.set_value(jj)

# %%
ppc_3_1 = pm.sample_posterior_predictive(trace_3_1, model=model_3_1, var_names=['score', 'mu'])

# %%
ppc_3_2 = pm.sample_posterior_predictive(trace_3_2, model=model_3_2, var_names=['score', 'mu'])

# %%
ppc_3_1

# %%
means = ppc_3_1['mu'].mean(axis=0)
stds = ppc_3_1['mu'].std(axis=0)
hpds = az.hpd(ppc_3_2['mu'])

fig, ax = plt.subplots(1,1)
fig.set_tight_layout(True)

y = np.array(range(8))[::-1]
for i, lbl in enumerate(lbls):
    # ax.errorbar(means, y, xerr=hpds.T, ls='none', marker='o')
    ax.scatter(means, y)
    ax.hlines(y, hpds[:,0], hpds[:,1])

ax.set_yticks(range(8))
ax.set_yticklabels(lbls[::-1])
ax.axvline(x=0,ls='--', c='C3')
ax.set_xlim(-0.7,0.7)

# %% [markdown]
# FFW means French Wine, French Judge, White Wine, i,e, the order of the labels is Wine, Judge, Flight

# %%
means = ppc_3_2['mu'].mean(axis=0)
stds = ppc_3_2['mu'].std(axis=0)
hpds = az.hpd(ppc_3_2['mu'])

fig, ax = plt.subplots(1,1)
fig.set_tight_layout(True)

y = np.array(range(8))[::-1]
for i, lbl in enumerate(lbls):
    # ax.errorbar(means, y, xerr=hpds.T, ls='none', marker='o')
    ax.scatter(means, y)
    ax.hlines(y, hpds[:,0], hpds[:,1])

ax.set_yticks(range(8))
ax.set_yticklabels(lbls[::-1])
ax.axvline(x=0,ls='--', c='C3')
ax.set_xlim(-0.7,0.7)

# %% [markdown]
# FFW means French Wine, French Judge, White Wine

# %%
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,6))
az.plot_forest(trace_3_1, var_names=['~mu', '~sigma'], combined=True, figsize=(6,4), ax=ax1)
ax1.axvline(0,c='k', ls='--')
ax1.set_title('Model Using F')

az.plot_forest(trace_3_2, var_names=['~mu', '~sigma'], combined=True, figsize=(6,4), ax=ax2)
ax2.axvline(0,c='k', ls='--')
ax2.axvline(0,c='k', ls='--')
ax2.set_title('Model Using R')

# %% [markdown]
# The choice of how to encode the wine flight makes a difference for the parameter values. It also makes a small difference for the predictions.

# %%
os.system('jupyter nbconvert --to html week05.ipynb')

# %%
