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
# # 1

# %%
data1 = pd.read_csv(os.path.join('data', 'NWOGrants.csv'), sep=';')

# %%
data1

# %% [markdown]
# ## Encode discipline and gender as integers

# %%
discipline_to_idx = {d: i for i, d in enumerate(data1['discipline'].unique())}

# %%
data1['discipline_id'] = data1['discipline'].apply(lambda x: discipline_to_idx[x])

# %%
data1['male'] = data1['gender'].apply(lambda x: 1 if x=='m' else 0)

# %%
data1

# %% [markdown]
# ## Total effect of gender

# %%
pm.math.invlogit(10).eval()

# %%
with pm.Model() as model_1_1:
    a = pm.Normal('a', 0, 1.5, shape=2)
    p = pm.Deterministic('p', pm.math.invlogit( a[data1['male']] ))
    awards = pm.Binomial('awards', data1['applications'], p, observed=data1['awards'])
    
    prior_1_1 = pm.sample_prior_predictive(var_names=['a', 'p'])
    
    trace_1_1 = pm.sample(1000, tune=1000, chains=4)

# %% [markdown]
# ### Prior Predictive Plots

# %%
prior_1_1['p'].shape # there are 18 rows in data1

# %%
# The piror has 18 columns, but the even columns belong to female and are all identical, while the odd columns
# belong to female and are also all identical. Below I am testing this.
for i in range(0,len(data1),2):
    print(i, np.any(prior_1_1['p'][:,0] != prior_1_1['p'][:,i]))
    print(i+1, np.any(prior_1_1['p'][:,1] != prior_1_1['p'][:,i+1]))

# %%
fig, ax = plt.subplots(1,1)

ax.hist(prior_1_1['p'][:,0], density=True, histtype='step', label='female')
ax.hist(prior_1_1['p'][:,1], density=True, histtype='step', label='male')
ax.set_xlabel('p')
ax.set_ylabel('Density')

# %% [markdown]
# ### Model Summary

# %%
pm.summary(trace_1_1, var_names=['~p'])

# %%
az.plot_forest(trace_1_1, var_names=['~p'],combined=True)

# %%
# Look at the difference in award probability and also convert from logit scale (the parameter a) to probability
# scale (the parameter p)

fig, ax = plt.subplots(1,1)

# remember that the trace for 'p' contains many identical columns. We only need 2 of them.
delta_p = trace_1_1['p'][:,data1['male']==1][:,0] - trace_1_1['p'][:,data1['male']==0][:,0]
diff_p = az.hpd(trace_1_1['p'][:,data1['male']==1][:,0] -\
                trace_1_1['p'][:,data1['male']==0][:,0],credible_interval=0.94)
hpd_delta_p = az.hpd(delta_p, credible_interval=0.89)
ax.scatter([delta_p.mean()], [1], label = 'delta p')
ax.hlines([1], hpd_delta_p[0], hpd_delta_p[1], color='C0')

delta_a = trace_1_1['a'][:,1] - trace_1_1['a'][:,0] 
hpd_delta_a = az.hpd(delta_a, credible_interval=0.94)
ax.scatter([delta_a.mean()], [1.1], label = 'delta a', color='C1')
ax.hlines([1.1], hpd_delta_a[0], hpd_delta_a[1], color='C1')

ax.legend(loc=4)

# %% [markdown]
# ## Direct effect of gender (include a backdoor through discipline)

# %%
n_discpipline = len(discipline_to_idx)
with pm.Model() as model_1_2:
    a = pm.Normal('a', 0, 1.2, shape=2)
    b = pm.Normal('b', 0, 1.2, shape=n_discpipline)
    p = pm.Deterministic('p', pm.math.invlogit( a[data1['male']] + b[data1['discipline_id']]))
    awards = pm.Binomial('awards', data1['applications'], p, observed=data1['awards'])
    
    prior_1_2 = pm.sample_prior_predictive(var_names=['a', 'p'])
    
    trace_1_2 = pm.sample(1000, tune=1000, chains=4)

# %% [markdown]
# ### Prior Predictive Plots

# %%
trace_1_2['p'].shape # again there are 18 columns because the data has 18 rows

# %%
fig, ax = plt.subplots(1,1)

ax.hist(prior_1_2['p'], density=True, histtype='step')
ax.set_xlabel('p')
ax.set_ylabel('Density')

# %% [markdown]
# ### Model Summary

# %%
pm.summary(trace_1_2, var_names=['~p'])

# %%
az.plot_forest(trace_1_1, var_names=['~p'],combined=True)

# %%
# Look at the difference in award probability and also convert from logit scale (the parameter a) to probability
# scale (the parameter p)

fig, ax = plt.subplots(1,1)

# remember that the trace for 'p' contains many identical columns. We only need 2 of them.
delta_p = trace_1_2['p'][:,data1['male']==1][:,0] - trace_1_2['p'][:,data1['male']==0][:,0]
diff_p = az.hpd(trace_1_2['p'][:,data1['male']==1][:,0] -\
                trace_1_2['p'][:,data1['male']==0][:,0],credible_interval=0.94)
hpd_delta_p = az.hpd(delta_p, credible_interval=0.89)
ax.scatter([delta_p.mean()], [1], label = 'delta p')
ax.hlines([1], hpd_delta_p[0], hpd_delta_p[1], color='C0')

delta_a = trace_1_2['a'][:,1] - trace_1_2['a'][:,0] 
hpd_delta_a = az.hpd(delta_a, credible_interval=0.94)
ax.scatter([delta_a.mean()], [1.1], label = 'delta a', color='C1')
ax.hlines([1.1], hpd_delta_a[0], hpd_delta_a[1], color='C1')

ax.legend(loc=4)

# %%
n_discpipline

# %%
fig, axes = plt.subplots(3,3, figsize=(8,8))
fig.set_tight_layout(True)
axes = axes.flatten()

for i, disc in enumerate(discipline_to_idx):
    fltr_male = (data1['discipline'] == disc) & (data1['male']==1)
    p_male = trace_1_2['p'][:,fltr_male].reshape(-1)
    
    fltr_female = (data1['discipline'] == disc) & (data1['male']==0)
    p_female = trace_1_2['p'][:,fltr_female].reshape(-1)
    
    axes[i].scatter([1],[p_male.mean()], c='C0', label='male')
    axes[i].vlines([1], az.hpd(p_male)[0], az.hpd(p_male)[1], color='C0')
    axes[i].scatter([2],[p_female.mean()], c='C1', label='female')
    axes[i].vlines([2], az.hpd(p_female)[0], az.hpd(p_female)[1], color='C1')
    
    if i==0:
        axes[i].legend(loc=8)
    
    axes[i].set_title(disc)


# %% [markdown]
# # 2

# %% [markdown]
# ## First Simulation (gender has effect in simulated data)

# %%
def sigmoid(x):
    return 1/(1 + np.exp(-x))


# %%
# Simulate Data
n_samples = 1000
seed = 569 
G = stats.bernoulli(0.5).rvs(n_samples, random_state=seed) # Gender 
C = stats.bernoulli(0.5).rvs(n_samples, random_state=seed+1) # Career Stage
D = stats.bernoulli(sigmoid(G + C)).rvs(n_samples, random_state=seed+2) # Discipline
A = stats.bernoulli(sigmoid(0.25*G + D +2.0*C-2 )).rvs(n_samples, random_state=seed+3)

with pm.Model() as model_2_1:
    g = pm.Normal('g',0,1) # gender coefficient
    d = pm.Normal('d',0,1) # discipline coefficient
    alpha = pm.Normal('alpha',0,1) # offset
    
    award = pm.Bernoulli('award', pm.math.invlogit(alpha + g*G + d*D), observed=A)
    
    trace_2_1 = pm.sample(1000, tune=1000, chains=4)
    
pm.summary(trace_2_1)

# %% [markdown]
# The estimated effect of gender (0.026) is smaller than the true effect (0.25).

# %% [markdown]
# ## Simulation 2 (gender has no effect in simulated data)

# %%
# Simulate Data
n_samples = 1000
seed = 234
G = stats.bernoulli(0.5).rvs(n_samples, random_state=seed) # Gender 
C = stats.bernoulli(0.5).rvs(n_samples, random_state=seed+1) # Career Stage
D = stats.bernoulli(sigmoid(2*G - C)).rvs(n_samples, random_state=seed+2) # Discipline
A = stats.bernoulli(sigmoid(D + C - 2 )).rvs(n_samples, random_state=seed+3)

with pm.Model() as model_2_2:
    g = pm.Normal('g',0,1) # gender coefficient
    d = pm.Normal('d',0,1) # discipline coefficient
    alpha = pm.Normal('alpha',0,1) # offset
    
    award = pm.Bernoulli('award', pm.math.invlogit(alpha + g*G + d*D), observed=A)
    
    trace_2_2 = pm.sample(1000, tune=1000, chains=4)
    
pm.summary(trace_2_2)

# %% [markdown]
# The estimated effect of gender is large, but the true effect is zero.

# %% [markdown]
# # 3

# %%
data3 = pd.read_csv(os.path.join('data', 'Primates301.csv'), sep=';')

# %%
data3.head()

# %% [markdown]
# ## Social Learning as function of brain size

# %%
data3['social_learning'].value_counts()

# %%
fltr_nan = (data3['social_learning'].isna()) | (data3['brain'].isna()) | data3['research_effort'].isna()
data3_1 = data3[['social_learning', 'brain', 'research_effort']][~fltr_nan]

data3_1['log_brain'] = np.log(data3_1['brain'])
data3_1['log_brain'] = ( data3_1['log_brain'] - data3_1['log_brain'].mean() ) / data3_1['log_brain'].std()

data3_1['log_effort'] = np.log(data3_1['research_effort'])
data3_1['log_effort'] = ( data3_1['log_effort'] - data3_1['log_effort'].mean() ) / data3_1['log_effort'].std()

# %%
with pm.Model() as model_3_1:
    a = pm.Normal('a',0,1)
    b = pm.Normal('b',0,0.5)
    lam = pm.Deterministic('lambda', pm.math.exp(a + b*data3_1['log_brain']))
    learning = pm.Poisson('social_learning', lam, observed=data3_1['social_learning'])
    
    prior_3_1 = pm.sample_prior_predictive(var_names=['a', 'lambda', 'social_learning'])
    
    trace_3_1 = pm.sample(1000, tune=1000, chains=4)

# %% [markdown]
# ### Prior Predictive Simulations

# %%
fig, ax =  plt.subplots(1,1)

ax.hist(prior_3_1['social_learning'], density=True)
ax.set_xlabel('social_learning')
ax.set_ylabel('density')

# %% [markdown]
# ### Model Summary

# %%
pm.summary(trace_3_1, var_names=['~lambda'])

# %%
ppc_3_1 = pm.sample_posterior_predictive(trace_3_1, samples=4000, model=model_3_1)

# %%
fig, axes =  plt.subplots(2,1, figsize=(8,6))
fig.set_tight_layout(True)

x = range(len(data3_1))
y = ppc_3_1['social_learning'].mean(axis=0)
y_err = ppc_3_1['social_learning'].std(axis=0)

axes[0].errorbar(x, y, yerr=y_err, ls='none', marker='+', label='model')
axes[0].scatter(x, data3_1['social_learning'], marker='x', c='C1', label='data')
axes[0].legend(loc=0)

#axes[0].set_xlabel('index of datapoint')
axes[0].set_ylabel('social learning')

axes[1].errorbar(x, y, yerr=y_err, ls='none', marker='+', label='model')
axes[1].scatter(x, data3_1['social_learning'], marker='x', c='C1', label='data')

axes[1].set_title('Zoom')
axes[1].set_ylim(-1,10)
axes[1].set_xlabel('index of datapoint')
axes[1].set_ylabel('social learning')

# %% [markdown]
# In many cases the social learning in the data is actually zero.

# %% [markdown]
# ## Include Log Research Effort

# %%
with pm.Model() as model_3_2:
    a = pm.Normal('a',0,1)
    b = pm.Normal('b',0,0.5)
    c = pm.Normal('c',0,0.5)
    lam = pm.Deterministic('lambda', pm.math.exp(a + b*data3_1['log_brain'] + c*data3_1['log_effort']))
    learning = pm.Poisson('social_learning', lam, observed=data3_1['social_learning'])
    
    prior_3_2 = pm.sample_prior_predictive(var_names=['a', 'lambda', 'social_learning'])
    
    trace_3_2 = pm.sample(1000, tune=1000, chains=4)

# %%
pm.summary(trace_3_2, var_names=['~lambda'])

# %% [markdown]
# These parameters are different from the solution, because the solution does not standardize $\log(\mathrm{effort})$. The conversion to the parameters from the solution is:

# %%
mean = np.log(data3['research_effort']).mean() 
std = np.log(data3['research_effort']).std()

print('a converted = {:.3f}'.format(trace_3_2['a'].mean() - trace_3_2['c'].mean() * mean/std))
print('c converted = {:.3f}'.format(trace_3_2['c'].mean() / std))

# %% [markdown]
# Note that these convertd parameters are closer to those from the solution:
# - $\tilde{a}$ = -5.97,
# - $\tilde{b}$ = 0.46,
# - $\tilde{c}$ = 1.53,
#
# but not identical. The reason is probably that the priors are different.

# %%
ppc_3_2 = pm.sample_posterior_predictive(trace_3_2, samples=4000, model=model_3_2)

# %%
fig, axes =  plt.subplots(2,1, figsize=(8,6))
fig.set_tight_layout(True)

x = range(len(data3_1))
y = ppc_3_2['social_learning'].mean(axis=0)
y_err = ppc_3_2['social_learning'].std(axis=0)

axes[0].errorbar(x, y, yerr=y_err, ls='none', marker='+', label='model')
axes[0].scatter(x, data3_1['social_learning'], marker='x', c='C1', label='data')
axes[0].legend(loc=0)

#axes[0].set_xlabel('index of datapoint')
axes[0].set_ylabel('social learning')

axes[1].errorbar(x, y, yerr=y_err, ls='none', marker='+', label='model')
axes[1].scatter(x, data3_1['social_learning'], marker='x', c='C1', label='data')

axes[1].set_title('Zoom')
axes[1].set_ylim(-1,10)
axes[1].set_xlabel('index of datapoint')
axes[1].set_ylabel('social learning')

# %%
waic_3_1 = pm.waic(trace_3_1, pointwise=True, scale='deviance')

# %%
waic_3_2 = pm.waic(trace_3_2, pointwise=True, scale='deviance')

# %%
fig, ax =  plt.subplots(1,1)

delta_waic = waic_3_1.waic_i.values - waic_3_2.waic_i.values
ax.scatter(delta_waic, data3_1['log_effort'])
ax.axvline(x=0,ls='--', color='k')
ax.set_xlabel('waic_3_1 - waic_3_2')
ax.set_ylabel('log research effort')

# %%
np.where(delta_waic > 30)

# %%
# The rows with large diffeerences in waic are 
data3[~fltr_nan].iloc[np.where(delta_waic > 30)[0]]

# %%
