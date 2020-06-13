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
import pandas as pd
import os

# %%
data = pd.read_csv(os.path.join('data', 'Howell1.csv'), sep=';')

# %%
data.head()

# %% [markdown]
# # Filter out Kids and Plot

# %%
adults = data[data['age'] > 18]

# %%
fig, ax = plt.subplots(1,1)
fig.set_tight_layout(True)

ax.plot(adults['weight'], adults['height'], 'xC0')
ax.set_xlabel('Weight')
ax.set_ylabel('Height')

# %% [markdown]
# # 1. Predict Height

# %%
adult_height = adults['height'].values
adult_weight = adults['weight'].values

# normalize the weight
mean_adult_weight = np.mean(adult_weight)
adult_weight = adult_weight - mean_adult_weight

# %%
linear_model = pm.Model()

with linear_model:
    
    alpha = pm.Normal('alpha', mu=178, sigma=20)
    beta = pm.Lognormal('beta', mu=0, sigma=1)
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    
    mu = alpha + beta*adult_weight
    
    height_obs = pm.Normal('height_obs', mu=mu, sigma=sigma, observed=adult_height)

# %%
map_estimate = pm.find_MAP(model=linear_model)

# %%
map_estimate

# %%
# compare alpha to mean
adult_height.mean(), map_estimate['alpha']

# %%
# the hessian can be used to calculate the curvature of the posterior (quadratic approximation)
cov = pm.find_hessian(map_estimate, model=linear_model)

# %%
cov

# %%
cov_inv = np.linalg.inv(cov)

# %%
mu = np.array([map_estimate['alpha'], map_estimate['beta'], map_estimate['sigma']])

# %%
post = pm.MvNormal.dist(mu = mu, cov=cov_inv)

# %%
fig, ax = plt.subplots(1,1)
fig.set_tight_layout(True)

weight_vals = np.linspace(-14,18,num=100)
ax.plot(adults['weight'], adults['height'], 'xC0')
samples = post.random(size=20)
for i in range(samples.shape[0]):
    a = samples[i,0]
    b = samples[i,1]
    ax.plot(weight_vals + mean_adult_weight, a + b*weight_vals, '-C2', alpha=0.2)
ax.set_xlabel('Weight')
ax.set_ylabel('Height')

# %%
# make predictions to complete the table from the assignment
target_weights = np.array([45, 40, 65, 31, 53])
target_heights = map_estimate['alpha'] + map_estimate['beta'] * (target_weights - mean_adult_weight )
for i, h in enumerate(target_heights):
    print(i, h)

# %%
# get the 89% inteval
q = (1 - 0.89)/2
height_vals = np.linspace(0,300,num=1000)

lower_bound = []
upper_bound = []
for i in range(target_heights.shape[0]):
    tmp = pm.Normal.dist(mu=target_heights[i], sigma=map_estimate['sigma'])
    logcdf = tmp.logcdf(height_vals).eval()
    idx_upper = np.where(logcdf > np.log(1-q))[0].min()
    idx_lower = np.where(logcdf < np.log(q))[0].max()
    
    lower_bound.append(height_vals[idx_lower])
    upper_bound.append(height_vals[idx_upper])

    print(height_vals[idx_lower], height_vals[idx_upper])

# %%
for i in range(target_heights.shape[0]):
    print(i, target_weights[i], target_heights[i], lower_bound[i], upper_bound[i])

# %% [markdown]
# # 2. Entire Dataset Using Log Transform

# %%
height = data['height'].values
log_weight = np.log(data['weight'].values)

# normalize the weight
mean_log_weight = np.mean(log_weight)
log_weight = log_weight - mean_log_weight

# %%
fig, ax = plt.subplots(1,1)
fig.set_tight_layout(True)

ax.plot(log_weight, height, 'xC0')
ax.set_xlabel('Log Weight')
ax.set_ylabel('Height')

# %%
log_model = pm.Model()

with log_model:
    
    alpha = pm.Normal('alpha', mu=178, sigma=20)
    beta = pm.Lognormal('beta', mu=0, sigma=1)
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    
    mu = alpha + beta*log_weight
    
    height_obs = pm.Normal('height_obs', mu=mu, sigma=sigma, observed=height)

# %%
map_estimate_log = pm.find_MAP(model=log_model)

# %%
map_estimate_log

# %%
mu_log = np.array([map_estimate_log['alpha'], map_estimate_log['beta'], map_estimate_log['sigma']])

# %%
# the hessian can be used to calculate the curvature of the posterior (quadratic approximation)
cov_log = pm.find_hessian(map_estimate_log, model=log_model)
cov_log = np.linalg.inv(cov_log)

# %%
post_log = pm.MvNormal.dist(mu = mu_log, cov=cov_log)

# %%
# plot using log scale
fig, ax = plt.subplots(1,1)
fig.set_tight_layout(True)

log_weight_vals = np.linspace(-2,0.75,num=100)
ax.plot(log_weight, height, 'xC0')
samples = post_log.random(size=100)
for i in range(samples.shape[0]):
    a = samples[i,0]
    b = samples[i,1]
    ax.plot(log_weight_vals, a + b*log_weight_vals, '-C1', alpha=0.2)
ax.set_xlabel('Log Weight')
ax.set_ylabel('Height')

# %%
# plot after converting back
fig, ax = plt.subplots(1,1)
fig.set_tight_layout(True)

log_weight_vals = np.linspace(-2,0.75,num=100)
ax.plot(np.exp(log_weight+mean_log_weight), height, 'xC0')
samples = post_log.random(size=1000)

h = []
for i in range(samples.shape[0]):
    
    a = samples[i,0]
    b = samples[i,1]
    s = samples[i,2]
    
    mu = a + b*log_weight_vals
    h.append(pm.Normal.dist(mu=mu, sigma=s).random().tolist())
    # hpd = pm.stats.hpd(h)
    
    ax.plot(np.exp(log_weight_vals+mean_log_weight), mu, '-C1', alpha=0.2)
    # ax.fill_between(np.exp(log_weight_vals+mean_log_weight), hpd[:,0], hpd[:,1], color='C2', alpha=0.2)

h = np.array(h)
hpd = pm.stats.hpd(h,credible_interval=0.99)
ax.fill_between(np.exp(log_weight_vals+mean_log_weight), hpd[:,0], hpd[:,1], color='C2', alpha=0.2)
    
ax.set_xlabel('Weight')
ax.set_ylabel('Height')

# %%
