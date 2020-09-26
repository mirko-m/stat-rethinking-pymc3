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

# %% [markdown]
# ## Adding Predictors

# %% [markdown]
# ### 1.2 Model with Predation

# %%
df1.pred.unique()

# %%
df1['pred_cat'] = df1.pred.apply(lambda x: 1 if x=='pred' else 0)

# %%
with pm.Model() as model_1_2:
    pond = pm.Data('pond', df1.index.values.astype('int'))
    n = pm.Data('n', df1.density.values.astype('int'))
    survival_obs = pm.Data('survival_obs', df1.surv)
    pred = pm.Data('pred', df1.pred_cat)
    
    a_bar = pm.Normal('a_bar', mu=0, sigma=1.5)
    sigma = pm.Exponential('sigma', lam=1)
    
    # This parametrization does not give a warning
    # a = pm.Normal('a', mu=a_bar, sigma=sigma, shape=N_TANKS)
    
    # use non-centered variables for better sampling
    # this gives me a warning, but I don't undertsand why:
    # "The number of effective samples is smaller than 25% for some parameters"
    z = pm.Normal('z', mu=0, sigma=1, shape=N_TANKS)
    a = pm.Deterministic('a', a_bar + z*sigma)
    
    b_p = pm.Normal('b_p', mu=-0.5, sigma=0.5) # expect predation to have negative effect on survival
    
    p = pm.invlogit(a[pond] + b_p*pred)
    survival = pm.Binomial('survival', n=n[pond], p=p, observed=survival_obs)
    
    trace_1_2 = pm.sample(2000, tune=2000, return_inferencedata=True, random_seed=RANDOM_SEED, chains=4)

# %%
az.summary(trace_1_2, var_names=['a_bar', 'sigma', 'b_p'])

# %% [markdown]
# ### 1.3 Model with size

# %%
df1['size'].unique()

# %%
df1['big'] = df1['size'].apply(lambda x: 1 if x=='big' else 0)

# %%
with pm.Model() as model_1_3:
    pond = pm.Data('pond', df1.index.values.astype('int'))
    n = pm.Data('n', df1.density.values.astype('int'))
    survival_obs = pm.Data('survival_obs', df1.surv)
    big = pm.Data('big', df1.big)
    
    a_bar = pm.Normal('a_bar', mu=0, sigma=1.5)
    sigma = pm.Exponential('sigma', lam=1)
    
    # This parametrization does not give a warning
    # a = pm.Normal('a', mu=a_bar, sigma=sigma, shape=N_TANKS)
    
    # use non-centered variables for better sampling
    # this gives me a warning, but I don't undertsand why:
    # "The number of effective samples is smaller than 25% for some parameters"
    z = pm.Normal('z', mu=0, sigma=1, shape=N_TANKS)
    a = pm.Deterministic('a', a_bar + z*sigma)
    
    b_b = pm.Normal('b_b', mu=0, sigma=0.5) 
    
    p = pm.invlogit(a[pond] + b_b*big)
    survival = pm.Binomial('survival', n=n[pond], p=p, observed=survival_obs)
    
    trace_1_3 = pm.sample(2000, tune=2000, return_inferencedata=True, random_seed=RANDOM_SEED, chains=4)

# %%
az.summary(trace_1_3, var_names=['a_bar', 'sigma', 'b_b'])

# %% [markdown]
# ### 1.4 Model with size and predation

# %%
with pm.Model() as model_1_4:
    pond = pm.Data('pond', df1.index.values.astype('int'))
    n = pm.Data('n', df1.density.values.astype('int'))
    survival_obs = pm.Data('survival_obs', df1.surv)
    pred = pm.Data('pred', df1.pred_cat)
    big = pm.Data('big', df1.big)
    
    a_bar = pm.Normal('a_bar', mu=0, sigma=1.5)
    sigma = pm.Exponential('sigma', lam=1)
    
    # This parametrization does not give a warning
    # a = pm.Normal('a', mu=a_bar, sigma=sigma, shape=N_TANKS)
    
    # use non-centered variables for better sampling
    # this gives me a warning, but I don't undertsand why:
    # "The number of effective samples is smaller than 25% for some parameters"
    z = pm.Normal('z', mu=0, sigma=1, shape=N_TANKS)
    a = pm.Deterministic('a', a_bar + z*sigma)
    
    b_p = pm.Normal('b_p', mu=-0.5, sigma=0.5) # expect predation to have negative effect on survival
    b_b = pm.Normal('b_b', mu=0, sigma=0.5) 
    
    p = pm.invlogit(a[pond] + b_b*big + b_p*pred)
    survival = pm.Binomial('survival', n=n[pond], p=p, observed=survival_obs)
    
    trace_1_4 = pm.sample(2000, tune=2000, return_inferencedata=True, random_seed=RANDOM_SEED, chains=4)

# %%
az.summary(trace_1_4, var_names=['~z', '~a'])

# %% [markdown]
# ### 1.5 Model with size, predation and interaction

# %%
with pm.Model() as model_1_5:
    pond = pm.Data('pond', df1.index.values.astype('int'))
    n = pm.Data('n', df1.density.values.astype('int'))
    survival_obs = pm.Data('survival_obs', df1.surv)
    pred = pm.Data('pred', df1.pred_cat)
    big = pm.Data('big', df1.big)
    
    a_bar = pm.Normal('a_bar', mu=0, sigma=1.5)
    sigma = pm.Exponential('sigma', lam=1)
    
    # This parametrization does not give a warning
    # a = pm.Normal('a', mu=a_bar, sigma=sigma, shape=N_TANKS)
    
    # use non-centered variables for better sampling
    # this gives me a warning, but I don't undertsand why:
    # "The number of effective samples is smaller than 25% for some parameters"
    z = pm.Normal('z', mu=0, sigma=1, shape=N_TANKS)
    a = pm.Deterministic('a', a_bar + z*sigma)
    
    b_p = pm.Normal('b_p', mu=-0.5, sigma=0.5) # expect predation to have negative effect on survival
    b_b = pm.Normal('b_b', mu=0, sigma=0.5) 
    b_pb = pm.Normal('b_pb', mu=0, sigma=0.5)
    
    p = pm.invlogit(a[pond] + b_b*big + b_p*pred + b_pb*big*pred)
    survival = pm.Binomial('survival', n=n[pond], p=p, observed=survival_obs)
    
    trace_1_5 = pm.sample(2000, tune=2000, return_inferencedata=True, random_seed=RANDOM_SEED, chains=4)

# %%
az.summary(trace_1_5, var_names=['~z', '~a'])

# %%
trace_dict = {'m_1_1': trace_1_1, 'm_1_2': trace_1_2, 'm_1_3': trace_1_3,
              'm_1_4': trace_1_4, 'm_1_5': trace_1_5}
az.compare(trace_dict, ic='waic', scale='deviance')

# %%
fig, ax = plt.subplots(1,1)
fig.set_tight_layout(True)

y = list(range(len(trace_dict)))
model = []
left = []
right = []
mean = []

for k in trace_dict:
    summary = az.summary(trace_dict[k], var_names=['sigma'])
    left.append(summary['hdi_3%'].values[0])
    right.append(summary['hdi_97%'].values[0])
    mean.append(summary['mean'].values[0])
    model.append(k)
    
left = np.array(left)
right = np.array(right)
mean = np.array(mean)
ax.errorbar(mean, y, xerr = np.vstack((mean-left, right-mean)), ls='none', marker='o')
ax.set_yticks(y)
ax.set_yticklabels(model)

# %% [markdown]
# The models with out the predation predictor (model_1_3 and model_1_1) have larger values for sigma. This indicates that predation accounts for some of the variance between tanks.
#
# The upshoot appears to be that predation matters, but size matters less.

# %% [markdown]
# # 2 Bangladesh

# %%
df2 = pd.read_csv(os.path.join('data', 'bangladesh.csv'), sep=';')

# %%
df2.head()

# %%
# Note that district 54 is missing
df2.district.unique()

# %%
# make district variable contiguous
distric_to_idx = {d: i for i, d in enumerate(df2.district.unique())}
df2['district_id'] = df2.district.apply(lambda x: distric_to_idx[x])

# %%
N_DISTRICTS = df2.district_id.max()+1

# %% [markdown]
# ## 2.1 Fixed Effect model

# %%
with pm.Model() as model_2_1:
    contraception_obs = pm.Data('contraception_obs', df2['use.contraception'].values)
    district_id = pm.Data('district_id', df2['district_id'].values)
    
    a = pm.Normal('a', mu=0, sigma=1.5, shape=N_DISTRICTS)
    
    p = pm.invlogit(a[district_id])
    contraception = pm.Bernoulli('contraception', p, observed=contraception_obs)
    
    trace_2_1 = pm.sample(2000, tune=2000, return_inferencedata=True, random_seed=RANDOM_SEED, chains=4)

# %% [markdown]
# ## 2.2 Multilevel Model

# %%
with pm.Model() as model_2_2:
    contraception_obs = pm.Data('contraception_obs', df2['use.contraception'].values)
    district_id = pm.Data('district_id', df2['district_id'].values)
    
    a_bar = pm.Normal('a_bar', mu=0, sigma=1.5)
    sigma = pm.Exponential('sigma', lam=1)
    
    z = pm.Normal('z', mu=0, sigma=1, shape=N_DISTRICTS)
    a = pm.Deterministic('a', a_bar + z*sigma)
    
    p = pm.invlogit(a[district_id])
    contraception = pm.Bernoulli('contraception', p, observed=contraception_obs)
    
    trace_2_2 = pm.sample(2000, tune=2000, return_inferencedata=True, random_seed=RANDOM_SEED, chains=4)

# %%
az.summary(trace_2_2, var_names=['~a', '~z'])

# %% [markdown]
# ## 2.3 Compare Model Predictions

# %%
trace_2_2.posterior.a.mean(dim=['chain', 'draw'])

# %%
fig, ax =  plt.subplots(1,1, figsize=(8,4))
fig.set_tight_layout(True)

x = list(range(N_DISTRICTS))
ax.plot(x, sigmoid(az.summary(trace_2_1, var_names='a')['mean'].values), 'oC0', label='fixed effects')
ax.plot(x, sigmoid(az.summary(trace_2_2, var_names='a')['mean'].values), 'xC1', label='multi-level')
ax.plot(x, df2.groupby('district_id').apply(lambda x: x['use.contraception'].mean()), '+C2', label='data')

ax.axhline(y=sigmoid(az.summary(trace_2_2, var_names='a_bar')['mean'].values[0]), ls='--', c='k')
ax.legend(loc=0)

# %% [markdown]
# Note the prediction for district 2. The shrinkage for the multi-level model is very large here, because there is only data from 2 women in district 2.

# %% [markdown]
# # 3 Trolley Data

# %%
df3 = pd.read_csv(os.path.join('data', 'Trolley.csv'), sep=';')

# %%
df3.head()

# %%
NUM_RESP_CATEGORIES  = df3.response.unique().shape[0]

# %%
df3.response.unique()

# %% [markdown]
# ## 3.1 Fixed Effect

# %%
np.linspace(-2.5, 2.5, num=NUM_RESP_CATEGORIES-1)

# %%
with pm.Model() as model_3_1:
    # The response needs to start at 0 for ordered logistic
    response_obs = pm.Data('response_obs', df3.response.values-1)
    
    # Variables
    contact = pm.Data('contact', df3.contact.values)
    action = pm.Data('action', df3.action.values)
    intention = pm.Data('intention', df3.intention.values)
    
    cutpoints = pm.Normal('cutpoints', mu=0, sigma=1.5, shape=NUM_RESP_CATEGORIES-1,
                          testval=np.linspace(-2.5, 2.5, num=NUM_RESP_CATEGORIES-1),
                          transform=pm.distributions.transforms.ordered)
    
    # Parameters
    bA = pm.Normal('bA', mu=0, sigma=0.5)
    bC = pm.Normal('bC', mu=0, sigma=0.5)
    bI = pm.Normal('bI', mu=0, sigma=0.5)
    bIA = pm.Normal('bIA', mu=0, sigma=0.5)
    bIC = pm.Normal('bIC', mu=0, sigma=0.5)
    
    BI = bI + bIA*action + bIC*contact
    phi = bA*action + bC*contact + BI*intention
    
    response = pm.OrderedLogistic('response', eta=phi, cutpoints=cutpoints, observed=response_obs)
    trace_3_1 = pm.sample(2000, tune=2000, return_inferencedata=True, random_seed=RANDOM_SEED, chains=4)

# %%
az.plot_trace(trace_3_1, var_names=['~cutpoints'])

# %%
az.summary(trace_3_1)

# %% [markdown]
# ## 3.2 Multi-Level Model

# %%
id_to_idx = {id_: j for j, id_ in enumerate(df3.id.unique())}
df3['consecutive_id'] = df3.id.apply(lambda x: id_to_idx[x])

# %%
NUM_IDS = df3.id.unique().shape[0]

# %%
with pm.Model() as model_3_2:
    # The response needs to start at 0 for ordered logistic
    response_obs = pm.Data('response_obs', df3.response.values-1)
    
    # Variables
    contact = pm.Data('contact', df3.contact.values)
    action = pm.Data('action', df3.action.values)
    intention = pm.Data('intention', df3.intention.values)
    id_ = pm.Data('id', df3.consecutive_id.values.astype('int'))
    
    cutpoints = pm.Normal('cutpoints', mu=0, sigma=1.5, shape=NUM_RESP_CATEGORIES-1,
                          testval=np.linspace(-2.5, 2.5, num=NUM_RESP_CATEGORIES-1),
                          transform=pm.distributions.transforms.ordered)
    
    # Parameters
    bA = pm.Normal('bA', mu=0, sigma=0.5)
    bC = pm.Normal('bC', mu=0, sigma=0.5)
    bI = pm.Normal('bI', mu=0, sigma=0.5)
    bIA = pm.Normal('bIA', mu=0, sigma=0.5)
    bIC = pm.Normal('bIC', mu=0, sigma=0.5)
    
    # Multi-Level Parameters
    # Do I really need a_bar in this case or are the cutpoints sufficient?
    a_bar = pm.Normal('a_bar', 0,1.5)
    sigma = pm.Exponential('sigma', lam=1)
    z = pm.Normal('z', mu=0, sigma=1, shape=NUM_IDS)
    a = pm.Deterministic('a', a_bar + z*sigma)
    
    BI = bI + bIA*action + bIC*contact
    phi = a[id_] + bA*action + bC*contact + BI*intention
    
    response = pm.OrderedLogistic('response', eta=phi, cutpoints=cutpoints, observed=response_obs)
    trace_3_2 = pm.sample(2000, tune=2000, return_inferencedata=True, random_seed=RANDOM_SEED, chains=4)

# %%
az.summary(trace_3_2, var_names=['~a', '~z'], round_to=2)

# %% [markdown]
# ## 3.3 Multi-Level Model without `a_bar`

# %%
with pm.Model() as model_3_3:
    # The response needs to start at 0 for ordered logistic
    response_obs = pm.Data('response_obs', df3.response.values-1)
    
    # Variables
    contact = pm.Data('contact', df3.contact.values)
    action = pm.Data('action', df3.action.values)
    intention = pm.Data('intention', df3.intention.values)
    id_ = pm.Data('id', df3.consecutive_id.values.astype('int'))
    
    cutpoints = pm.Normal('cutpoints', mu=0, sigma=1.5, shape=NUM_RESP_CATEGORIES-1,
                          testval=np.linspace(-2.5, 2.5, num=NUM_RESP_CATEGORIES-1),
                          transform=pm.distributions.transforms.ordered)
    
    # Parameters
    bA = pm.Normal('bA', mu=0, sigma=0.5)
    bC = pm.Normal('bC', mu=0, sigma=0.5)
    bI = pm.Normal('bI', mu=0, sigma=0.5)
    bIA = pm.Normal('bIA', mu=0, sigma=0.5)
    bIC = pm.Normal('bIC', mu=0, sigma=0.5)
    
    # Multi-Level Parameters
    sigma = pm.Exponential('sigma', lam=1)
    z = pm.Normal('z', mu=0, sigma=1, shape=NUM_IDS)
    a = pm.Deterministic('a', z*sigma)
    
    BI = bI + bIA*action + bIC*contact
    phi = a[id_] + bA*action + bC*contact + BI*intention
    
    response = pm.OrderedLogistic('response', eta=phi, cutpoints=cutpoints, observed=response_obs)
    trace_3_3 = pm.sample(1000, tune=2000, return_inferencedata=True, random_seed=RANDOM_SEED, chains=4,
                          target_accept=0.9)

# %%
az.summary(trace_3_3, var_names=['~a', '~z'], round_to=2)

# %% [markdown]
# The absoulte value of the coefficients is larger than for the fixed effect model. This indicates that when individual differences are taken into account the effect of action, intention, etc is stronger.

# %% [markdown]
# ## Compare Models

# %%
az.compare({'m_3_1': trace_3_1, 'm_3_3': trace_3_3}, ic='waic', scale='deviance')

# %% [markdown]
# The `WAIC` for the multi-level model is quite a bit better than for the fixed effect model.

# %%
with model_3_1:
    ppc_3_1 = pm.sample_posterior_predictive(trace_3_1, var_names=['response'],
                                             random_seed=RANDOM_SEED, samples=100)

# %%
with model_3_3:
    ppc_3_3 = pm.sample_posterior_predictive(trace_3_3, var_names=['response'],
                                             random_seed=RANDOM_SEED, samples=100)

# %%
fig, ax = plt.subplots(1,1, figsize=(8,4))

# Remember that the predicted response is from 0 to 6 instead of 1 to 7.
# That is why we need to add 1.
ax.plot(range(fltr.sum()), ppc_3_1['response'].mean(axis=0)+1 - df3.response, 'xC1',
        label='fixed effect')
ax.plot(range(fltr.sum()), ppc_3_3['response'].mean(axis=0)+1 - df3.response, '+C2',
        label='multi-level')
ax.legend(loc=0)
ax.set_xlabel('row number')
ax.set_ylabel('mean of posterior sample - observed')

# %% [markdown]
# The multi-level model appears to make predictions that are better, but there are some cases where it makes more extreme predictions that are wrong.

# %%
