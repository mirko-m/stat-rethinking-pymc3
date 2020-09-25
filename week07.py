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
import pymc3 as pm
import arviz as az
import theano.tensor as tt
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import graphviz

# %%
pm.__version__

# %%
RANDOM_SEED = 8927 # same as used by pymc3 devs

# %%
df_data = pd.read_csv('data/Trolley.csv', sep=';')
df_data.head()

# %% [markdown]
# Higher response values mean the action is considered more morally permissible.

# %%
NUM_RESP_CATEGORIES = len(df_data['response'].unique())
NUM_RESP_CATEGORIES

# %% [markdown]
# # 0 Reproduce Model from Lecture

# %% [markdown]
# ## Calculate Cutpoints directly from the Data

# %%
cum_prob_k = np.array([(df_data['response']<=k).sum() for k in range(1, NUM_RESP_CATEGORIES+1)])/len(df_data)
log_cum_odds_k = np.log(cum_prob_k/(1-cum_prob_k))

fig, axes = plt.subplots(1,3,figsize=(8,4),sharex=True)
fig.set_tight_layout(True)
axes = axes.flatten()

axes[0].hist(df_data['response'], bins=np.array(range(NUM_RESP_CATEGORIES+1))+0.5, rwidth=0.5)
axes[0].set_xlabel('response')
axes[0].set_ylabel('counts')

axes[1].plot(range(1,NUM_RESP_CATEGORIES+1), cum_prob_k, '-o')
axes[1].set_xlabel('response')
axes[1].set_ylabel('Cumulative Probability')

axes[2].plot(range(1,NUM_RESP_CATEGORIES+1), log_cum_odds_k, '-o')
axes[2].set_xlabel('response')
axes[2].set_ylabel('Log Cumulative Odds')

# %% [markdown]
# ## Calculate the Cutpoints with Ordered Logistic Regression

# %% [markdown]
# The response needs to start at zero. This is somewhat surprising because the pymc3 docs say that `OrderedLogistic` is  "useful for regression on ordinal data values whose values range from 1 to K." On the other hand, `OrderedLogistic` inherits from `Categorical` which has support $x \in [0, 1, 2, \dots K-1]$.

# %%
with pm.Model() as model_0:
    cutpoints = pm.Normal('cutpoints', mu=0, sd=1.5,
                          transform=pm.distributions.transforms.ordered,
                          shape=NUM_RESP_CATEGORIES-1,
                          testval = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    
    # the response needs to start at 0 (see the comment above)
    response_shared = pm.Data('response_obs', df_data['response'] - 1)
                              
    _ = pm.OrderedLogistic('response', eta=0, cutpoints=cutpoints, observed=response_shared)
    
    trace_0 = pm.sample(1000, tune=1000, return_inferencedata=True)

# %%
az.summary(trace_0, round_to=2)

# %%
# compare to the manually calculated cutpoints
print('pymc3\tmanual\tdifference')
print('--------------------------')
for v1, v2 in zip(trace_0.posterior.cutpoints.mean(axis=(0,1)).values, log_cum_odds_k[:-1]):
    print('{:.2f} \t{:.2f}\t{:.2e}'.format(v1, v2, v1-v2))

# %% [markdown]
# # 1

# %% [markdown]
# The DAG looks as follows, where
#
# - A = Age
# - E = Education
# - R = Response

# %%
dag = graphviz.Digraph()
dag.node('A', 'A')
dag.node('E', 'E')
dag.node('R', 'R')
dag.edges(['ER', 'AR', 'AE'])
dag

# %% [markdown]
# ## Data Transformations

# %% [markdown]
# ### Encode Eduacation as an Ordered Variable

# %%
for v in df_data['edu'].unique():
    print(v)

# %%
edu_to_num = {"Elementary School": 0,
              "Middle School": 1,
              "Some High School": 2,
              "High School Graduate": 3,
              "Some College": 4,
              "Bachelor's Degree": 5,
              "Master's Degree": 6,
              "Graduate Degree": 7
             }
num_to_edu = {edu_to_num[k]: k for k in edu_to_num}
df_data['edu_ord'] = df_data['edu'].apply(lambda x: edu_to_num[x])

# %%
df_data['edu'].value_counts()

# %%
df_data['edu_ord'].value_counts()

# %%
NUM_EDU_CATEGORIES = len(edu_to_num)

# %% [markdown]
# ### Standardize Age

# %% [markdown]
# This is needed for the model to converge.

# %%
df_data['age_std'] = (df_data['age'] - df_data['age'].mean())/df_data['age'].std()

# %% [markdown]
# ## 1.1 Model Inlcuding Age and Education

# %% [markdown]
# This model blocks the backdoor from Education through Age to Response.

# %% [markdown]
# Note: for some reason these models don't work with pm.sample_prior_predictive . This is probably related to
# https://discourse.pymc.io/t/valueerror-probabilities-are-not-non-negative-when-trying-to-sample-prior-predictive/4559 .
#
# **Update:** the solution appears to be to "pass `theano.tensor.sort(cutpoints)` to the ordered logistic distribution."

# %%
with pm.Model() as model_1_1:
    
    # cutpoints
    cutpoints = pm.Normal('cutpoints', mu=0, sd=1.5,
                          transform=pm.distributions.transforms.ordered,
                          shape=NUM_RESP_CATEGORIES-1,
                          testval = log_cum_odds_k[:-1])
    
    # Create shared variables
    action_shared = pm.Data('action', df_data['action'])
    intention_shared = pm.Data('intention', df_data['intention'])
    contact_shared = pm.Data('contact', df_data['contact'])
    age_shared = pm.Data('age', df_data['age_std']) # need to use standardized values
    edu_shared = pm.Data('edu', df_data['edu_ord']) # this is an ordered variable
    
    # the response needs to start at 0 (see the comment above)
    response_shared = pm.Data('response_obs', df_data['response'] - 1)
    
    # Parameters
    bA = pm.Normal('bA', mu=0, sd=0.5)
    bC = pm.Normal('bC', mu=0, sd=0.5)
    bI = pm.Normal('bI', mu=0, sd=0.5)
    bIA = pm.Normal('bIA', mu=0, sd=0.5)
    bIC = pm.Normal('bIC', mu=0, sd=0.5)
    bAge = pm.Normal('bAge', mu=0, sd=0.5)
    bE = pm.Normal('bE', mu=0, sd=0.5)
    
    delta = pm.Dirichlet('delta', np.full(NUM_EDU_CATEGORIES-1,2))
    delta_j = pm.math.concatenate((tt.zeros(1), delta))
    delta_j_cumsum = tt.cumsum(delta_j)
    
    # regression
    BI = bI + bIA*action_shared + bIC*contact_shared
    phi = pm.Deterministic('phi', bA*action_shared + BI*intention_shared + bC*contact_shared \
                           + bAge*age_shared + bE*delta_j_cumsum[edu_shared])
    
    _ = pm.OrderedLogistic('response', eta=phi, cutpoints=cutpoints, observed=response_shared)
    
    # prior_predicitive_1_1 = pm.sample_prior_predictive() # Throws an error. Why?
    trace_1_1 = pm.sample(2000, tune=2000, return_inferencedata=True, random_seed=RANDOM_SEED)
    # posterior_predictive_1_1 = pm.sample_posterior_predictive(trace_1_1)

# %%
az.summary(trace_1_1, var_names=['~phi'], round_to=2)

# %%
az.plot_forest(trace_1_1, var_names=['~cutpoints', '~delta', '~phi'], combined=False)

# %% [markdown]
# Note how one of the chains gets a lot more variance for `bE`. When I use `combined=True` the values are slightly different from the summary, because the forest plot shows the median instead of the mean.

# %%
trace_1_1.posterior.bE.values.mean(axis=1)

# %%
np.median(trace_1_1.posterior.bE.values, axis=1)

# %%
_ =az.plot_trace(trace_1_1, var_names=['~cutpoints', '~delta', '~phi'])

# %% [markdown]
# You can see that there is an issue with the blue chain.

# %% [markdown]
# ### 1.1.1 Fixing the Trace

# %%
with pm.Model() as model_1_1_1:
    
    # cutpoints
    cutpoints = pm.Normal('cutpoints', mu=0, sd=1.5,
                          transform=pm.distributions.transforms.ordered,
                          shape=NUM_RESP_CATEGORIES-1,
                          testval = np.array(range(6))-2.5)
    
    # Create shared variables
    action_shared = pm.Data('action', df_data['action'])
    intention_shared = pm.Data('intention', df_data['intention'])
    contact_shared = pm.Data('contact', df_data['contact'])
    age_shared = pm.Data('age', df_data['age_std']) # need to use standardized values
    edu_shared = pm.Data('edu', df_data['edu_ord']) # this is an ordered variable
    
    # the response needs to start at 0 (see the comment above)
    response_shared = pm.Data('response_obs', df_data['response'] - 1)
    
    # Parameters
    bA = pm.Normal('bA', mu=0, sd=0.5)
    bC = pm.Normal('bC', mu=0, sd=0.5)
    bI = pm.Normal('bI', mu=0, sd=0.5)
    bIA = pm.Normal('bIA', mu=0, sd=0.5)
    bIC = pm.Normal('bIC', mu=0, sd=0.5)
    bAge = pm.Normal('bAge', mu=0, sd=0.5)
    bE = pm.Normal('bE', mu=0, sd=0.5)
    
    delta = pm.Dirichlet('delta', np.full(NUM_EDU_CATEGORIES-1,2))
    delta_j = pm.math.concatenate((tt.zeros(1), delta))
    delta_j_cumsum = tt.cumsum(delta_j)
    
    # regression
    BI = bI + bIA*action_shared + bIC*contact_shared
    phi = pm.Deterministic('phi', bA*action_shared + BI*intention_shared + bC*contact_shared \
                           + bAge*age_shared + bE*delta_j_cumsum[edu_shared])
    
    _ = pm.OrderedLogistic('response', eta=phi, cutpoints=cutpoints, observed=response_shared)
    
    # prior_predicitive_1_1_1 = pm.sample_prior_predictive() # Throws an error. Why?
    trace_1_1_1 = pm.sample(2000, tune=2000, target_accept=0.9, return_inferencedata=True, random_seed=RANDOM_SEED)
    # posterior_predictive_1_1_1 = pm.sample_posterior_predictive(trace_1_1_1)

# %%
az.summary(trace_1_1_1, var_names=['~phi'], round_to=2)

# %%
az.plot_forest(trace_1_1_1, var_names=['~cutpoints', '~delta', '~phi'], combined=False)

# %%
_ =az.plot_trace(trace_1_1_1, var_names=['~cutpoints', '~delta', '~phi'])

# %% [markdown]
# ## 1.2 Model Without Age

# %%
with pm.Model() as model_1_2:
    
    # cutpoints
    cutpoints = pm.Normal('cutpoints', mu=0, sd=1.5,
                          transform=pm.distributions.transforms.ordered,
                          shape=NUM_RESP_CATEGORIES-1,
                          testval = log_cum_odds_k[:-1])
    
    # Create shared variables
    action_shared = pm.Data('action', df_data['action'])
    intention_shared = pm.Data('intention', df_data['intention'])
    contact_shared = pm.Data('contact', df_data['contact'])
    edu_shared = pm.Data('edu', df_data['edu_ord']) # this is an ordered variable
    
    # the response needs to start at 0 (see the comment above)
    response_shared = pm.Data('response_obs', df_data['response'] - 1)
    
    # Parameters
    bA = pm.Normal('bA', mu=0, sd=0.5)
    bC = pm.Normal('bC', mu=0, sd=0.5)
    bI = pm.Normal('bI', mu=0, sd=0.5)
    bIA = pm.Normal('bIA', mu=0, sd=0.5)
    bIC = pm.Normal('bIC', mu=0, sd=0.5)
    bE = pm.Normal('bE', mu=0, sd=0.5)
    
    delta = pm.Dirichlet('delta', np.full(NUM_EDU_CATEGORIES-1,2))
    delta_j = pm.math.concatenate((tt.zeros(1), delta))
    delta_j_cumsum = tt.cumsum(delta_j)

    
    # regression
    BI = bI + bIA*action_shared + bIC*contact_shared
    phi = pm.Deterministic('phi', bA*action_shared + BI*intention_shared + bC*contact_shared \
                           + bE*delta_j_cumsum[edu_shared])

    
    _ = pm.OrderedLogistic('response', eta=phi, cutpoints=cutpoints, observed=response_shared)
    # prior_predicitive_1_2 = pm.sample_prior_predictive() # Throws an error. Why?
    trace_1_2 = pm.sample(2000, tune=2000, return_inferencedata=True)
    # posterior_predictive_1_2 = pm.sample_posterior_predictive(trace_1_2)

# %%
az.summary(trace_1_2, var_names=['~phi'], round_to=2)

# %%
az.plot_forest(trace_1_2, var_names=['~cutpoints', '~delta', '~phi'], combined=False)

# %% [markdown]
# In this case the effect of Education is negative.

# %%
_ =az.plot_trace(trace_1_2, var_names=['~cutpoints', '~delta', '~phi'])

# %% [markdown]
# # 2

# %%
 with pm.Model() as model_2_1:
    
    # cutpoints
    cutpoints = pm.Normal('cutpoints', mu=0, sd=1.5,
                          transform=pm.distributions.transforms.ordered,
                          shape=NUM_RESP_CATEGORIES-1,
                          testval = np.array(range(6))-2.5)
    
    # Create shared variables
    action_shared = pm.Data('action', df_data['action'])
    intention_shared = pm.Data('intention', df_data['intention'])
    contact_shared = pm.Data('contact', df_data['contact'])
    age_shared = pm.Data('age', df_data['age_std']) # need to use standardized values
    edu_shared = pm.Data('edu', df_data['edu_ord']) # this is an ordered variable
    male_shared = pm.Data('male', df_data['male'])
    
    # the response needs to start at 0 (see the comment above)
    response_shared = pm.Data('response_obs', df_data['response'] - 1)
    
    # Parameters
    bA = pm.Normal('bA', mu=0, sd=0.5)
    bC = pm.Normal('bC', mu=0, sd=0.5)
    bI = pm.Normal('bI', mu=0, sd=0.5)
    bIA = pm.Normal('bIA', mu=0, sd=0.5)
    bIC = pm.Normal('bIC', mu=0, sd=0.5)
    bAge = pm.Normal('bAge', mu=0, sd=0.5)
    bE = pm.Normal('bE', mu=0, sd=0.5)
    bM = pm.Normal('bG', mu=0, sd=0.5)
    
    delta = pm.Dirichlet('delta', np.full(NUM_EDU_CATEGORIES-1,2))
    delta_j = pm.math.concatenate((tt.zeros(1), delta))
    delta_j_cumsum = tt.cumsum(delta_j)
    
    # regression
    BI = bI + bIA*action_shared + bIC*contact_shared
    phi =  bA*action_shared + BI*intention_shared + bC*contact_shared \
           + bAge*age_shared + bE*delta_j_cumsum[edu_shared] \
           + bM*male_shared
    
    _ = pm.OrderedLogistic('response', eta=phi, cutpoints=cutpoints, observed=response_shared)
    
    # prior_predicitive_2_1 = pm.sample_prior_predictive() # Throws an error. Why?
    trace_2_1 = pm.sample(2000, tune=2000, target_accept=0.9, return_inferencedata=True, random_seed=RANDOM_SEED)
    # posterior_predictive_2_1 = pm.sample_posterior_predictive(trace_2_1)

# %%
az.summary(trace_2_1, round_to=2)

# %% [markdown]
# Now the effect of Education is vanishing. Maybe there is an "artifical" relationship between Gender and Education based on how the data was obtained.

# %%
az.plot_forest(trace_2_1, var_names=['~cutpoints', '~delta'], combined=False)

# %%
_ = az.plot_trace(trace_2_1, var_names=['~cutpoints', '~delta'])

# %%
