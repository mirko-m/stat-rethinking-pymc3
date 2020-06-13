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
import pymc3


# %% [markdown]
# # Note

# %% [markdown]
# This implementation suffers from overflow/underflow problems. It is probably better to work only with logs.

# %%
def log_n_choose_k(n, k):
    return np.log(range(n-k+1, n+1)).sum() - np.log(range(1,k+1)).sum()


# %%
def likelihood(m,theta,N=15):
    '''
    m: (int) number of tosses with water as outcome
    N: (int) total number of tosses
    theta: (float) fraction of water
    '''
    n = N - m
    logp = m*np.log(theta) + n*np.log(1-theta)
    log_norm = log_n_choose_k(N, m) # the norm reduces the chance of underflow
    return np.exp(logp + log_norm)


# %%
def grid_search(m,prior,N=15, eps=1e-12):
    
    thetas = np.linspace(0.001,1.0,num=1000, endpoint=False)
    posterior = likelihood(m,thetas,N=N) * prior(thetas) # not normalized
    norm = np.trapz(posterior,x=thetas) + eps # add eps for numerical stability
    return posterior/norm, thetas


# %% [markdown]
# # Flat prior

# %%
def flat_prior(x):
    return np.ones(x.shape[0])


# %%
post_flat, thetas = grid_search(8,flat_prior)

fig, ax = plt.subplots(1,1)
fig.set_tight_layout(True)
ax.plot(thetas, post_flat)
ax.set_xlabel('$ \\theta$')
ax.set_ylabel('$P( \\theta | m)$')

print('Mode of posterior:')
print(thetas[np.argmax(post_flat)])

print('')
print('Expectation value of theta:')
print(np.trapz(thetas * post_flat, x=thetas))


# %% [markdown]
# # Informed prior

# %%
def informed_prior(x):
    out = 2*np.ones(x.shape[0])
    out[x < 0.5] = 0
    return out


# %%
informed_prior(np.linspace(0,1)).shape

# %%
post_informed, thetas = grid_search(8, informed_prior)

fig, ax = plt.subplots(1,1)
fig.set_tight_layout(True)
ax.plot(thetas, post_informed)
ax.set_xlabel('$ \\theta$')
ax.set_ylabel('$P( \\theta | m)$')

print('Mode of posterior:')
print(thetas[np.argmax(post_informed)])

print('')
print('Expectation value of theta:')
print(np.trapz(thetas * post_informed, x=thetas))


# %%
def percentiles(posterior, thetas, p=0.99):
    
    q = (1 - p)/2
    step = thetas[1] - thetas[0]
    cumsum = np.cumsum(posterior)*step # should really be using trapezoidal rule here
    idx_upper = np.where(cumsum > 1-q)[0].min()
    idx_lower = np.where(cumsum < q)[0].max()
    return thetas[idx_lower], thetas[idx_upper]


# %%
# Getting 99 percentiles
fig, ax = plt.subplots(1,1)
fig.set_tight_layout(True)

step = thetas[1] - thetas[0]
cumsum = np.cumsum(post_informed)*step
ax.plot(thetas, cumsum) # should really be using trapezoidal rule here

p1, p2 = percentiles(post_informed, thetas,p=0.99)
ax.axvline(p1, ls='--', c='C3')
ax.axvline(p2, ls='--', c='C3')
print(p1)
print(p2)

# %% [markdown]
# # How Many Tosses

# %%
fig, ax = plt.subplots(1,1)
fig.set_tight_layout(True)

# If I make the number of tosses too large there can be overflow/underflow errors
n_tosses = np.logspace(1,4,num=10) # number of tosses
sample_size = 10
theta0_vals = [0.51, 0.6, 0.7, 0.8, 0.9]
for count, theta0 in enumerate(theta0_vals):
    widths = np.empty((n_tosses.shape[0], sample_size))
    for i, n in enumerate(n_tosses):
        N = int(n)
        binomial = pymc3.distributions.discrete.Binomial.dist(N, theta0)
        samples = binomial.random(size=sample_size)
        for j, m in enumerate(samples):
            # print(N, m, theta0)
            post, thetas = grid_search(m,informed_prior,N=N)
            p1, p2 = percentiles(post, thetas, p=0.99)
            widths[i,j] = p2 - p1

    means = np.mean(widths, axis=1)
    stds = np.std(widths, axis=1)
    ax.errorbar(n_tosses, means, yerr=stds, marker='x', color='C{:d}'.format(count),
                label=theta0)
ax.legend(loc=0)
ax.axhline(y=0.05, ls='--', c='k')
ax.set_xlabel('N')
ax.set_ylabel('Distribution Width')

# %%
