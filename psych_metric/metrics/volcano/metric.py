import numpy as np
from scipy.stats import multinomial
import itertools
import matplotlib.pyplot as plt
import theano.tensor as tt
import pymc3 as pm

from psych_metric.metrics.base_metric import BaseMetric

class VolcanoMetricPaperEM(BaseMetric):
    def __init__(self):
        pass

class VolcanoMetricMultinomialEM(BaseMetric):
    def __init__(self, n_classes=2):
        self.n_classes = n_classes
        self.n_features = None
        self.params = dict()
        self.epsilon = 1e-5

    def copy_params(self, params):
        if isinstance(params, dict): return {k: self.copy_params(v) for k, v in params.items()}
        else: return params.copy()

    def init_params(self, X, params, method='random'):
        params['prior'] = np.ones((self.n_classes,)) / float(self.n_classes)
        if method == 'random':
            posterior = np.random.uniform(0, 1, (self.n_classes, self.n_features))
            posterior = posterior / (np.sum(posterior, axis=1, keepdims=True) + self.epsilon)
        else:
            raise NotImplementedError

        params['posterior'] = posterior
        return params

    def expectation(self, X, params):
        posterior = [
            params['prior'][i] * multinomial.pmf(X, n=np.sum(X, axis=1), p=params['posterior'][i])
            for i in range(self.n_classes)
        ]
        posterior = np.stack(posterior, axis=1)
        # posterior_norm = posterior / np.sum(posterior, axis=1, keepdims=True)
        yhat = np.argmax(posterior, axis=1)
        return yhat

    def maximization(self, X, Yhat, params, normalize_first=False):
        for i in range(self.n_classes):
            params['prior'][i] = np.mean(Yhat == i)
        
        if normalize_first: X = X / (np.sum(X, axis=1, keepdims=True) + self.epsilon)
        
        for i in range(self.n_classes):
            Xi = X[Yhat == i]
            params['posterior'][i] = np.sum(Xi, axis=0) / (np.sum(Xi) + self.epsilon)
            
        return params

    def has_converged(self, params1, params2, threshold=0.001):
        p1 = params1['posterior']
        p2 = params2['posterior']
        delta = np.sqrt(np.sum(np.square(p1 - p2)))
        return (delta < threshold)

    def train(self, X, convergence_threshold=0.001):
        N, self.n_features = X.shape[0], X.shape[1]
        self.params = self.init_params(X, self.params, method='random')
        last_params = self.copy_params(self.params)
        for i in itertools.count():
            yhat = self.expectation(X, self.params)
            self.params = self.maximization(X, yhat, self.params)
            
            if self.has_converged(last_params, self.params, convergence_threshold): break
            last_params = self.copy_params(self.params)

    def plot_prior(self):
        classes = range(self.n_classes)
        features = range(self.n_features)
        fig, ax = plt.subplots(1,1, figsize=(self.n_features*5, 5))
        ax.bar(classes, self.params['prior'], alpha=0.9)
        ax.set_ylim([0,1])
        ax.set_title('Prior (Occurrence of Class)', fontsize=20)
        ax.set_ylabel('Portion of data points', fontsize=15)
        ax.set_xlabel('Class Number', fontsize=15)
        ax.set_xticks(classes)
        plt.show()

    def plot_posterior(self):
        classes = range(self.n_classes)
        features = range(self.n_features)
        fig, axarr = plt.subplots(1, self.n_classes, figsize=(self.n_classes*5, 5), sharex=True, sharey=True)
        for i in range(self.n_classes):
            axarr[i].bar(features, self.params['posterior'][i])
            axarr[i].set_ylim([0,1])
            axarr[i].set_xticks(features)
            axarr[i].set_xlabel('Features', fontsize=15)
        axarr[0].set_ylabel('Probability', fontsize=15)
        plt.suptitle('Posterior (Representative Multinomials)', fontsize=20)
        plt.show()


class VolcanoMetricMultinomialMC(BaseMetric):
    def __init__(self, n_classes=3):
        self.n_classes = n_classes
        self.trace=None

    def train(self, X):
        self.n_data, self.n_features = X.shape[0], X.shape[1]
        K = np.sum(X, axis=1)
        with pm.Model() as model:
            prior = pm.Dirichlet("prior", a=np.ones((self.n_classes,)), shape=(self.n_classes,))
            p_min_potential = pm.Potential('p_min_potential', tt.switch(tt.min(prior) < .05, -np.inf, 0))

            category = pm.Categorical('category', p=prior, shape=self.n_data)
            
            H = pm.Dirichlet("H", a=np.ones((self.n_classes, self.n_features)), shape=(self.n_classes, self.n_features))
            Hi = pm.Deterministic('Hi', H[category])
            
            yhat = pm.Multinomial('M', n=K, p=Hi, observed=X)

        with model:
            trace = pm.sample(2000, chains=1, tune=300) # Chains 1 because there is currently no ordering potential

        self.trace = trace

    def plot_posterior_pymc3(self):
        pm.traceplot(self.trace, var_names=['prior', 'H',])
        plt.show()
