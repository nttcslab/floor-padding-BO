import sys,os,argparse

import math
import numpy as np
import scipy.stats as sstat
import scipy.linalg as slin
import scipy.special as ssp
import sklearn.gaussian_process as GP

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

import torch, gpytorch

class padding_BO:
	def __init__(self, kernel, conf):
		self.init_kernel_ = kernel
		self.kernel_ = kernel
		self.floor_ = conf['floor']

		self.gp_ = None

	def fit(self, X_nd, Y_n):
		_Y_n = Y_n.copy()

		if (self.floor_ is None) or (self.floor_ is np.nan):
			fY_n = _Y_n[np.isfinite(_Y_n)]
			if fY_n.size > 0:
				_Y_n[np.isnan(_Y_n)] = np.min(fY_n)
			else:
				_Y_n[np.isnan(_Y_n)] = 0
		else:
			_Y_n[np.isnan(_Y_n)] = self.floor_

		return self._fit(X_nd, _Y_n)

	def _fit(self, X_nd, Y_n):
		simplefilter('ignore', category=ConvergenceWarning)
		gp = GP.GaussianProcessRegressor(kernel=self.init_kernel_, n_restarts_optimizer=4, normalize_y=False, alpha=0)
		gp.fit(X_nd, Y_n)

		self.gp_ = gp

		return gp

	def predict(self, X_nd):
		m_n, s_n = self.gp_.predict(X_nd, return_std=True)
		return m_n, s_n
		
	def acquisition(self, X_nd, return_prob=False, return_ms=False):
		if self.gp_ is None:
			return np.random.uniform(size=(X_nd.shape[0]))

		m_n, s_n = self.gp_.predict(X_nd, return_std=True)

		diff_best_n = (m_n - self.gp_.y_train_.max())
		z_n = diff_best_n / s_n

		cdf_n = sstat.norm.cdf(z_n)
		pdf_n = sstat.norm.pdf(z_n)

		EI_n = np.fmax(0, diff_best_n*cdf_n + s_n*pdf_n)

		ret = (EI_n, np.ones_like(EI_n) if return_prob else None, m_n if return_ms else None, s_n if return_ms else None)
		ret = tuple([elem for elem in ret if elem is not None])

		if len(ret)==1:
			return EI_n
		else:
			return ret

	def ucb(self, X_nd, return_prob=False, return_ms=False, scale=.1):
		if self.gp_ is None:
			return np.random.uniform(size=(X_nd.shape[0]))

		m_n, s_n = self.gp_.predict(X_nd, return_std=True)
		ucb_n = m_n + scale * s_n

		ret = (ucb_n, np.ones_like(ucb_n) if return_prob else None, m_n if return_ms else None, s_n if return_ms else None)
		ret = tuple([elem for elem in ret if elem is not None])

		if len(ret)==1:
			return ucb_n
		else:
			return ret

class GPClassifier(gpytorch.models.AbstractVariationalGP):
	def __init__(self, X_nd):
		var_dist = gpytorch.variational.CholeskyVariationalDistribution(X_nd.size(0))
		var_strategy = gpytorch.variational.VariationalStrategy(self, X_nd, var_dist)
		super(GPClassifier, self).__init__(var_strategy)
		self.mean_module = gpytorch.means.ZeroMean()
		self.covar_module = gpytorch.kernels.ScaleKernel(
			gpytorch.kernels.MaternKernel(nu=2.5, lengthscale_constraint=gpytorch.constraints.Interval(0.08,.1,initial_value=0.09)), 
			outputscale_constraint=gpytorch.constraints.Interval(1e-2,1e2))

	def forward(self, X_nd):
		mx = self.mean_module(X_nd)
		covx = self.covar_module(X_nd)

		latent_pred = gpytorch.distributions.MultivariateNormal(mx, covx)
		return latent_pred

class classifier_BO:
	def __init__(self, kernel, conf):
		self.init_kernel_ = kernel
		self.kernel_ = kernel
		self.floor_ = conf['floor']

		self.gp_ = None
		self.gpc_ = None
		self.ll_ = None


	def fit(self, X_nd, Y_n):
		_Y_n = Y_n.copy()


		if (self.floor_ is None) or np.isnan(self.floor_):
			fY_n = _Y_n[np.isfinite(_Y_n)]
			if fY_n.size > 0:
				_Y_n[np.isnan(_Y_n)] = np.min(fY_n)
			else:
				_Y_n[np.isnan(_Y_n)] = 0
		else:
			_Y_n[np.isnan(_Y_n)] = self.floor_

		simplefilter('ignore', category=ConvergenceWarning)
		gp = GP.GaussianProcessRegressor(kernel=self.init_kernel_, n_restarts_optimizer=1, normalize_y=False, alpha=0)
		gp.fit(X_nd, _Y_n)

		# classifier
		notnan_n = np.ones_like(_Y_n)
		notnan_n[np.isnan(Y_n)] = 0

		train_x = torch.tensor(X_nd.astype(np.float32))
		train_y = torch.tensor(notnan_n.astype(np.float32))

		gpc = GPClassifier(train_x)
		likelihood = gpytorch.likelihoods.BernoulliLikelihood()
		gpc.train()
		likelihood.train()

		opt = torch.optim.Adam(gpc.parameters(), lr=1e-1)
		lr_schedule = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[1000,1500,], gamma=math.sqrt(.1))
		mll = gpytorch.mlls.variational_elbo.VariationalELBO(likelihood, gpc, train_y.numel())

		ITER=200

		for t in range(ITER):
			opt.zero_grad()
			loss = -mll(gpc(train_x), train_y)
			loss.backward()
			# if t % 50 == 0:
			# 	print('{}/{} - Loss: {:.3f}'.format(t, ITER, loss.item()))
			opt.step()
			lr_schedule.step()


		self.gp_ = gp
		self.gpc_ = gpc
		self.ll_ = likelihood

		return self.gp_

	def predict(self, X_nd):
		m_n, s_n = self.gp_.predict(X_nd, return_std=True)
		return m_n, s_n

	def predict_notnan(self, X_nd):
		self.gpc_.eval()
		self.ll_.eval()

		with torch.no_grad():
			p_n = self.ll_(self.gpc_(torch.tensor(X_nd.astype(np.float32))))

		return p_n.mean.numpy()

	def raw_EI(self, X_nd):
		m_n, s_n = self.gp_.predict(X_nd, return_std=True)

		diff_best_n = (m_n - self.gp_.y_train_.max())
		z_n = diff_best_n / s_n

		cdf_n = sstat.norm.cdf(z_n)
		pdf_n = sstat.norm.pdf(z_n)

		EI_n = np.fmax(0, diff_best_n*cdf_n + s_n*pdf_n)

		return EI_n


	def acquisition(self, X_nd, return_prob=False, return_ms=False):
		if self.gp_ is None:
			return np.random.uniform(size=(X_nd.shape[0]))

		m_n, s_n = self.gp_.predict(X_nd, return_std=True)

		diff_best_n = (m_n - self.gp_.y_train_.max())
		z_n = diff_best_n / s_n

		cdf_n = sstat.norm.cdf(z_n)
		pdf_n = sstat.norm.pdf(z_n)

		EI_n = np.fmax(0, diff_best_n*cdf_n + s_n*pdf_n)

		notnan_prob_n = self.predict_notnan(X_nd)

		ret = (EI_n * notnan_prob_n, notnan_prob_n if return_prob else None, m_n if return_ms else None, s_n if return_ms else None)
		ret = tuple([elem for elem in ret if elem is not None])

		if len(ret)==1:
			return EI_n * notnan_prob_n
		else:
			return ret
