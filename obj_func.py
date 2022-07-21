import sys,os,math
import numpy as np

class Objective:
	def __init__(self):
		self.max_value=1

	def __call__(self, x_nd):
		return np.zeros(x_nd.shape[0])

	def constraint(self, x_nd):
		return np.asarray([True]*x_nd.shape[0])

	def failure(self, x_nd):
		return np.logical_not(self.constraint(x_nd))


class Circle(Objective):
	def __init__(self, normalizer=1.5309):
		super(Circle,self).__init__()
		self.max_value = normalizer

	def __call__(self, x_nd):
		if x_nd.ndim<2:
			x_nd = x_nd[None,:]

		h_k = np.asarray([1.5,1.,1.,1.])
		c_kd = np.asarray([[0,.7],[.7,0],[0,-.7],[-.7,0]])

		wide = 5
		thin = 1

		# M_kdd = np.asarray([[[wide,0],[0,thin]],[[thin,0],[0,wide]],[[wide,0],[0,thin]],[[thin,0],[0,wide]]])
		M_kdd = np.asarray([[[thin,0.],[0.,wide]],[[wide,0.],[0.,thin]],[[thin,0.],[0.,wide]],[[wide,0.],[0.,thin]]])

		l0_nk = np.concatenate([np.fabs((x_nd - c_kd[[k],:]) @ M_kdd[k,:,:]).sum(axis=1)[:,None] for k in range(c_kd.shape[0])], axis=1)

		lnm_n = np.amax(-l0_nk, axis=1)
		obj_n = np.exp(np.log(h_k)[None,:] - l0_nk - lnm_n[:,None]).sum(axis=1) * np.exp(lnm_n)

		return obj_n / self.max_value

	def constraint(self, x_nd):
		if x_nd.ndim<2:
			x_nd = x_nd[None,:]

		return np.sum(x_nd**2,axis=1)<=1


class Hole(Objective):
	def __init__(self, normalizer=1.8529):
		super(Hole,self).__init__()
		self.max_value = normalizer

	def __call__(self, x_nd):
		if x_nd.ndim<2:
			x_nd = x_nd[None,:]

		h_k = np.asarray([1.5,1.,1.,1.])
		c_kd = np.asarray([[0,.75],[.75,0],[0,-.75],[-.75,0]])

		wide = 5
		thin = 1

		rot = np.asarray([[1.,1],[-1,1]])/np.sqrt(2)

		# M_kdd = np.asarray([[[wide,0],[0,thin]],[[thin,0],[0,wide]],[[wide,0],[0,thin]],[[thin,0],[0,wide]]])
		M_kdd = np.asarray([[[thin,0.],[0.,wide]],[[wide,0.],[0.,thin]],[[thin,0.],[0.,wide]],[[wide,0.],[0.,thin]]])
		M_kdd = np.concatenate([(rot @ M_kdd[k,:,:])[None,:,:] for k in range(M_kdd.shape[0])], axis=0)

		l0_nk = np.concatenate([np.fabs((x_nd - c_kd[[k],:]) @ M_kdd[k,:,:]).sum(axis=1)[:,None] for k in range(c_kd.shape[0])], axis=1)

		lnm_n = np.amax(-l0_nk, axis=1)
		obj_n = np.exp(np.log(h_k)[None,:] - l0_nk - lnm_n[:,None]).sum(axis=1) * np.exp(lnm_n)

		return obj_n / self.max_value

	def constraint(self, x_nd):
		if x_nd.ndim<2:
			x_nd = x_nd[None,:]

		circ_cond_n = np.sum(x_nd**2,axis=1)<=1

		edge_len = np.sqrt(np.pi - 4*0.5)/2

		return np.logical_and(circ_cond_n, np.logical_or(np.fabs(x_nd[:,0])>=edge_len, np.fabs(x_nd[:,1])>=edge_len))


class Softplus(Circle):
	def __init__(self, normalizer=1.632):
		super(Softplus,self).__init__()
		self.max_value = normalizer

	def __call__(self, x_nd):
		if x_nd.ndim<2:
			x_nd = x_nd[None,:]

		_x = np.sum(x_nd, axis=1)
		return (np.log1p(np.exp(-np.fabs(_x))) + np.maximum(_x,0)) / self.max_value

