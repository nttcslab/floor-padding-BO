import sys,os,argparse,tqdm

import math
import numpy as np
import sklearn.gaussian_process as GP

import matplotlib
import matplotlib.pyplot as plt

import lhsmdu # latin hypercube sampling package
import BO_core, obj_func, visualize

###################################
# acquisition utility
###################################
def grid_uniform(space_d):
	grid_d = np.meshgrid(*space_d, indexing='ij')

	for d in range(len(grid_d)):
		# calculate interval
		diff = np.diff(sorted(list(set(grid_d[d].ravel()))))
		if diff.size == 0:
			continue
		interval = np.mean(diff)

		if np.isfinite(interval):
			grid_d[d] += np.random.uniform(high=interval, size=grid_d[d].shape)

	return np.c_[[g.ravel() for g in grid_d]].T

def num_grid_uniform(num_d, low=0, high=1):
	space_d = [np.linspace(low,high,num+1)[:num] for num in num_d]

	return grid_uniform(space_d)

def fixed_grid_uniform(num_d, l_fix, low=0, high=1):
	space_d = [np.linspace(low,high,num+1)[:num] for num in num_d]
	for d,val in l_fix:
		space_d[d] = np.asarray([val])

	return grid_uniform(space_d)


def lhs_uniform(x_d, len_d, l_fix=[], num=64):
	uni01_nd = np.asarray(lhsmdu.sample(x_d.size, num)).T
	grid_X_nd = np.clip(x_d[np.newaxis,:] + len_d[np.newaxis,:] * (uni01_nd - 0.5), a_min=0, a_max=1)

	for d,val in l_fix:
		grid_X_nd[:,d] = val

	return grid_X_nd


####################################
# [0,1] -- raw config value convertor
# x: [0,1]
# raw: values used in actual experiments
####################################
def uni_to_raw(x_d, bb):
	return bb[0,:] + x_d * (bb[1,:] - bb[0,:])

def raw_to_uni(raw_d, bb):
	return (raw_d - bb[0,:]) / (bb[1,:] - bb[0,:])

def all_u2r(x_nd, bb):
	return np.asarray([uni_to_raw(x_nd[n,:], bb) for n in range(x_nd.shape[0])])

def all_r2u(raw_config_nd, bb):
	return np.asarray([raw_to_uni(raw_config_nd[n,:], bb) for n in range(raw_config_nd.shape[0])])

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='BO')
	parser.add_argument('--num', type=int, default=100)
	parser.add_argument('--init', type=int, default=5)
	parser.add_argument('--run', type=int, default=1)
	parser.add_argument('--binary', action='store_true')


	parser.add_argument('--alpha', type=float, default=0., help='Observation noise variance')
	parser.add_argument('--floor', type=float, help='Value to replace failed observation. Default: floor padding')
	parser.add_argument('--obj', choices=['circle','hole', 'softplus'], default='circle')
	parser.add_argument('--seed', type=int, default=3649)
	parser.add_argument('--out', help='specify a prefix to save result. Example: "--out foo" saves all y values to foo_y.npy and x values to foo_x.npy.')

	args = parser.parse_args()

	if args.obj=='circle':
		func = obj_func.Circle()
	elif args.obj=='hole':
		func = obj_func.Hole()
	elif args.obj=='softplus':
		func = obj_func.Softplus()

	D = 2
	np.random.seed(args.seed)

	# parse bounding boxes
	xbb = np.zeros((2,D));
	xbb[0,:] = -1
	xbb[1,:] = 1 # search space is in [-1, 1]

	ybb = np.asarray([[0],[1]]) # shape=(2,1), no normalization on y

	gp_kernel = GP.kernels.Product(GP.kernels.ConstantKernel(constant_value=1., constant_value_bounds=(1e-2,1e2)), GP.kernels.Matern(length_scale=1, length_scale_bounds=(3e-2, 10.),nu=2.5))
	gp_kernel = GP.kernels.Sum(gp_kernel, GP.kernels.WhiteKernel(1e-7, noise_level_bounds=(1e-8,1e-6)))

	y_rn = np.zeros((args.run, args.num))
	x_rnd = np.zeros((args.run, args.num, D))

	# entering loop here
	for r in tqdm.tqdm(range(args.run), desc='Runs'):
		raw_config_nd = all_u2r(np.asarray(lhsmdu.sample(xbb.shape[1], args.init, randomSeed=args.seed)).T, xbb) 
		rawy_n = func(raw_config_nd)
		rawy_n[func.failure(raw_config_nd)] = np.nan
		noise_level = math.sqrt(args.alpha)
		eps_n = np.random.normal(size=rawy_n.size) * noise_level

		opt_progress_bar = tqdm.tqdm(total=args.num, leave=False, desc='observations')
		while raw_config_nd.shape[0] < args.num:
			# normalize values
			X_nd = all_r2u(raw_config_nd, xbb)
			Y_n = raw_to_uni(rawy_n + eps_n, ybb)

			if args.binary:
				bo = BO_core.classifier_BO(gp_kernel, {'floor':args.floor,})
			else:
				bo = BO_core.padding_BO(gp_kernel, {'floor':args.floor,})

			# fitting
			bo.fit(X_nd, Y_n,)

			# acquisition over grid
			grid_X_nd = fixed_grid_uniform(np.asarray((100,)*D), [])
			acq_n, m_n, s_n = bo.acquisition(grid_X_nd, return_ms=True,) # expected improvement (EI)
			# acq_n, m_n, s_n = bo.ucb(grid_X_nd, return_ms=True, scale=np.sqrt(raw_config_nd.shape[0])) # UCB

			idx_n = np.argsort(acq_n)[::-1]

			# observe oracle (objective function call)
			raw_config_nd = np.r_[raw_config_nd, np.asarray(uni_to_raw(grid_X_nd[idx_n[0],:], xbb))[None,:]]
			rawy_n = np.r_[rawy_n, (func(raw_config_nd[-1,:]) if func.constraint(raw_config_nd[-1,:])[0] else np.nan)]
			eps_n = np.r_[eps_n, np.random.normal(size=1) * noise_level]

			opt_progress_bar.update(1)

		y_rn[r,:] = rawy_n
		x_rnd[r,:,:] = raw_config_nd

	if args.out is not None and len(args.out)>0:
		np.save('{}_x.npy'.format(args.out), x_rnd)
		np.save('{}_y.npy'.format(args.out), y_rn)


	visualize.fill_plot(visualize.get_cummax(y_rn))
	plt.show()