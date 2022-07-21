import sys,os,glob,math
import argparse

import numpy as np
import matplotlib.pyplot as plt

def rescale(val, bb):
	return (val - bb[0]) / (bb[1] - bb[0])

def get_cummax(y_rn):
	return np.asarray([np.nanmax(y_rn[:,:n], axis=1) for n in range(1,y_rn.shape[1]+1)]).T

def fill_plot(y_rn):
	N = y_rn.shape[1]
	plt.fill_between(np.arange(N)+1, np.min(y_rn, axis=0), np.max(y_rn, axis=0), color=(.553,.674,.796), alpha=.2)
	hndl, = plt.plot(np.arange(N)+1, np.mean(y_rn, axis=0), lw=2, color=(.553,.674,.796), marker='o', markevery=10)

	return hndl

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='show nanBO result npz')
	# parser.add_argument('arg1')
	parser.add_argument('arg1', help='path to file y.npy')

	args = parser.parse_args()

	fill_plot(get_cummax(args.arg1))

	plt.show()