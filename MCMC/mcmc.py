#
#File:   mcmc.py
#Brief:  implementation of universal sample generator
#Author: XXHRXX
#Date:   03-19-2024
#

import os
import sys
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt

#implementation of three universal sample generator for complex posterior distribution
#1. Accept / Reject
#2. Importance Sampling
#3. MCMC

def accept_reject(posterior):
	#given f(x), choose g(x) that is easy to sample from, f(x)/g(x) <= M

	res = []

	#we use g(x) as a normal distribution with (10, 1), then M = 1

	batch_size = 1000
	M = 200
	g_distribution = scipy.stats.norm(10, 1)
	uniform_distribution = scipy.stats.uniform(0, M)
	while len(res) < 1000:
		g_samples = g_distribution.rvs(size = batch_size)
		uniform_samples = uniform_distribution.rvs(size = batch_size)

		for i in range(len(g_samples)):
			temp_uniform = uniform_samples[i]

			if temp_uniform <= posterior.pdf(g_samples[i])/g_distribution.pdf(g_samples[i]):
				res.append(g_samples[i])
		
		print(len(res))

	return res



def main():
	#use a normal distribution (10, 20) as the posterior distribution that we want to sample from
	posterior = scipy.stats.norm(10, 20)

	res1 = accept_reject(posterior)

	bins = np.arange(0, 20, 1)
	plt.hist(x = res1, bins = bins)
	plt.show()


if __name__ == '__main__':
	main()
