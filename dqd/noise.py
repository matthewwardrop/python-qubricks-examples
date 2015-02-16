
import math
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from dqd import DQD
dqd = DQD('params.py')


p_DC = lambda J_base,t,c_hbar,e_0,e_deviation: 0.5*(1+np.exp(-0.5*(J_base/c_hbar*t/e_0*e_deviation)**2))
p_HF = lambda J_base,t,D,c_hbar,e_0: 0.5*(1+ np.exp(-0.5*D*J_base**2*t/e_0**2/c_hbar**2) )

### Plot singlet-return probability with DC charge noise (to turn on HF noise, uncomment the line indicated below)
def singlet_return_with_dc_noise(**params):
	plt.figure()

	operators = ['evolution']
	#operators.append('J_hf') # UNCOMMENT THIS LINE TO INCLUDE HF NOISE AS WELL AS DC

	times = dqd.p.range('t',t=(0,'3*T_2s',1000),**params)
	times2 = dqd.p.convert(times,output='ns')
	results = dqd.measure.amplitude.iterate(ranges={'de':('-3*e_deviation','3*e_deviation',30)},times=times,psi_0s=['ground'],params=params,operators=operators)
	ranges,ranges_eval,r = results.ranges, results.ranges_eval, results.results['amplitude']
	des = ranges_eval['de']
	weights = stats.norm.pdf(des,loc=0,scale=dqd.p('e_deviation',**params))
	amps = np.average(r['amplitude'],axis=0,weights=weights)[0]

	for i in xrange(r['amplitude'].shape[3]):
		plt.plot(times2, amps[:,i], label='amplitude_%d'%i)

	plt.plot(times2,[dqd.p(p_DC,t=t,**params) for t in times])
	plt.vlines([dqd.p('_T_2s',**params).value],0,1,linestyles='dashed')
	plt.grid()
	xext = plt.xlim()
	plt.xlabel("ns")
	plt.ylim(-0.1,1.1)
	plt.hlines([0.5+0.5/math.e,0.5-0.5/math.e],*xext)

	plt.savefig('noise_DC.pdf')


### Plot singlet-return probability with only HF charge noise in the interaction picture
def singlet_return_with_hf_noise(**params):
	plt.figure()
	print params
	operators = ['J_hf']
	#operators.append('evolution') # UNCOMMENT THIS LINE TO LEAVE THE INTERACTION PICTURE

	times = dqd.p.range('t',t=(0,'3*T_2',1000),**params)
	times2 = dqd.p.convert(times,output='ns')
	result = dqd.measure.amplitude.integrate(times,['ground'],operators=operators,params=params)

	for i in xrange(result.shape[0]):
		for j in xrange(result['amplitude'].shape[-1]):
			plt.plot(times2, result['amplitude'][i,:,j])

	plt.plot(times2,[dqd.p(p_HF,t=t,**params) for t in times])
	plt.vlines([dqd.p('_T_2',**params).value],0,1,linestyles='dashed')
	plt.grid()
	xext = plt.xlim()
	plt.xlabel("ns")
	plt.ylim(-0.1,1.1)
	plt.hlines([0.5+0.5/math.e,0.5-0.5/math.e],*xext)

	plt.savefig('noise_HF.pdf')

singlet_return_with_dc_noise()
singlet_return_with_hf_noise()
