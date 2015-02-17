import math
from tqd import TQD
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
plt.ion()

d = TQD('params.py')

T_2s = 25 # ns
T_2 = 20000 # ns
e_0 = 0.35 # mV
hbar = d.p._c_hbar('{mu}eV*ns').value # hbar
J = d.p._J_12_avg.value # {mu}eV representative J

# Determine values of noise parameters from T_2 times # Approximately true in limit that sigma is small compared to J
sigma = np.sqrt(2) * hbar * e_0 / J/ T_2s # mV
D = 2 * hbar**2 * e_0**2 / J**2 / T_2 # mV^2 * ns

# Theoretical Gaussian Decay due to T_2*
print sigma, 'mV'
p = lambda Javg,t: 0.5*(1+np.exp(-0.5*(Javg/d.p._c_hbar('{mu}eV*ns').value*t/e_0*sigma)**2))

# Theoretical Exponential Decay due to T_2
print D, 'mV^2*ns'
p_D = lambda Javg,t: 0.5*(1+ np.exp(-0.5*D*Javg**2*t/e_0**2/d.p._c_hbar('{mu}eV*ns').value**2) )

def get_random_sampler(ratio=1):
	def random_sampler(mean,std,count):
		assert(count % ratio == 0)
		return list(np.random.normal(mean,std,(count/ratio,)))*ratio
	return random_sampler

states = ['l0','l1','ground','S1']

pseudostatic = True

if pseudostatic:
	times = list(np.linspace(0,d.p( (30,'ns') ),1000))
	d.enable_noise(True,'12','23')
	results = d.measure.projection.iterate(ranges={
			'de_12':(0, (3*sigma,'mV') ,100, get_random_sampler()),
			'de_23':(0, (3*sigma,'mV') ,100, get_random_sampler()),
			'B_1': (0,'-B_oh_stddev',100,get_random_sampler()),
			'B_2': (0,'-B_oh_stddev',100,get_random_sampler()),
			'B_3': (0,'-B_oh_stddev',100,get_random_sampler()),
		},times=times,psi_0s=['S1'],output='working',components=['coupling','zeeman'],states=states,operators=['evolution'],params={'e_deviation':(sigma,'mV')})
	ranges = results.ranges
	ranges_eval = results.ranges_eval
	r = results.results['projection']
	amps = np.average(r['probabilities'],axis=0)[0]
else:
	times = list(np.linspace(0,d.p( (20,'{mu}s') ),1000))
	r = d.measure.projection.integrate(times=times,psi_0s=['ground'],output='working',components=['coupling'],states=states,operators=['evolution','J_12_hf','J_23_hf'],params={'D':(D,'mV^2*ns')})
	amps = r[0]['probabilities']

plt.figure()
for i in xrange(r['probabilities'].shape[-1]):
	plt.plot(d.p.convert(times,output='ns',value=True), amps[:,i], label=states[i])
plt.legend()

if pseudostatic:
	plt.plot(d.p.convert(times,output='ns',value=True),[p(d.p._J_12.value,t) for t in d.p.convert(times,output='ns',value=True)],linestyle='--')
else:
	plt.plot(d.p.convert(times,output='ns',value=True),[p_D(d.p._J_12.value,t) for t in d.p.convert(times,output='ns',value=True)],linestyle='--')

#plt.vlines([d.p.T,d.p('20*T'),np.sqrt(2)*d.p.c_hbar*d.p((0.35,'mV'))/d.p.J/sigma/d.p((1,'mV'))],0,1,linestyles='dashed')
plt.grid()
xext = plt.xlim()
plt.xlabel("ns")
plt.ylim(-0.1,1.1)
plt.hlines([0.5+0.5/math.e,0.5-0.5/math.e],*xext)

'''plt.annotate(str(amplitudes2['T'][2].real), xy=(d.p.T, amplitudes2['T'][2].real),  xycoords='data',
	        xytext=(50, -30), textcoords='offset points',
	        bbox=dict(boxstyle="round", fc="0.8"),
	        arrowprops=dict(arrowstyle="->",
	                        shrinkA=0, shrinkB=0,
	                        connectionstyle="angle,angleA=90,angleB=180,rad=10"),
	        )'''

plt.savefig('pseudo-static.pdf')

#-------------------------------------------------------------------------------
