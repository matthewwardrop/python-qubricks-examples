	#!/usr/bin/python2

def start_pdb(signal, trace):
    import pdb
    pdb.set_trace()

import signal
signal.signal(signal.SIGQUIT, start_pdb)

import sys,os,shelve
sys.path.append('..')
sys.path.append('/Storage/Education/University/PhD/libraries')
sys.path.append('/Storage/PhD/libraries')

import numpy as np
from scipy import stats
from dqd import DQD
from qubricks.analysis import ModelAnalysis


res, profile = sys.argv[1:3]
res = int(res)

operators = ['evolution']
operators.extend(sys.argv[3:])

def path(name=''):
	return os.path.join('fidelity_noise','%s-%s'%(profile,','.join(operators)),'%dx%d'%(res,res),name)


class NoisyFidelity(ModelAnalysis):

	def prepare(self):
		self.system = DQD(parameters='../params.py',profile=profile)

	def simulate(self):
		ranges = [
				{
					'dB': ( (0.01,'mT'), (200,'mT'), res)
				},
				{
					'J_1_avg': ( (0.01,'{mu}eV'), (0.7,'{mu}eV'), res)
				},
				{
					'de1': ( -3*self.system.p('e_deviation'), 3*self.system.p('e_deviation'), 21)
				}
			]

		return self.system.measure.entanglement_fidelity.iterate_to_file(path=path('data.shelf'),ranges=ranges,times=['T'],subspace='logical',operators=operators,params={'J_2':0,'J_2_avg':0},nprocs=-1,yield_every=100)#,error_abs=1e-8,error_rel=1e-8)

	def process(self,results=None):
		ranges,ranges_eval,results = results.ranges, results.ranges_eval,results.results['entanglement_fidelity']
		##### Data preparation #############################################
		X,Y = [
			self.system.p.asvalue(**{
				ranges[i].keys()[0]: self.system.p.range(ranges[i].keys()[0],**{
									ranges[i].keys()[0]: ranges[i].values()[0]
							})
					}) for i in xrange(2)]

		des = self.system.p.range('de1',de1=ranges[2]['de1'])

		Z = results['fidelity'][:,:,:,-1] # get results at time 'T' for all parameter values in ranges
		L = results['leakage'][:,:,:,-1]

		weights = stats.norm.pdf(des,loc=0,scale=self.system.p('e_deviation'))

		fidelities = np.average(Z,axis=2,weights=weights)
		leakages = np.average(L,axis=2,weights=weights)

		Z = fidelities
		L = leakages

		# Investigate what plot looks like without pseudo-static noise
		#Z = results.fidelity['T'][:,:,10]
		#L = results.fidelity['T'][:,:,10]

		return {'X': X, 'Z':Z, 'L':L, 'Y':Y}

	def plot(self,X,Y,Z,L):
		##### PLOTTING RESULTS ############################################


		import matplotlib
		matplotlib.use('agg')
		import matplotlib.pyplot as plt
		from matplotlib import cm,colors
		from mpl_toolkits.mplot3d import Axes3D
		from mplstyles import SampleStyle
		from mplstyles.plots import contour_image

		style = SampleStyle()
		with style:

			cmap = colors.LinearSegmentedColormap.from_list('Bob',[(0,"#444444"),(0.99,"#FFFFFF"),(1,"#0055BB")],10000)
			cmap2 = colors.LinearSegmentedColormap.from_list('Bob',[(0,"#0055BB"),(0.01,"#FFFFFF"),(1,"#444444")],10000)

			plt.figure()
			contour_image(X,Y,Z,cmap=cmap,contour_opts={'levels':[0.995,0.996,0.997,0.998,0.999,0.9991,0.9992,0.9993,0.9994,1]},vmin=0,vmax=1,interpolation='nearest',label=True, label_opts={'fmt':'%1.4f'})
			plt.ylabel("$J (\mu eV)$")
			plt.xlabel("$\Delta B (mT)$")
			plt.grid(True)
			style.savefig(path('fidelity_contour.pdf'))

			plt.figure()
			contour_image(X,Y,L,cmap=cmap2,contour_opts={'levels':[0.98,0.99,0.998]},vmin=0,vmax=1,interpolation='nearest')
			plt.ylabel("$J (\mu eV)$")
			plt.xlabel("$\Delta B (mT)$")
			plt.grid(True)
			style.savefig(path('leakage_contour.pdf'))

			plt.figure()
			contour_image(X,Y,Z+L,cmap=cmap,contour_opts={'levels':[0.98,0.99,0.998]},vmin=0,vmax=1,interpolation='nearest')
			plt.ylabel("$J (\mu eV)$")
			plt.xlabel("$\Delta B (mT)$")
			plt.grid(True)
			style.savefig(path('leakage_difference.pdf'))

			fig = plt.figure()
			ax = fig.gca(projection='3d')
			x,y = np.meshgrid(X,Y)
			ax.view_init(elev=45,azim=-45)

			#np.where(Z>0.9,Z,np.nan)

			ax.plot_surface(x,y,Z,rstride=1, cstride=1, cmap=cmap,
			linewidth=0.0001, antialiased=False,vmin=0,vmax=1,edgecolor=(0,0,0,0))
			ax.contour(x,y,Z,levels=np.linspace(0,1,20),offset=0)
			plt.ylabel("$\Delta B$")
			plt.xlabel("$J_{avg}$")
			ax.set_zlabel("$\mathcal{F}$")
			ax.set_zlim(0,1)
			#style()
			style.savefig(path('fidelity_contour_3d.pdf'))

		#print test.measure.leakage(['T'],y_0s=['ground'],subspace='logical',operators=['evolution'],params={'J_1_avg':0.15,'J_2_avg':0.15,'dB':-0.1})
		#print test.measure.entanglement_fidelity(['T'],subspace='logical',operators=['evolution'],params={'J_1_avg':(0.1,'{mu}eV'),'J_2_avg':(0.1,'{mu}eV'),'dB':(30,'mT')},ideal_op=ideal_op)

		#f3 = test.measure.leakage.iterate(ranges,params={'J_2_avg':0},y_0s=['ground'],times=['T'],subspace='logical',operators=['evolution'],error_abs=1e-16,error_rel=1e-16)
		#print f3
		#print f3.structure()
		#f3 >> 'fidelity_new.hdf5'

nf = NoisyFidelity()
nf.plot()
