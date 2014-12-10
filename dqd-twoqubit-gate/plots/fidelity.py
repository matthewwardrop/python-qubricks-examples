#!/usr/bin/python2

import sys,os,shelve

sys.path.append('..')


import matplotlib
matplotlib.use('agg')
from matplotlib import cm,colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from qubricks.utility import ModelAnalysis

from mplstyles import SampleStyle
from mplstyles.plots import contour_image

from dqd import DQD


import matplotlib.pyplot as plt

res, profile = sys.argv[1:3]
res = int(res)

operators = ['evolution']
operators.extend(sys.argv[3:])

def path(name=''):
	return os.path.join('fidelity','%s-%s'%(profile,','.join(operators)),'%dx%d'%(res,res),name)

class Fidelity(ModelAnalysis):

	def prepare(self):
		try:
			self.system = DQD(parameters='../params.py',profile=profile)
			self.system.set_ansatz('exp')
		except Exception as e:
			print e

	def simulate(self):
		ranges = [
			{
				'dB': ( (0.01,'mT'), (200,'mT'), res)
			},
			#{
			#	"D_1": ( (0.01*1.171e-05,'mV^2*ns'), (10*1.171e-05,'mV^2*ns'), 10 ),
			#},
			{
				'J_1_avg': ( (0.01,'{mu}eV'), (0.7,'{mu}eV'), res)
			},
		]

		self.system.measure.entanglement_fidelity.multiprocessing = True
		return self.system.measure.entanglement_fidelity.iterate_to_file(path=path('data.shelf'),ranges=ranges,times=['T'],subspace='logical',operators=operators,params={'J_2':0,'J_2_avg':0},nprocs=-1,yield_every=100)#,error_abs=1e-9,error_rel=1e-9),'dB':(200,'mT')

	def process(self,results=None):
		ranges,ranges_eval,results = results

		##### Data preparation #############################################
		X,Y = [
			self.system.p.asvalue(**{
				ranges[i].keys()[0]: self.system.p.range(ranges[i].keys()[0],**{
									ranges[i].keys()[0]: ranges[i].values()[0]
							})
					}) for i in xrange(2)]

		Z = results['fidelity'][:,:,-1] # get results at time 'T' for all parameter values in ranges
		L = results['leakage'][:,:,-1]

		return {'X': X, 'Y':Y, 'Z':Z, 'L':L}

	def plot(self,X,Y,Z,L):
		##### PLOTTING RESULTS ############################################

		import matplotlib
		matplotlib.use('agg')
		import matplotlib.pyplot as plt
		from matplotlib import cm,colors
		from mpl_toolkits.mplot3d import Axes3D
		from mplstyles import SampleStyle

		style = SampleStyle()
		with style:

			cmap = colors.LinearSegmentedColormap.from_list('Bob',[(0,"#444444"),(0.99,"#FFFFFF"),(1,"#0055BB")],10000)
			cmap2 = colors.LinearSegmentedColormap.from_list('Bob',[(0,"#0055BB"),(0.01,"#FFFFFF"),(1,"#444444")],10000)

			plt.figure()
			contour_image(X,Y,Z,cmap=cmap,contour_opts={'levels':[0.999,0.9991,0.9992,0.9993,0.9994,0.9995,0.9996,0.9997,0.9998,0.9999]},vmin=0,vmax=1,interpolation='nearest',label=False)
			plt.ylabel("$J (\mu eV)$")
			plt.xlabel("$\Delta B (mT)$")
			plt.grid(True)
			style.savefig(path('fidelity_contour.pdf'))

			plt.figure()
			contour_image(X,Y,L,cmap=cmap2,contour_opts={'levels':[1-0.999,1-0.9991,1-0.9992,1-0.9993,1-0.9994,1-0.9995,1-0.9996,1-0.9997,1-0.9998,1-0.9999]},vmin=0,vmax=1,interpolation='nearest')
			plt.ylabel("$J (\mu eV)$")
			plt.xlabel("$\Delta B (mT)$")
			plt.grid(True)
			style.savefig(path('leakage_contour.pdf'))

			plt.figure()
			print Z-np.sqrt(1.-2.*L)
			contour_image(X,Y,Z+L,cmap=cmap,contour_opts={'levels':[0.999,0.9991,0.9992,0.9993,0.9994,0.9995,0.9996,0.9997,0.9998,0.9999]},vmin=0,vmax=1,interpolation='nearest')
			plt.ylabel("$J (\mu eV)$")
			plt.xlabel("$\Delta B (mT)$")

			plt.grid(True)
			style.savefig(path('leakage_noise_components.pdf'))

			plt.figure()
			plt.gca().set_yscale('log')
			plt.gca().set_xscale('log')
			Xm, Ym = np.meshgrid(X, Y)
			plt.contour(Xm,Ym,np.transpose(Z+L),levels=[0.999,0.9991,0.9992,0.9993,0.9994,0.9995,0.9996,0.9997,0.9998,0.9999])
			plt.ylabel("$J (\mu eV)$")
			plt.xlabel("$\Delta B (mT)$")

			plt.grid(True)
			style.savefig(path('leakage_noise_components_contour.pdf'))


			fig = plt.figure()
			ax = fig.gca(projection='3d')
			x,y = np.meshgrid(X,Y)
			ax.view_init(elev=45,azim=-45)

			#np.where(Z>0.9,Z,np.nan)
			ax.plot_surface(x,y,Z,rstride=1, cstride=1, cmap=cmap,
			linewidth=0.0001, antialiased=False,vmin=0,vmax=1,edgecolor=(0,0,0,0))
			ax.contour(x,y,Z+L,levels=np.linspace(0,1,20),offset=0)
			plt.ylabel("$\Delta B$")
			plt.xlabel("$J_{avg}$")
			ax.set_zlabel("$\mathcal{F}$")
			ax.set_zlim(0,1)
			#style()
			style.savefig(path('fidelity_contour_3d.pdf'))

f = Fidelity()
f.plot()
