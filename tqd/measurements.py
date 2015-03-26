import numpy as np
import math

from qubricks import Measurement,StateOperator

class ProjectionProbabilities(Measurement):
	'''
	ProjectionProbabilities is a sample Measurement subclass that measures the amplitude of being
	in certain states as function of time throughout some state evolution.
	'''

	def init(self):
		pass

	def result_type(self,*args,**kwargs):
		return [
					('time',float),
					('probabilities',float,(len(kwargs.get('states')),) )
				]

	def result_shape(self,*args,**kwargs):
		return (len(kwargs['psi_0s']),len(kwargs.get('times',0)))

	def measure(self,data,times=None,psi_0s=None,params={},subspace=None,states=[],**kwargs):

		if len(states) == 0:
			return ValueError("No states have been specified against which to project.")
		states = [ self.system.state(state,input=kwargs.get("input"),output=kwargs.get("output"),threshold=kwargs.get("threshold"),params=params) for state in states ]

		rval = np.empty((len(data),len(times)),dtype=self.result_type(psi_0s=psi_0s,times=times,states=states))

		self.__P = None
		for i,resultset in enumerate(data):
			for j,time in enumerate(resultset['time']):
				rval[i,j] = (time,self.projection(resultset['state'][j],states))

		return rval

	def projection(self,state,states):
		r = np.zeros(len(states))
		for i,s in enumerate(states):
			if len(state.shape) > 1:
				r[i] = np.trace(np.outer(s.transpose().conj(),s).dot(state))
			else:
				r[i] = np.abs(np.dot(state,s))**2
		return r


class EntanglementFidelity(Measurement):

	def init(self,ideal_ops=None,**kwargs):
		self.__ideal_ops = ideal_ops

	@property
	def name(self):
		return 'fidelity'

	def result_type(self,ranges=[],shape=None,params={},*args,**kwargs):
		return [
					('time',float),
					('fidelity',float),
					('leakage',float)
				]

	def result_shape(self,*args,**kwargs):
		return (len(kwargs.get('times',0)),)

	@property
	def result_units(self):
		return None

	def measure(self,times,params={},operators=None,input=None,output=None,subspace=None,ideal_ops=None,threshold=True,**kwargs):

		ideal_ops = self.__ideal_ops if ideal_ops is None else ideal_ops

		if not isinstance(ideal_ops[0],StateOperator):
			raise ValueError, "Ideal operator must be a StateOperator object."

		use_ensemble = self.system.use_ensemble(operators)

		y_0s = self.__get_y_0s(subspace,operators,input=input,output=output,threshold=threshold,params=params)
		r = self.system.integrate(times,psi_0s=y_0s,input=output,output=output,threshold=threshold,operators=operators,params=params,**kwargs)

		ideal_y_0 = np.sum(self.system.subspace(subspace,input=input,output=output,threshold=threshold,params=params),axis=0)
		ideal_y_0 = ideal_y_0/np.linalg.norm(ideal_y_0)
		r_bell = self.system.integrate(times,psi_0s=[ideal_y_0],input=output,output=output,threshold=threshold,operators=['evolution'],params=params,**kwargs)

		for i in range(len(ideal_ops)):
			ideal_ops[i] = ideal_ops[i].change_basis(self.system.basis(output),threshold=threshold)

		leakage_projector = self.system.subspace_projector(subspace, input=input, output=output, threshold=threshold, invert=True, params=params)

		fidelities = [self.fidelity(times, r, r_bell, y_0s, ideal_op, leakage_projector, use_ensemble, params = params) for ideal_op in ideal_ops]

		r = self.__max_fidelities(fidelities)

		r = np.array([ (time,r['fidelity'][time],r['leakage'][time]) for time in sorted(r['fidelity']) ],dtype=self.result_type(times=times))

		return r


	def __max_fidelities(self,fidelities):
		keys = sorted(fidelities[0]['fidelity'].keys())

		max_fidelity = np.max(np.array([ [f['fidelity'][key] for key in keys] for f in fidelities]),axis=0)
		min_leakage = np.min(np.array([ [f['leakage'][key] for key in keys] for f in fidelities]),axis=0)

		fidelity = {}
		leakage = {}
		for i,key in enumerate(keys):
			fidelity[key] = max_fidelity[i]
			leakage[key] = min_leakage[i]

		r = {
				'fidelity': fidelity, 'leakage': leakage
			}

		#import pdb; pdb.set_trace()
		return r

	def __get_y_0s(self,subspace,operators,input=None,output=None,threshold=False,params={}):

		use_ensemble = self.system.use_ensemble(operators)

		y_0s = self.system.subspace(subspace,input=input,output=output,threshold=threshold,params=params)

		if use_ensemble:
			# If p_0s is None, create a self-entangled set of density operators
			p_0s = []
			for i in xrange(len(y_0s)):
				i_state = y_0s[i]
				for j in xrange(i, len(y_0s)):
					j_state = y_0s[j]
					p_0 = np.outer(i_state.conjugate(), j_state)
					p_0s.append(p_0)
			return p_0s

		return y_0s

	def fidelity(self, times, r, r_bell, y_0s, ideal_op, leakage_projector, use_ensemble, params = {}):

		if use_ensemble:
			dim = int(0.5*(-1 + math.sqrt(1 + 8*len(y_0s))))
		else:
			dim = len(y_0s)

		fidelities = {}
		leakages = {}

		#print times
		#print r[0]['time']

		n = 0
		for i,resultset_i in enumerate(r):
			if use_ensemble:
				m = 2
				if n/2.*(2*dim-n+1) == i:
					n += 1
					m = 1

				y_ij = y_0s[i]

				for k,(t,label,state_ij) in enumerate(resultset_i):

					bell_state = r_bell[0][k]['state']

					y_ij_ideal = ideal_op(t,y_ij,bell_state,params=params)

					fidelities[t] = fidelities.get(t,0.0) + np.real(float(m) / dim**2 * np.trace ( state_ij.conjugate().transpose().dot( y_ij_ideal ) ))

					if m == 1:
						leakages[t] = leakages.get(t,0.0) + np.trace( state_ij.dot(leakage_projector) ).real / dim
			else:
				y_0_i = y_0s[i]
				for j, resultset_j in enumerate(r[i:]):
					y_0_j = y_0s[i+j]
					for k,(t,label,state_i) in enumerate(resultset_i):

						state_j =  resultset_j[k]['state']

						bell_state = r_bell[0][k]['state']
						y_ij = np.outer(y_0_i,y_0_j.conjugate())
						y_ij_ideal = ideal_op(t,y_ij,bell_state,params=params)

						state_ji = np.outer(state_j,state_i.conjugate())

						fidelities[t] = fidelities.get(t,0.0) + np.real(float(1 if j==0 else 2) / dim**2 * np.trace ( state_ji.dot(y_ij_ideal) ))

						if j == 0:
							leakages[t] = leakages.get(t,0.0) + np.trace( state_ji.dot(leakage_projector) ).real / dim

		#print fidelities

		return {'fidelity': fidelities, 'leakage': leakages}
