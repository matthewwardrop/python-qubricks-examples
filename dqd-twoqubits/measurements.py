import numpy as np
import math

from qubricks import Measurement,StateOperator


class EntanglementFidelity(Measurement):

	def init(self,ideal_ops=None,**kwargs):
		self.__ideal_ops = ideal_ops

	def measure(self,times,params={},operators=None,input=None,output=None,subspace=None,ideal_ops=None,**kwargs):
		ideal_ops = self.__ideal_ops if ideal_ops is None else ideal_ops

		if not isinstance(ideal_ops[0],StateOperator):
			raise ValueError, "Ideal operator must be a StateOperator object."

		use_ensemble = self.system.use_ensemble(operators)

		y_0s = self.__get_y_0s(subspace,operators,input=input,output=output)
		r = self.system.integrate(times,initial=y_0s,input=input,output=output,operators=operators,params=params,**kwargs)

		ideal_y_0 = np.sum(self.system.subspace(subspace,output=output),axis=0)
		ideal_y_0 = ideal_y_0/np.linalg.norm(ideal_y_0)
		r_bell = self.system.integrate(times,initial=[ideal_y_0],input=input,output=output,operators=['evolution'],params=params,**kwargs)

		for i in range(len(ideal_ops)):
			ideal_ops[i] = ideal_ops[i].change_basis(output)

		leakage_projector = self.system.subspace_projector(subspace, input=input, output=output, invert=True, params=params)

		fidelities = [self.fidelity(times, r, r_bell, y_0s, ideal_op, leakage_projector, use_ensemble, params = params) for ideal_op in ideal_ops]

		r = self.__max_fidelities(fidelities)

		r = np.array([ (time,r['fidelity'][time],r['leakage'][time]) for time in sorted(r['fidelity']) ],dtype=self.result_type(times=times))

		return r

	def result_type(self,ranges=[],shape=None,params={},*args,**kwargs):
		return [
					('time',float),
					('fidelity',float),
					('leakage',float)
				]

	def result_shape(self,*args,**kwargs):
		return (len(kwargs.get('times',0)),)

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

		return r

	@property
	def is_independent(self):
		return True

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

	def __get_y_0s(self,subspace,operators,input=None,output=None):

		use_ensemble = self.system.use_ensemble(operators)

		y_0s = self.system.subspace(subspace,input=input,output=output)

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
