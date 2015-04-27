import re
import math

import numpy as np
import sympy
import scipy.optimize as spo
import scipy.linalg as spla
from parampy import Parameters

from qubricks import QuantumSystem, StateOperator, Basis, Operator, OrthogonalOperator
from qubricks.wall import LindbladStateOperator, SpinBasis, AmplitudeMeasurement
from qubricks.utility import tensor, dot

from measurements import EntanglementFidelity,ProjectionProbabilities

I,X,Y,Z = np.array([[1,0],[0,1]]),np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.array([[1,0],[0,-1]])

class TQD (QuantumSystem):

	def init_parameters(self):
		#self.set_noise()
		subscripts = ['12','23']
		for subscript in subscripts:
			self.setup_noisy_exchange(subscript)
		self.enable_noise(False,*subscripts)

	def enable_noise(self,enable,*subscripts):
		for subscript in subscripts:
			if enable:
				self.p << {'J_%s'%subscript: ('J_%s_noisy'%subscript,'{mu}eV')}
			else:
				self.p << {'J_%s'%subscript: ('J_%s_noiseless'%subscript,'{mu}eV')}

	def setup_noisy_exchange(self,subscript):

		e_proto = '''
def e(J_{0}_noiseless,J_0,J_1,e_0,e_{0}=None):
	if e_{0} is None:
		return e_0*math.log( (J_{0}_noiseless - J_0) / J_1 )
	return J_0 + J_1*math.exp( e_{0} / e_0 ), J_0, J_1, e_0'''
		exec(e_proto.format(subscript))

		J_noisy_proto = '''
def J_noisy(e_{0},de_{0},J_0,J_1,e_0):
	return J_0 + J_1*math.exp( ( e_{0} + de_{0} ) / e_0 )
		'''
		exec(J_noisy_proto.format(subscript))

		N_proto = '''
def N(J_{0},J_0,e_0):
	return (J_{0}-J_0)**2/e_0**2
		'''
		exec(N_proto.format(subscript))

		self.p << {
			'e_%s'%subscript: (e,'mV'),
			'J_%s_noisy'%subscript: (J_noisy,'{mu}eV'),
			'N_%s'%subscript: (N,'{mu}eV^2/mV^2'),
			'de_%s'%subscript: (0,'mV'),
			}

	def init_states(self):
		g = self.state_fromString

		states = {
			'l0':g('|uud>+|duu>-2|udu>')[:8],
			'l1':g('|uud>-|duu>')[:8],
			'ml0':g('|ddu>+|udd>-2|dud>')[:8],
			'ml1':g('|ddu>-|udd>')[:8],
			'Q':g('|uud>+|duu>+|udu>')[:8],
			'q':g('|ddu>+|udd>+|dud>')[:8],
			'u':g('|uuu>')[:8],
			'S1': g('|udu>-|duu>')[:8]
		}
		states['ground'] = states['l0']/np.linalg.norm(states['l0'])+self.state(states['l1'])/np.linalg.norm(states['l1'])

		for state in states:
			self.add_state(state,states[state])

	def init_bases(self):
		self.add_basis("default", SpinBasis(dim=self.dim,parameters=self.p))
		self.add_basis("working", WorkingBasis(dim=self.dim,parameters=self.p,exact=False))

	def init_hamiltonian(self):
		components = {
			'coupling': self.Operator( {
				"J_12/4": tensor(X,X,I) + tensor(Y,Y,I) + tensor(Z,Z,I) - tensor(I,I,I),
				"J_23/4": tensor(I,X,X) + tensor(I,Y,Y) + tensor(I,Z,Z) - tensor(I,I,I),
				} ),
			'zeeman': self.Operator( {
				"B_1": tensor(Z,I,I),
				"B_2": tensor(I,Z,I),
				"B_3": tensor(I,I,Z),
				} ),
		}
		return self.OperatorSet(components)

	def get_derivative_ops(self,components=None):

		def coupling(subscript,sites=3):
			base = [I]*sites
			r = - tensor(*base)
			for i in [X,Y,Z]:
				c = base[:]
				c[int(subscript[0])-1] = i
				c[int(subscript[1])-1] = i
				r += tensor(*c)
			return r

		for subscript in ['12','23']:
			self.add_derivative_op('J_%s_hf'%subscript, LindbladStateOperator(coefficient='D*N_%s'%subscript, operator=self.Operator(0.25*coupling(subscript))) )

	def init_measurements(self):
		self.add_measurement('amplitude',AmplitudeMeasurement())
		self.add_measurement('projection',ProjectionProbabilities())

class WorkingBasis(Basis):

	def init(self, exact=False):
		if self.dim is None:
			raise ValueError("Dimension of basis must be specified.")
		self.exact = exact

	@property
	def operator(self):
		try:
			return self.__operator_cache
		except:
			pass

		f = SpinBasis(dim=2**3,parameters=self.p)

		import sympy
		from sympy import S
		from qubricks.basis import QubricksKet as ket, QubricksBra as bra

		if True:#self.exact:
			l0 = S("1/sqrt(6)")*(ket('uud')+ket('duu')-2*ket('udu'))
			l1 = S("1/sqrt(2)")*(ket('uud')-ket('duu'))
			Q = S("1/sqrt(3)")*(ket('uud')+ket('duu')+ket('udu'))

			ml0 = S("1/sqrt(6)")*(ket('ddu')+ket('udd')-2*ket('dud'))
			ml1 = S("1/sqrt(2)")*(ket('ddu')-ket('udd'))
			q = S("1/sqrt(3)")*(ket('ddu')+ket('udd')+ket('dud'))
			u = ket('uuu')
			d = ket('ddd')

			states = [
					l0,
					l1,
					Q,
					ml0,
					ml1,
					q,
					u,
					d
					]

			states = [x.expand().simplify().expand() for x in states]
			states = map(lambda x: f.state_fromSymbolic(x),states)

		if len(states) != self.dim:
			raise ValueError, "Number of basis states (%d) != self.dim (%d)" % (len(states),self.dim)

		if self.exact:
			self.__operator_cache = OrthogonalOperator(sympy.Matrix(states).transpose(), parameters=self.p, exact=self.exact)
		else:
			self.__operator_cache = OrthogonalOperator(np.array(states,dtype=float).transpose(), parameters=self.p, exact=self.exact) # ,dtype='complex'
		#self.__operator_cache = Operator(np.array(states,dtype=complex).transpose(),parameters=self.p)
		return self.__operator_cache
		#return Operator(sympy.Matrix(states).transpose(),parameters=self.p)

	def state_latex(self,state,params={}):
		spinString = ''
		index = state.index(1)
		for i in xrange(3):
			mod = index%2
			if mod == 0:
				spinString = '\\uparrow' + spinString
			else:
				spinString = '\\downarrow' + spinString
			index = (index-index%2)/2
		return '\\left|%s\\right>' % (spinString)

	def state_fromString(self,state,params={}):
		states = [
				'l0',
				'l1',
				'Q',
				'ml0',
				'ml1',
				'q',
				'u'
				]

		print state
		matches = re.findall("(([\+\-]?(?:[0-9\.]+)?)\|([0-1Qu\-]+)\>)",state)

		ostate = [0.]*self.dim

		for m in matches:
			if m[2] in states:

				try:
					c = float(m[1])
				except:
					c = 1.

				ostate[states.index(m[2])] += c
			else:
				raise ValueError( "'%s' does not qualify as a known state." % m[0] )

		return ostate/np.linalg.norm(ostate)

	def state_toString(self,state,params={}):
		states = [
				'l0',
				'l1',
				'Q',
				'ml0',
				'ml1',
				'q',
				'u',
				'd'
				]

		s = ""
		for i,v in enumerate(state):
			if v != 0:

				if i < 8:
					repr = states[i]
				else:
					repr = self.__state_str(i)

				if s != "" and v >= 0:
					s +="+"
				if v == 1:
					s += "|%s>" % repr
				else:
					if np.imag(v) != 0:
						s += "(%.3f + %.3fi)|%s>" % (np.real(v),np.imag(v),repr)
					else:
						s += "%.3f|%s>" % (v,repr)
		return s

	def __state_str(self,index):
		s = ""
		for i in xrange(3):
			mod = index%2
			if mod == 0:
				s = 'u' + s
			else:
				s = 'd' + s
			index = (index-index%2)/2
		return s

	def state_info(self,state,params={}):
		totalSpin = 0
		index = state.index(1)
		for i in xrange(3):
			mod = (index%2-1.0/2.0)
			totalSpin -= mod
			index = (index-index%2)/2
		return {'spin':totalSpin}
