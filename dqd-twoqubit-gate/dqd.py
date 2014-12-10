import math, numpy as np

from qubricks import QuantumSystem
from qubricks.wall import LindbladStateOperator,StateOperator,DummyStateOperator, LeakageMeasurement as Leakage
from qubricks.utility import tensor
from measurements import EntanglementFidelity

class DQD (QuantumSystem):

	# Required methods
	def setup_environment(self,profile='sinusoidal',ansatz='exp',**kwargs):
		self.__profile = profile
		self.__ansatz = ansatz

	def setup_parameters(self):
		self.set_ansatz(self.__ansatz)
		self.set_profile(self.__profile)

		self.p.show()

	def setup_states(self):
		self.add_state('00', [0,0,0,0,1,0])
		self.add_state('01', [0,0,0,1,0,0])
		self.add_state('10', [0,0,1,0,0,0])
		self.add_state('11', [0,1,0,0,0,0])

		self.add_subspace('logical', [[0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0]])

	def setup_hamiltonian(self):

		I,X,Y,Z = np.array([[1,0],[0,1]]),np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.array([[1,0],[0,-1]])

		components = {
			'B_0+0.5*D_B+0.5*D_12': tensor(Z,I,I,I), # Parameter operator is equal to B_1
			'B_0+0.5*D_B-0.5*D_12': tensor(I,Z,I,I), # Parameter operator is equal to B_2
			'B_0-0.5*D_B+0.5*D_34': tensor(I,I,Z,I), # Parameter operator is equal to B_3
			'B_0-0.5*D_B-0.5*D_34': tensor(I,I,I,Z), # Parameter operator is equal to B_4

			"J_23/4": tensor(I,X,X,I) + tensor(I,Y,Y,I) + tensor(I,Z,Z,I) - tensor(I,I,I,I),
			"J_14/4": tensor(X,I,I,X) + tensor(Y,I,I,Y) + tensor(Z,I,I,Z) - tensor(I,I,I,I),
		}

		return self.Operator(components).restrict(3,5,6,9,10,12).collapse()
		# We have restricted by fiat to the m_z=0 subspace. Not necessary, except that it makes the symbolic matrices smaller

	def setup_measurements(self):
		self.add_measurement('leakage',Leakage())

		# Add an entanglement_fidelity measure.
		self.add_measurement('entanglement_fidelity',EntanglementFidelity(ideal_ops=[IdealOperator(self.p)]))

		# Add an entanglement_fidelity measure when in the interaction picture
		self.add_measurement('entanglement_fidelity_eye',EntanglementFidelity(ideal_ops=[IdentityOperator(self.p)]))

	@property
	def default_derivative_ops(self):
		return ["evolution"]

	def setup_derivative_ops(self):
		I,X,Y,Z = np.array([[1,0],[0,1]]),np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.array([[1,0],[0,-1]])

		# J_23 noise
		j1 = self.Operator( 0.25*(tensor(I,Z,Z,I) + tensor(I,X,X,I) + tensor(I,Y,Y,I) - tensor(I,I,I,I)) ).restrict(3,5,6,9,10,12)
		self.add_derivative_op("J_24_hf", LindbladStateOperator(coefficient='D_23*N_23', operator=j1))

		# J_14 noise
		j2 = self.Operator( 0.25*(tensor(Z,I,I,Z) + tensor(X,I,I,X) + tensor(Y,I,I,Y) - tensor(I,I,I,I)) ).restrict(3,5,6,9,10,12)
		self.add_derivative_op("J_14_hf", LindbladStateOperator(coefficient='D_14*N_14', operator=j2))

	def setup_bases(self):
		pass



	# Ancillary helper functions
	def set_ansatz(self,ansatz):

		if ansatz == 'exp':

			J_0 = self.p.exp_J_0
			J_1 = self.p.exp_J_1
			e_0 = self.p.exp_e_0

			def set_e_23(J_23_noiseless,e_23=None):
				if e_23 is None:
					return e_0*math.log( (J_23_noiseless - J_0) / J_1 )
				return J_0 + J_1*math.exp( e_23 / e_0 )
			def set_e_14(J_14_noiseless,e_14=None):
				if e_14 is None:
					return e_0*math.log( (J_14_noiseless - J_0) / J_1 )
				return J_0 + J_1*math.exp( e_14 / e_0 )
			self.p << {'e_23':(set_e_23,'mV'),'e_14':(set_e_14,'mV')}

			# Noisy J
			def J_23_noisy(e_23,de1):
				return J_0 + J_1*math.exp( ( e_23 + de1 ) / e_0 )
			def J_14_noisy(e_14,de2):
				return J_0 + J_1*math.exp( ( e_14 + de2 ) / e_0 )
			self.p << {'J_23_noisy': (J_23_noisy, '{mu}eV'),'J_14_noisy': (J_14_noisy, '{mu}eV') }

			self.p << {
				"N_1": (lambda J_1: (J_1-J_0)**2/e_0**2, '{mu}eV^2/mV^2'),
				"N_2": (lambda J_1: (J_1-J_0)**2/e_0**2, '{mu}eV^2/mV^2'),
				}

		elif ansatz == 'hyp':
			a = self.p.hyp_a
			b = self.p.hyp_b
			c = self.p.hyp_c

			def set_e_23(J_23_noiseless,e_23=None):
				if e_23 is None:
					return a/(J_23_noiseless-c) + b
				return a/(e_23-b) + c
			def set_e_14(J_14_noiseless,e_14=None):
				if e_14 is None:
					return a/(J_14_noiseless-c) + b
				return a/(e_14-b) + c
			self.p << {'e_23':(set_e_23,'mV'),'e_14':(set_e_14,'mV')}

			# Noisy J
			def J_23_noisy(e_23,de1):
				return a/(e_23 + de1 - b) + c
			def J_14_noisy(e_14,de2):
				return a/(e_14 + de2 - b) + c
			self.p << {'J_23_noisy':(J_23_noisy, '{mu}eV'),'J_14_noisy':(J_14_noisy, '{mu}eV')}

			self.p << { # TODO: Match with exp ansatz; use actual value of J
				"N_1": (lambda J_1: (J_1-c)**4/a**2, '{mu}eV^2/mV^2'),
				"N_2": (lambda J_14: (J_14-c)**4/a**2, '{mu}eV^2/mV^2'),
				}

		else:
			raise ValueError, "Unknown ansatz for J(e) '%s'" % ansatz

	def set_profile(self,profile):
		'''
		In this method we set the profile for J. Note that we use lambda methods for marginal speed improvements
		over setting via a string representation of the mathematics.
		'''
		if profile in ['square','square_noisy']:
				self.p << {'J_23_noiseless': lambda J_23_avg: J_23_avg}
				self.p << {'J_14_noiseless': lambda J_14_avg: J_14_avg}
		elif profile in ['linear','linear_noisy']:
				self.p << {'J_23_noiseless': lambda J_23_avg,t,T: 2*J_23_avg*(1-abs(2*t/T-1))}
				self.p << {'J_14_noiseless': lambda J_14_avg,t,T: 2*J_14_avg*(1-abs(2*t/T-1))}
		elif profile in ['sinusoidal','sinusoidal_noisy']:
				self.p << {'J_23_noiseless': lambda J_23_avg,t,T: J_23_avg*(1-math.cos(2*t*math.pi/T)) }
				self.p << {'J_14_noiseless': lambda J_14_avg,t,T: J_14_avg*(1-math.cos(2*t*math.pi/T)) }
		elif profile in ['xsinusoidal','xsinusoidal_noisy']:
				self.p << {'J_23_noiseless':lambda J_23_avg,t,T: J_23_avg*6*math.pi**2/(math.pi**2+3)*t*(T-t)/T**2*(1-math.cos(2*t*math.pi/T))}
				self.p << {'J_14_noiseless':lambda J_14_avg,t,T: J_14_avg*6*math.pi**2/(math.pi**2+3)*t*(T-t)/T**2*(1-math.cos(2*t*math.pi/T))}
		else:
				raise ValueError, "Unknown profile '%s'" % profile
		self.p & {'J_23_noiseless':'{mu}eV', 'J_14_noiseless':'{mu}eV'}

		if profile.endswith('_noisy'):
			self.p << {
				'J_23': lambda J_23_noisy: J_23_noisy,
				'J_14': lambda J_14_noisy: J_14_noisy,
				}
		else:
			self.p << {
				'J_23': lambda J_23_noiseless: J_23_noiseless,
				'J_14': lambda J_14_noiseless: J_14_noiseless,
				}

		self.p & {'J_23':'{mu}eV', 'J_14':'{mu}eV'}

	# A simple diagnostic tool to plot single runs of the two-qubit gate
	def plot_fidelity(self,period='T',times=None,operators=['evolution'],figure=None,count=100,**params):
		plot=False
		if times is None:
			times = map (lambda x: self.p('t',t=x), list(np.linspace(0,self.p(period,**params) if isinstance(period,str) else period,count)) )
			plot=True

		if 'evolution' in operators:
			results = self.measure.entanglement_fidelity.integrate(times,subspace='logical',operators=operators,params=params)
		else:
			results = self.measure.entanglement_fidelity_eye.integrate(times,subspace='logical',operators=operators,params=params)

		if plot:
			import matplotlib.pyplot as plt
			plt.ion()
			plt.figure(figure)
			plt.subplot(2,1,1)
			plt.title("Entanglement Fidelity")
			plt.plot(times, map(lambda x: results['fidelity'][x],xrange(len(times))) , label='fidelity')

			plt.subplot(2,1,2)
			plt.title("Leakage")
			plt.plot(times, map(lambda x: results['leakage'][x],xrange(len(times))) , label='leakage' )

		return f1

# Custom StateOperator objects used by DQD
class IdentityOperator(StateOperator):
	def __call__(self,t,y,y_b,params={}):
		return y
	def on_attach_to_system(self,system):
		pass
	def change_basis(self,basis):
		return self
	def connected(self,*indices,**params):
		raise NotImplementedError
	@property
	def for_ensemble(self):
		return True
	@property
	def for_state(self):
		return True
	def init(self,*args,**kwargs):
		pass
	def restrict(self, *indicies):
		raise NotImplementedError
	def transform(self, transform_op):
		raise NotImplementedError

class IdealOperator(StateOperator):
	def __call__(self,t,y,y_b,params={}):
		phi00 = np.angle(y_b[1])
		phi01 = np.angle(y_b[2])
		phi10 = np.angle(y_b[3])
		phi11 = np.angle(y_b[4])

		p00, p01, p10, p11 = 0.25*np.array([[3,1,1,-1,1],[1,3,-1,1,-1],[1,-1,3,1,-1],[-1,1,1,3,1]]).dot([phi00,phi01,phi10,phi11,math.pi])

		U = np.diag([1,np.exp(1j*p00),np.exp(1j*p01),np.exp(1j*p10),np.exp(1j*p11),1])
		if len(y.shape) > 1:
			return U.dot(y).dot(U.conjugate().transpose())
		return U.dot(y)
	def on_attach_to_system(self,system):
		pass
	def change_basis(self,basis):
		return self
	def connected(self,*indices,**params):
		raise NotImplementedError
	@property
	def for_ensemble(self):
		return True
	@property
	def for_state(self):
		return True
	def init(self,*args,**kwargs):
		pass
	def restrict(self, *indicies):
		raise NotImplementedError
	def transform(self, transform_op):
		raise NotImplementedError
