
import sys
import numpy as np
import math
import matplotlib.pyplot as plt

from qubricks import QuantumSystem
from qubricks.wall import LindbladStateOperator,DummyStateOperator,AmplitudeMeasurement
from qubricks.utility import tensor

# Define the Pauli Matrices
I,X,Y,Z = np.array([[1,0],[0,1]]),np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.array([[1,0],[0,-1]])

class DQD (QuantumSystem):

	def setup_environment(self,**kwargs):
		pass

	def setup_parameters(self):
		# All parameters are configured in params.py. We just print them here.
		self.p.show()

	def setup_states(self):
		self.add_state('ground', [0,0,1,0])
		self.add_state('singlet', np.array([0,1,-1,0])/np.sqrt(2))
		self.add_state('triplet', np.array([0,1,1,0])/np.sqrt(2))

		self.add_subspace('logical',['singlet','triplet'])

	def setup_hamiltonian(self):


		components = {
			'B_1': tensor(Z,I),
			'B_2': tensor(I,Z),
			"J/4": tensor(X,X) + tensor(Y,Y) + tensor(Z,Z) - tensor(I,I),
		}

		return self.Operator(components)

	@property
	def default_derivative_ops(self):
		return ["evolution"]

	def setup_derivative_ops(self):
		pass

	def get_derivative_ops(self,components=None):
		return {

		# High frequency noise in J
		"J_hf": LindbladStateOperator(self.p,
									coefficient='D*N',
									operator= self.Operator( 0.25 * ( tensor(Z,Z) + tensor(X,X) + tensor(Y,Y) - tensor(I,I) ) ) )

		}

	def setup_measurements(self):
		self.add_measurement('amplitude',AmplitudeMeasurement())

	def setup_bases(self):
		pass
