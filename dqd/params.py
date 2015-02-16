title = "Square Gate (Physical)"

import math
hbar = 1.054571726e-34 # 'J*s'
mu_B = 9.274e-24 # 'J/T'
g = -0.44 # Unitless
energy_scale = -300e-3*g*mu_B

dimension_scalings = {
	'time': (hbar/energy_scale,'s'),
	'mass':(energy_scale,'kg'),
	'length': (hbar/energy_scale,'m')
}

def e(J_base,J_0,J_1,e_0,e=None):
	if e is None:
		return e_0*math.log( (J_base - J_0) / J_1 )
	return J_0 + J_1*math.exp( e / e_0 ), J_0, J_1, e_0

def J(e,de,J_0,J_1,e_0):
	return J_0 + J_1*math.exp( ( e + de ) / e_0 )

parameters = {

	"J_base": (0.32,"{mu}eV"),
	'T': (lambda J_base,c_hbar:2*math.pi*c_hbar/J_base,'ns'), # Ideal gate time

	# Parameters for the exponential ansatz for J
	# J = J_0 + J_1 exp (e / e_0)
	"J_0": (-0.0001, "{mu}eV"), # Tiny offset to ensure invertible at e=0
	"J_1": (82.71335433934895, "{mu}eV"),
	"e_0": (0.35, 'mV'),
	"N": (lambda J,J_0,e_0: (J-J_0)**2/e_0**2, '{mu}eV^2/mV^2'),

	# Detuning e as an invertible function of J
	"e": ( e, 'mV' ),
	"de": (0,'mV'),

	# J with noise possible in e (non-invertible)
	"J": (J, '{mu}eV'),

	# Noise scaling parameters
	"T_2": (1,'{mu}s'), # T_2 when J=0.32 {\mu}eV
	'D': (lambda c_hbar,N,T_2: 2*c_hbar**2/T_2/N, 'mV^2*ns'), # The spectral density of fluctuations

	"T_2s": (100,'ns'), # T_2* when J=0.32 {\mu}eV
	'e_deviation': (lambda J,e_0,c_hbar,T_2s: math.sqrt(2)*e_0*c_hbar/J/T_2s, 'mV'), # Standard devation of e

	# Magnetic field
	"B_1": (500,"mT"),
	"B_2": (500,"mT"),
}

parameters_units = {
	't': 'ns'
}
