import math
g = -0.44 # Unitless

hbar = 1.054571726e-34 #,'J*s'
mu_B = 9.274e-24 #,'J/T'
energy_scale = -300e-3*g*mu_B # J,  Arbitrary energy scale

# Scalings chosen to make hbar = 1
dimension_scalings = {
	'time': (hbar/energy_scale,'s'),
	'mass':(energy_scale,'kg'),
	'length': (hbar/energy_scale,'m')
}

# Override mT unit, so that it reports itself as the Zeeman energy, rather than a magnetic field.
units_custom = [
	{"names":"tesla","abbr":"T","rel":g*mu_B,"dimensions":{"mass":1,"length":2,"time":-2}},
]

parameters = {
	# Magnetic field parameters
	"B_0": (500,"mT"),
	"D_B": (200,"mT"),
	"D_12": (0,"mT"),
	"D_34": (0,"mT"),
	
	"B_1": (lambda B_0,D_B,D_12: B_0+0.5*D_B+0.5*D_12,"mT"), # Energy gap due to Zeeman splitting with field of 500mT. Corresponds to background field of 250mT
	"B_2": (lambda B_0,D_B,D_12: B_0+0.5*D_B-0.5*D_12,"mT"),
	"B_3": (lambda B_0,D_B,D_34: B_0-0.5*D_B+0.5*D_34,"mT"),
	"B_4": (lambda B_0,D_B,D_34: B_0-0.5*D_B-0.5*D_34,"mT"),

	# Exchange coupling parameters
	"J_14_avg": (0.16,"{mu}eV"),
	"J_23_avg": (0.16,"{mu}eV"),
	
	# Ansatze for J
	"exp_J_0": (-0.0001, "{mu}eV"),
	"exp_J_1": (82.71335433934895, "{mu}eV"),
	"exp_e_0": (0.35, "mV"),
	"hyp_a": (-0.148873, '{mu}eV*mV'),
	"hyp_b": (-0.492732, 'mV'),
	"hyp_c": (-0.00001, '{mu}eV'),

	# Ideal gate operation period
	"T": (lambda J_23_avg,J_14_avg:math.pi/(J_23_avg+J_14_avg),'ns'),

	# Charge noise parameters
	"de_23": (0,'mV'), # Pseudo-static detunings
	"de_14": (0,'mV'),
	
	"e_deviation": (10.1811968615,'{mu}V'), # Pseudo-static noise in gate detuning for a T2* of 100 ns in single qubit recoveries
	"D_23": (64.785,'{mu}V^2*ns'), # Power spectral density for a T2 of 1{mu}s in single qubit recoveries
	"D_14": (64.785,'{mu}V^2*ns'), # //

	# Utility parameters
	"J_MHz": (lambda J,c_hbar: J/c_hbar/2./math.pi,'MHz'), # Conversion utility from Energy units to frequencies
}

# Some parameteres are not defined in dqd.py because they depend on runtime logic. We specify their
# units here.
parameters_units = {
	't': 'ns',
	'e_23': 'mV',
	'e_14': 'mV',
	'N_23': '{mu}eV^2/mV^2',
	'N_14': '{mu}eV^2/mV^2',
	'J_23': '{mu}eV',
	'J_14': '{mu}eV',
}
