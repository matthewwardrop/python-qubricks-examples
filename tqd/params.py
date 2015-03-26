title = "Square Gate (Physical)"
import math
import numpy as np

g = -0.34 # Unitless
hbar = 1.054571726e-34 # J*s
mu_B = 9.274e-24 # J/T

energy_scale = -300e-3*g*mu_B

dimension_scalings = {
	'time': (hbar/energy_scale,'s'),
	'mass':(energy_scale,'kg'),
	'length': (hbar/energy_scale,'m')
}

units_custom = [
	{"name":"tesla","abbr":"T","rel":g*mu_B,"dimensions":{"mass":1,"length":2,"time":-2}},
]

parameters = {

	"B_1": (0,"mT"),
	"B_2": (0,"mT"),
	"B_3": (0,"mT"),

	#"B_1": (lambda B_1_avg,B_1_oh: B_1_avg+B_1_oh,'mT'),
	#"B_2": (lambda B_2_avg,B_2_oh: B_2_avg+B_2_oh,'mT'),
	#"B_3": (lambda B_3_avg,B_3_oh: B_3_avg+B_3_oh,'mT'),
	#"B_4": (lambda B_4_avg,B_4_oh: B_4_avg+B_4_oh,'mT'),
	#"B_5": (lambda B_5_avg,B_5_oh: B_5_avg+B_5_oh,'mT'),
	#"B_6": (lambda B_6_avg,B_6_oh: B_6_avg+B_6_oh,'mT'),
	#"B_avg": (lambda B_1,B_2,B_3,B_4,B_5,B_6: np.mean([B_1,B_2,B_3,B_4,B_5,B_6]), 'mT'),

	#"B_1_avg": (500,"mT"), # 2*B
	#"B_2_avg": (500,"mT"),
	#"B_3_avg": (500,"mT"),
	#"B_4_avg": (500,"mT"),
	#"B_5_avg": (500,"mT"),
	#"B_6_avg": (500,"mT"),

	# Overhauser noise parameters
	#"B_1_oh": (0,"mT"),
	#"B_2_oh": (0,"mT"),
	#"B_3_oh": (0,"mT"),
	#"B_4_oh": (0,"mT"),
	#"B_5_oh": (0,"mT"),
	#"B_6_oh": (0,"mT"),
	"B_oh_stddev": (1.95,'mT'),

	"J_12_avg": (1.65,"{mu}eV"),
	"J_23_avg": (1.65,"{mu}eV"),
	"J_45_avg": (1.65,"{mu}eV"),
	"J_56_avg": (1.65,"{mu}eV"),

	"J_zA": (1.65,"{mu}eV"),
	"J_zB": (1.65,"{mu}eV"),

	# White noise parameters
	"D": (64.745,'{mu}V^2*ns'), # While noise decay in STQ with T_2 approx 1\mus.

	"J_int": (1.65,"{mu}eV"),

	#"J_z": (lambda J_12_avg: J_12_avg,"{mu}eV"),
	"J_12_noiseless": (lambda J_12_avg: J_12_avg,"{mu}eV"),
	"J_23_noiseless": (lambda J_23_avg: J_23_avg,"{mu}eV"),
	"J_45_noiseless": (lambda J_45_avg: J_45_avg,"{mu}eV"),
	"J_56_noiseless": (lambda J_56_avg: J_56_avg,"{mu}eV"),

	# Inter qubit couplings
	"J_16_avg": (0.16,"{mu}eV"),
	"J_25_avg": (0.16,"{mu}eV"),
	"J_34_avg": (0.16,"{mu}eV"),

	"J_16_noiseless": (lambda J_16_avg: J_16_avg,"{mu}eV"),
	"J_25_noiseless": (lambda J_25_avg: J_25_avg,"{mu}eV"),
	"J_34_noiseless": (lambda J_34_avg: J_34_avg,"{mu}eV"),

	#"T": (lambda J_tq: math.pi/4./J_tq, 'ns')  #(J_25_avg/9. + J_25_avg**2/243*(8./J_zA+8./J_zB-5/(J_zA+J_zB))),'ns') #

	# Exponential ansatz parameters
	"J_0": (-0.0001, "{mu}eV"),
	"J_1": (82.71335433934895, "{mu}eV"),
	"e_0": (0.35, "mV"),

	"J_MHz": (lambda J,c_hbar: J/c_hbar/2./math.pi,'MHz'),
	"e_deviation": (1e-9,'{mu}V'),

}

parameter_units = {
	'N_zA': '{mu}eV^2/mV^2',
	'N_zB': '{mu}eV^2/mV^2',
	'N_25': '{mu}eV^2/mV^2',
}
