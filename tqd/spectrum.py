import numpy as np
from matplotlib import pyplot as plt

from qubricks.analysis import energy_spectrum

from tqd import TQD

t = TQD('params.py')
H = t.H('coupling')

window = 1
states = ['l0','l1','ml0','ml1','Q','q','u']

results = energy_spectrum(
                    system=t,
                    states=states,
                    ranges={'e_12':((-window,'mV'),(window,'mV'),100),'e_23':((window,'mV'),(-window,'mV'),100)},
                    input=None,
                    output='working',
                    components=['coupling'])

for i in xrange(results.shape[0]):
    plt.plot(np.linspace(-window,window,100),t.p.convert(results[i,:],output='{mu}eV'), label="$\left|%s\\right>$"% states[i])
plt.legend()
plt.ylim( (-300,10) )
plt.grid()
plt.xlabel('$\\varepsilon\,(mV)$')
plt.ylabel('Energy $(\mu eV)$')
plt.title("Resonant Exchange Qubit Spectrum")

plt.savefig('spectrum.pdf')
