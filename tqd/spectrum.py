import sys
sys.path.insert(0,"/Storage/Repositories/python-qubricks")
sys.path.insert(0,"/Storage/Repositories/python-parameters")

from matplotlib import pyplot as plt
import numpy as np

def numerical_energies(qs, op, states, ranges, input=None, output=None, params={}):
    '''
    Numerical energies are used to calculate properly expected two-qubit phases.
    '''
    op = op.change_basis(qs.basis(output))
    states = qs.subspace(states, input=input, output=output, params=params)
    rvals = qs.p.range(*ranges.keys(),**ranges)

    #Solving eigenstates
    #print "EIGENSTATES"
    evals,evecs = np.linalg.eig(op(**params))
    evecs = evecs[:,np.argsort(evals)]
    evals = np.sort(evals)
    indicies = []

    #print evecs.shape

    #print "GETTING ARGMAX"
    for state in states:
        #print np.dot(state,evecs)
        indicies.append(np.argmax(np.dot(state,evecs)))

    #print indicies



    if type(rvals) != dict:
        rvals = {ranges.keys()[0]: rvals}

    results = np.zeros((len(states),len(rvals.values()[0])))
    for i in xrange(len(rvals.values()[0])):
        vals = {}
        for val in rvals:
            vals[val] = rvals[val][i]
        evals = sorted(np.linalg.eigvals(op(**vals)))
        results[:,i] = [evals[indicies[j]] for j in xrange(len(indicies))]

    #print results
    return results


from tqd import TQD

t = TQD('params.py')
H = t.H('coupling')

window = 1
states = ['l0','l1','ml0','ml1','Q','q','u']
results = numerical_energies(t, H, states, {'e_12':((-window,'mV'),(window,'mV'),100),'e_23':((window,'mV'),(-window,'mV'),100)},input=None,output='working')

for i in xrange(results.shape[0]):
    plt.plot(np.linspace(-window,window,100),t.p.convert(results[i,:],output='{mu}eV'), label="$\left|%s\\right>$"% states[i])
plt.legend()
plt.ylim( (-300,10) )
plt.grid()
plt.xlabel('$\\varepsilon\,(mV)$')
plt.ylabel('Energy $(\mu eV)$')
plt.title("Resonant Exchange Qubit Spectrum")

plt.savefig('spectrum.pdf')
