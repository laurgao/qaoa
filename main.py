import networkx as nx
import numpy as np
import scipy

# omg i have to construct the quantum circuit from scratch


edges = [(0,1,0.7),(2,3,5), (0,2,2.2), (0,3,2.5)]

graph = nx.Graph()
graph.add_weighted_edges_from(edges)

n = len(graph.nodes)

n_layers = 2
num_iters = 100


def inference(out):
    opt_params = out["x"]
    return forward(opt_params)
    # figure stuff out idk

def train():
    init_params = np.random.uniform(-np.pi, np.pi, n_layers * 2)
    from scipy.optimize import minimize
    out = minimize(cost_function, x0=init_params, method="COBYLA", options={'maxiter':num_iters}) 
    # This optimizer changes our initialized params from a 2x4 array into a 1x8 array

    print(f'Out: {out}')
    return out

z = np.array([
    [1, 0],
    [0, -1]
])

def generate_maxcut_hamiltonian(graph: nx.Graph) -> np.ndarray:
                # hamiltonian += PauliTerm({i: "Z", j: "Z"}, weight)
            # shift -= weight

        # return 0.5 * (hamiltonian + shift)
    ham = np.zeros([2**n, 2**n])
    for edge in edges:
        q1, q2, weight = edge
        ham -= np.kron(np.kron(np.kron(np.kron(np.identity(2**q1), z), np.identity(2**(q2-q1-1))), z), np.identity(2**(n-q2-1))) * weight /2
        ham += weight / 2
    return ham
    


def cost_function(params: np.ndarray) -> float:
    state = forward(params, n_layers=n_layers)
    hamiltonian = generate_maxcut_hamiltonian(graph)

    # state = np.abs(state)
    expval = np.matmul(state, np.matmul(hamiltonian, state))
    # probabilities = state ** 2
    # expvals = expectation_value_per_bitstring
    print(np.abs(expval))

    return np.abs(expval)

def kron(matrix: np.ndarray, n_times: int) -> np.ndarray:
    if n_times == 2:
        return np.kron(matrix, matrix)
    else:
        return np.kron(matrix, kron(matrix, n_times - 1))

x = np.array([
    [0, 1],
    [1, 0]
])


def forward(params: np.ndarray, n_layers: int=1) -> np.ndarray:
    # convert graph to circuit
    
    state = np.zeros(2 ** n)
    state[0] = 1
    # apply hadamard gate everywhere
    H1 = np.array([
        [1, 1],
        [1, -1]
    ]) / (2 ** .5)
    Hn = kron(H1, n)

    state = np.matmul(Hn, state)

    for l in range(n_layers):
        state = np.matmul(cost_layer(params[2*l]), state)
        state = np.matmul(mixer_layer(params[2*l+1]), state)
    
    return state


def rx(theta: float) -> np.ndarray:
    return np.array([
        [ np.cos(theta/2), -1j*np.sin(theta/2) ],
        [ -1j*np.sin(theta/2), np.cos(theta/2) ]
    ])

def rz(theta: float) -> np.ndarray:
    return np.array([
        [np.exp(-1j*theta/2), 0],
        [0, np.exp(1j*theta/2)]
    ]) 

# Mixer layer with parameter beta
def mixer_layer(beta: float) -> np.ndarray:
    return kron(rx(beta), n)
    # kron rx(beta) n times
    # for qubit in range(n):

        # each rx has the same beta per layer
        

        # qml.RX(2 * beta, wires=wire)


# Cost layer with parameter gamma
def cost_layer(gamma) -> np.ndarray:
    for edge in edges:
        wire1 = edge[0]
        wire2 = edge[1]
        weight = edge[2]

        # cnot = np.kron(np.identity(2 ** wire1))
        ket_0 = np.array([1, 0])
        ket_1 = np.array([0, 1])
        proj_0 = np.outer(ket_0, ket_0)
        proj_1 = np.outer(ket_1, ket_1)
        cnot_1 = np.kron(np.kron(np.identity(2**wire1), proj_0), np.identity(2**(n-wire1-1)))
        temp = np.kron(np.identity(2**wire1), proj_1)
        temp2 = np.kron(temp, np.identity(2**(wire2-wire1-1)))
        cnot_2 = np.kron(np.kron(temp2, x), np.identity(2**(n-wire2-1)))
        cnot = cnot_1 + cnot_2
        rz_q2 = rz(gamma * weight)
        rz_total = np.kron(np.kron(np.identity(2**(wire2)), rz_q2), np.identity(2**(n-wire2-1)))

        return cnot @ rz_total @ cnot

        # qml.CNOT(wires=[wire1, wire2])
        # qml.RZ(gamma*weight, wires=wire2) # Multiply gamma by the weight - this is the first algorithmetic change from the unweighted maxcut code
        # qml.CNOT(wires=[wire1, wire2])


if __name__ == "__main__":
    out = train()
    state = forward(out["x"])

    probs = np.abs(state**2)
    print(probs)
    print(np.sum(probs))

    # make graph idk
    import matplotlib.pyplot as plt

    xticks = range(0, 16)
    xtick_labels = list(map(lambda x: format(x, "04b"), xticks))
    bins = np.arange(0, 17) - 0.5

    plt.title(f"{n_layers} layers")
    plt.xlabel("bitstrings")
    plt.ylabel("probability")
    plt.xticks(xticks, xtick_labels, rotation="vertical")
    plt.hist(probs, bins=bins)

    plt.tight_layout()
    plt.show()
