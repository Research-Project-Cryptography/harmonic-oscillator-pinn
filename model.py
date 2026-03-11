import torch
import torch.nn as nn
import pennylane as qml
import matplotlib.pyplot as plt

class FCN(nn.Module):
    "Defines a connected network"
    
    def __init__(self, N_INPUT: int, N_OUTPUT: int, N_HIDDEN: int, N_LAYERS: int):
        super().__init__()
        activation = nn.Tanh

        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

class Hybrid_QN(nn.Module):
    '''
    Quantum layer -> Classical 1 layer
    '''

    def __init__(self, Q_DEVICE, INPUT_DIM:int, OUTPUT_DIM: int, N_QUBITS:int, N_LAYERS:int = 1, ROTATION:str = 'Ry'):
        super().__init__()

        self.wires = list(range(N_QUBITS))
        
        weight_shape = {
            'weights': self.quantum_circuit_shape(wires=N_QUBITS, n_layers=N_LAYERS, rot=ROTATION)
        }

        qc = self.quantum_circuit(wires = self.wires, rot=ROTATION)
        self.q_node = qml.QNode(qc, Q_DEVICE)


        self.input_layer= nn.Linear(INPUT_DIM, N_QUBITS)
        self.quantum_layer = qml.qnn.TorchLayer(self.q_node, weight_shape)
        # self.quantum_layer = nn.Linear(N_QUBITS, N_QUBITS)
        self.output_layer = nn.Linear(N_QUBITS, OUTPUT_DIM)

    def quantum_circuit_shape(self, wires, n_layers=1, rot='Ry'):
        if rot == 'Ry':
            shape = qml.BasicEntanglerLayers.shape(n_layers=n_layers, n_wires=wires)
        elif  rot == 'Rxyz':
            shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=wires)
        return shape

    def quantum_circuit(self, wires, rot='Ry'):
        def _quantum_circuit(inputs, weights): 
            # Prepare state H 
            [qml.Hadamard(i) for i in wires]

            # Encode classical -> quantum
            qml.AngleEmbedding(inputs, rotation='Y', wires=wires)

            # Process
            if rot == 'Ry':
                qml.BasicEntanglerLayers(weights, rotation=qml.RY, wires=wires)
            elif rot == 'Rxyz':
                qml.StronglyEntanglingLayers(weights, wires=wires)
            
            # Measurement quantum -> classical
            return [qml.expval(qml.PauliZ(wires=w)) for w in wires]
        return _quantum_circuit
    
    def draw_circuit(self, fontsize=20, style='pennylane', scale=None, title=None, decimals=2):
        data_in = torch.linspace(1, 2, len(self.wires))

        @torch.no_grad()
        def _draw_circuit(*args, **kwargs):
            nonlocal fontsize, style, scale, title
            qml.drawer.use_style(style)
            fig, ax = qml.draw_mpl(self.q_node, decimals=decimals)(*args, **kwargs)
            if scale is not None:
                fig.set_dpi(fig.get_dpi() * scale)
            if title is not None:
                fig.suptitle(title, fontsize=fontsize)
            plt.show()
        _draw_circuit(data_in, self.quantum_layer.weights)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.quantum_layer(x)
        x = self.output_layer(x)
        return x


class Pure_QN(nn.Module):
    def __init__(self, DEVICE, N_INPUT, N_QUBITS, N_INPUT_WIRES, N_LAYER, N_OUTPUT_WIRES, ROTATION= 'Ry'):
        super().__init__()
        self.N_INPUT=N_INPUT
        self.N_QUBITS=N_QUBITS
        self.N_INPUT_WIRES=N_INPUT_WIRES
        self.N_LAYER=N_LAYER
        self.N_OUTPUT_WIRES=N_OUTPUT_WIRES


        # self.fcs = nn.Linear(N_INPUT, len(N_INPUT_WIRES)).requires_grad_(False)

        # Quantum Circuit Configurations
        weight_shape = {
            'weights': self.quantum_circuit_shape(wires=N_QUBITS, n_layers=N_LAYER, rot=ROTATION)
        }

        wires = list(range(N_QUBITS))

        qc = self.quantum_circuit(wires=wires, input_wires=N_INPUT_WIRES, output_wires=N_OUTPUT_WIRES, rot=ROTATION)

        q_node = qml.QNode(qc, DEVICE)
        self.q_node = q_node

        # Quantum layer and model

        quantum_torch_layer = qml.qnn.TorchLayer(q_node, weight_shape)

        self.q_layer = quantum_torch_layer

    def quantum_circuit_shape(self, wires, n_layers=1, rot='Ry'):
        if rot == 'Ry':
            shape = qml.BasicEntanglerLayers.shape(n_layers=n_layers, n_wires=wires)
        elif  rot == 'Rxyz':
            shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=wires)
        return shape

    def quantum_circuit(self, wires, input_wires, output_wires, rot='Ry'):
        def _quantum_circuit(inputs, weights): 
            # Prepare state H 
            [qml.Hadamard(i) for i in wires]

            # Encode classical -> quantum
            qml.AngleEmbedding(inputs, rotation='Y', wires=input_wires)

            # Process
            if rot == 'Ry':
                qml.BasicEntanglerLayers(weights, rotation=qml.RY, wires=wires)
            elif rot == 'Rxyz':
                qml.StronglyEntanglingLayers(weights, wires=wires)
            
            # Measurement quantum -> classical
            return [qml.expval(qml.PauliZ(wires=w)) for w in output_wires]
        return _quantum_circuit
    
    def draw_circuit(self, fontsize=20, style='pennylane', scale=None, title=None, decimals=2):
        data_in = torch.linspace(1, 2, len(self.N_INPUT_WIRES))

        @torch.no_grad()
        def _draw_circuit(*args, **kwargs):
            nonlocal fontsize, style, scale, title
            qml.drawer.use_style(style)
            fig, ax = qml.draw_mpl(self.q_node, decimals=decimals)(*args, **kwargs)
            if scale is not None:
                fig.set_dpi(fig.get_dpi() * scale)
            if title is not None:
                fig.suptitle(title, fontsize=fontsize)
            plt.show()
        _draw_circuit(data_in, list(self.parameters()))
        
    def forward(self, x):
        # if self.fcs != None:
        #     x = self.fcs(x)
        x = self.q_layer(x)
        return x