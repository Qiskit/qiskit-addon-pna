# This code is a Qiskit project.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for primary PNA functionality."""

import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_addon_pna import generate_noise_mitigating_observable
from qiskit_aer import AerSimulator
from qiskit_aer.noise.errors import PauliLindbladError


class TestPNA(unittest.TestCase):
    def test_generate_noise_mitigating_observable(self):
        num_qubits = 3
        num_steps = 5
        theta_rx = np.pi / 4
        observable = SparsePauliOp("Z" * num_qubits)
        edges = [(0, 1), (1, 2)]
        rng = np.random.default_rng()

        noise_models = [
            SparsePauliOp(
                ["IXI", "IIX", "IYI", "IIY", "IZI", "IIZ", "IXX", "IYY", "IZZ"],
                rng.uniform(1e-5, 1e-2, size=9),
            ),
            SparsePauliOp(
                ["IXI", "XII", "IYI", "YII", "IZI", "ZII", "XXI", "YYI", "ZZI"],
                rng.uniform(1e-5, 1e-2, size=9),
            ),
        ]
        circuit_noiseless = QuantumCircuit(num_qubits)
        for _ in range(num_steps):
            circuit_noiseless.rx(theta_rx, [i for i in range(num_qubits)])
            for edge in edges:
                circuit_noiseless.sdg(edge)
                circuit_noiseless.ry(np.pi / 2, edge[1])
                circuit_noiseless.cx(edge[0], edge[1])
                circuit_noiseless.ry(-np.pi / 2, edge[1])
        circuit_noiseless.save_density_matrix()

        circuit_noisy = QuantumCircuit(num_qubits)
        for _ in range(num_steps):
            circuit_noisy.rx(theta_rx, [i for i in range(num_qubits)])
            for i, edge in enumerate(edges):
                circuit_noisy.sdg(edge)
                circuit_noisy.ry(np.pi / 2, edge[1])
                circuit_noisy.cx(edge[0], edge[1])
                circuit_noisy.append(
                    PauliLindbladError(noise_models[i].paulis, noise_models[i].coeffs.real),
                    qargs=circuit_noisy.qubits,
                    cargs=circuit_noisy.clbits,
                )
                circuit_noisy.ry(-np.pi / 2, edge[1])

        backend = AerSimulator(method="density_matrix")

        rho_noiseless = backend.run(circuit_noiseless).result().data()["density_matrix"]
        exact_ev = rho_noiseless.expectation_value(observable)

        otilde = generate_noise_mitigating_observable(
            circuit_noisy,
            observable,
            max_err_terms=4**num_qubits,
            max_obs_terms=(4**num_qubits) ** 3,
            search_step=4**num_qubits,
            atol=0.0,
        )
        circuit_noisy.save_density_matrix()
        rho_noisy = backend.run(circuit_noisy).result().data()["density_matrix"]
        noisy_ev = rho_noisy.expectation_value(observable)
        mitigated_ev = rho_noisy.expectation_value(otilde)

        assert not np.isclose(exact_ev, noisy_ev)
        assert np.isclose(exact_ev, mitigated_ev)
