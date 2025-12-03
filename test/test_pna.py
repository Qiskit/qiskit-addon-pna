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
from qiskit.quantum_info import PauliLindbladMap, SparsePauliOp
from qiskit_addon_pna import generate_noise_mitigating_observable
from qiskit_aer import AerSimulator
from qiskit_aer.noise.errors import PauliLindbladError
from samplomatic.annotations import InjectNoise
from samplomatic.transpiler import generate_boxing_pass_manager
from samplomatic.utils import get_annotation


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
                ["XI", "IX", "YI", "IY", "ZI", "IZ", "XX", "YY", "ZZ"],
                rng.uniform(1e-5, 1e-2, size=9),
            ),
            SparsePauliOp(
                ["IX", "XI", "IY", "YI", "IZ", "ZI", "XX", "YY", "ZZ"],
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

        circuit_noisy = QuantumCircuit(num_qubits)
        for _ in range(num_steps):
            circuit_noisy.rx(theta_rx, [i for i in range(num_qubits)])
            for i, edge in enumerate(edges):
                circuit_noisy.sdg(edge)
                circuit_noisy.ry(np.pi / 2, edge[1])
                circuit_noisy.cx(edge[0], edge[1])
                circuit_noisy.append(
                    PauliLindbladError(noise_models[i].paulis, noise_models[i].coeffs.real),
                    qargs=circuit_noisy.qubits[edge[0] : edge[1] + 1],
                    cargs=circuit_noisy.clbits[edge[0] : edge[1] + 1],
                )
                circuit_noisy.ry(-np.pi / 2, edge[1])

        backend = AerSimulator(method="density_matrix")

        circuit_noiseless_cp = circuit_noiseless.copy()
        circuit_noiseless_cp.save_density_matrix()
        rho_noiseless = backend.run(circuit_noiseless_cp).result().data()["density_matrix"]
        exact_ev = rho_noiseless.expectation_value(observable)

        otilde = generate_noise_mitigating_observable(
            circuit_noisy,
            observable,
            max_err_terms=4**num_qubits,
            max_obs_terms=(4**num_qubits) ** 3,
            search_step=4**num_qubits,
            atol=0.0,
        )
        circuit_noisy_cp = circuit_noisy.copy()
        circuit_noisy_cp.save_density_matrix()
        rho_noisy = backend.run(circuit_noisy_cp).result().data()["density_matrix"]
        noisy_ev = rho_noisy.expectation_value(observable)
        mitigated_ev = rho_noisy.expectation_value(otilde)

        assert not np.isclose(exact_ev, noisy_ev)
        assert np.isclose(exact_ev, mitigated_ev)

        boxing_pm = generate_boxing_pass_manager(
            enable_gates=True,
            inject_noise_targets="gates",
            inject_noise_strategy="individual_modification",
        )
        boxed_circuit = boxing_pm.run(circuit_noiseless)
        boxed_circuit.data = boxed_circuit.data[:-1]
        boxed_circuit.ry(-np.pi / 2, 2)

        paulis_r0 = [p.to_label() for p in noise_models[0].paulis]
        coeffs_r0 = noise_models[0].coeffs.real
        paulis_r1 = [p.to_label() for p in noise_models[1].paulis]
        coeffs_r1 = noise_models[1].coeffs.real

        nm_0 = PauliLindbladMap.from_list(
            [(p, c) for p, c in zip(paulis_r0, coeffs_r0, strict=True)]
        )
        nm_1 = PauliLindbladMap.from_list(
            [(p, c) for p, c in zip(paulis_r1, coeffs_r1, strict=True)]
        )
        ref1 = get_annotation(boxed_circuit[0].operation, InjectNoise).ref
        ref2 = get_annotation(boxed_circuit[1].operation, InjectNoise).ref
        otilde = generate_noise_mitigating_observable(
            boxed_circuit,
            observable,
            refs_to_noise_model_map={ref1: nm_0, ref2: nm_1},
            max_err_terms=4**num_qubits,
            max_obs_terms=(4**num_qubits) ** 3,
            inject_noise_before=False,
            search_step=4**num_qubits,
            num_processes=8,
            atol=0.0,
        )
        mitigated_ev = rho_noisy.expectation_value(otilde)
        assert np.isclose(exact_ev, mitigated_ev)
        
        otilde = generate_noise_mitigating_observable(
            circuit_noisy,
            observable,
            max_err_terms=4**num_qubits,
            max_obs_terms=(4**num_qubits) ** 3,
            search_step=4**num_qubits,
            atol=0.0,
            batch_size=4,
        )
        mitigated_ev = rho_noisy.expectation_value(otilde)

        assert np.isclose(exact_ev, mitigated_ev, atol=1e-3)

    def test_pna_inputs(self):
        qc = QuantumCircuit(2)
        spo = SparsePauliOp("Z")
        with self.assertRaises(ValueError):
            generate_noise_mitigating_observable(qc, spo, max_err_terms=1, max_obs_terms=1)
        qc = QuantumCircuit(1)
        with self.assertRaises(ValueError):
            generate_noise_mitigating_observable(qc, spo, max_err_terms=1, max_obs_terms=1, batch_size=0)
        with self.assertRaises(ValueError):
            generate_noise_mitigating_observable(qc, spo, max_err_terms=1, max_obs_terms=1, num_processes=0)
        spo = SparsePauliOp(["Z", "X"])
        with self.assertRaises(ValueError):
            generate_noise_mitigating_observable(qc, spo, max_err_terms=1, max_obs_terms=1)
        spo = SparsePauliOp("Z", 1.0+1.0j)
        with self.assertRaises(ValueError):
            generate_noise_mitigating_observable(qc, spo, max_err_terms=1, max_obs_terms=1)
