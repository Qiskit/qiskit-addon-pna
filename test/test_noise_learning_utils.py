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

"""Tests for the noise learning utils module."""

import unittest

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Pauli, PauliLindbladMap
from qiskit_addon_pna.pna import (
    _inject_learned_noise_to_boxed_circuit,
    _pauli_lindblad_map_to_layer_error,
)
from qiskit_aer.noise.errors import PauliLindbladError
from samplomatic.annotations import InjectNoise, Twirl
from samplomatic.transpiler import generate_boxing_pass_manager
from samplomatic.utils import find_unique_box_instructions


class TestNoiseInjection(unittest.TestCase):
    def setUp(self):
        self.refs_to_noise_models_left_dressing = {}
        self.refs_to_noise_models_right_dressing = {}
        num_qubits = 4
        qubits = np.arange(num_qubits)

        # create pauli lindblad noise maps
        self.pl = PauliLindbladMap.from_list(
            [
                ("IIIX", 0.1),
                ("IIXI", 0.11),
                ("IZII", 0.2),
                ("ZIII", 0.21),
                ("XZII", 0.3),
                ("IIXZ", 0.31),
                ("IXZI", 0.32),
            ],
            num_qubits=num_qubits,
        )

        # get pauli lindblad error instructions
        pl_qubits_01_23 = PauliLindbladMap.from_list(
            [("IIYX", 0.01), ("YXII", 0.23)], num_qubits=num_qubits
        )
        pl_qubits_12 = PauliLindbladMap.from_list([("ZY", 0.12)], num_qubits=2)
        ple_01_23 = _pauli_lindblad_map_to_layer_error(pl_qubits_01_23)
        ple_12 = _pauli_lindblad_map_to_layer_error(pl_qubits_12)

        #################   LEFT DRESSING   #################
        # left dressed manually boxed circuit
        self.manually_boxed_circuit_left_dressed = QuantumCircuit(num_qubits)
        with self.manually_boxed_circuit_left_dressed.box(
            [Twirl(dressing="left"), InjectNoise(ref="r0", modifier_ref="r0")]
        ):
            self.manually_boxed_circuit_left_dressed.rx(3 * np.pi / 8, 0)
            self.manually_boxed_circuit_left_dressed.sdg(0)
            self.manually_boxed_circuit_left_dressed.rx(3 * np.pi / 8, 1)
            self.manually_boxed_circuit_left_dressed.sdg(1)
            self.manually_boxed_circuit_left_dressed.cz(0, 1)
            self.manually_boxed_circuit_left_dressed.rx(3 * np.pi / 8, 2)
            self.manually_boxed_circuit_left_dressed.sdg(2)
            self.manually_boxed_circuit_left_dressed.rx(3 * np.pi / 8, 3)
            self.manually_boxed_circuit_left_dressed.sdg(3)
            self.manually_boxed_circuit_left_dressed.cz(2, 3)

        with self.manually_boxed_circuit_left_dressed.box(
            [Twirl(dressing="left"), InjectNoise(ref="r1", modifier_ref="r1")]
        ):
            self.manually_boxed_circuit_left_dressed.rx(3 * np.pi / 8, 1)
            self.manually_boxed_circuit_left_dressed.sdg(1)
            self.manually_boxed_circuit_left_dressed.rx(3 * np.pi / 8, 2)
            self.manually_boxed_circuit_left_dressed.sdg(2)
            self.manually_boxed_circuit_left_dressed.cz(1, 2)

        # get unique instructions
        unique_instructions_left = find_unique_box_instructions(
            self.manually_boxed_circuit_left_dressed, undress_boxes=True, normalize_annotations=None
        )
        self.assertEqual(len(unique_instructions_left), 2)

        # expected left dressed noisy (before) circuit
        self.noisy_before_circuit_left_dressed = QuantumCircuit(num_qubits)
        qargs = self.noisy_before_circuit_left_dressed.qubits
        self.noisy_before_circuit_left_dressed.rx(3 * np.pi / 8, qubits)
        self.noisy_before_circuit_left_dressed.sdg(qubits)
        self.noisy_before_circuit_left_dressed.append(ple_01_23, qargs=qargs)
        self.noisy_before_circuit_left_dressed.cz(0, 1)
        self.noisy_before_circuit_left_dressed.cz(2, 3)
        self.noisy_before_circuit_left_dressed.rx(3 * np.pi / 8, qubits[1:3])
        self.noisy_before_circuit_left_dressed.sdg(qubits[1:3])
        self.noisy_before_circuit_left_dressed.append(ple_12, qargs=qargs[1:3])
        self.noisy_before_circuit_left_dressed.cz(1, 2)

        # expected left dressed noisy (after) circuit
        self.noisy_after_circuit_left_dressed = QuantumCircuit(num_qubits)
        qargs = self.noisy_after_circuit_left_dressed.qubits
        self.noisy_after_circuit_left_dressed.rx(3 * np.pi / 8, qubits)
        self.noisy_after_circuit_left_dressed.sdg(qubits)
        self.noisy_after_circuit_left_dressed.cz(0, 1)
        self.noisy_after_circuit_left_dressed.cz(2, 3)
        self.noisy_after_circuit_left_dressed.append(ple_01_23, qargs=qargs)
        self.noisy_after_circuit_left_dressed.rx(3 * np.pi / 8, qubits[1:3])
        self.noisy_after_circuit_left_dressed.sdg(qubits[1:3])
        self.noisy_after_circuit_left_dressed.cz(1, 2)
        self.noisy_after_circuit_left_dressed.append(ple_12, qargs=qargs[1:3])

        # map inject noise refs to pauli lindblad maps
        self.refs_to_noise_models_left_dressing[
            unique_instructions_left[0].operation.annotations[1].ref
        ] = pl_qubits_01_23
        self.refs_to_noise_models_left_dressing[
            unique_instructions_left[1].operation.annotations[1].ref
        ] = pl_qubits_12
        self.assertEqual(len(self.refs_to_noise_models_left_dressing), 2)

        #################   RIGHT DRESSING   #################
        # right dressed manually boxed circuit
        self.manually_boxed_circuit_right_dressed = QuantumCircuit(num_qubits)
        with self.manually_boxed_circuit_right_dressed.box(
            [Twirl(dressing="right"), InjectNoise(ref="r0", modifier_ref="r0")]
        ):
            self.manually_boxed_circuit_right_dressed.cz(0, 1)
            self.manually_boxed_circuit_right_dressed.rx(3 * np.pi / 8, 0)
            self.manually_boxed_circuit_right_dressed.sdg(0)
            self.manually_boxed_circuit_right_dressed.rx(3 * np.pi / 8, 1)
            self.manually_boxed_circuit_right_dressed.sdg(1)
            self.manually_boxed_circuit_right_dressed.cz(2, 3)
            self.manually_boxed_circuit_right_dressed.rx(3 * np.pi / 8, 2)
            self.manually_boxed_circuit_right_dressed.sdg(2)
            self.manually_boxed_circuit_right_dressed.rx(3 * np.pi / 8, 3)
            self.manually_boxed_circuit_right_dressed.sdg(3)

        with self.manually_boxed_circuit_right_dressed.box(
            [Twirl(dressing="right"), InjectNoise(ref="r1", modifier_ref="r1")]
        ):
            self.manually_boxed_circuit_right_dressed.cz(1, 2)
            self.manually_boxed_circuit_right_dressed.rx(3 * np.pi / 8, 1)
            self.manually_boxed_circuit_right_dressed.sdg(1)
            self.manually_boxed_circuit_right_dressed.rx(3 * np.pi / 8, 2)
            self.manually_boxed_circuit_right_dressed.sdg(2)

        # get unique instructions
        unique_instructions_right = find_unique_box_instructions(
            self.manually_boxed_circuit_right_dressed,
            undress_boxes=True,
            normalize_annotations=None,
        )
        self.assertEqual(len(unique_instructions_right), 2)

        # overwrite old map with new one
        self.refs_to_noise_models_right_dressing[
            unique_instructions_right[0].operation.annotations[1].ref
        ] = pl_qubits_01_23
        self.refs_to_noise_models_right_dressing[
            unique_instructions_right[1].operation.annotations[1].ref
        ] = pl_qubits_12
        self.assertEqual(len(self.refs_to_noise_models_right_dressing), 2)

        # expected right dressed noisy (before) circuit
        self.noisy_before_circuit_right_dressed = QuantumCircuit(num_qubits)
        qargs = self.noisy_before_circuit_right_dressed.qubits
        self.noisy_before_circuit_right_dressed.append(ple_01_23, qargs=qargs)
        self.noisy_before_circuit_right_dressed.cz(0, 1)
        self.noisy_before_circuit_right_dressed.cz(2, 3)
        self.noisy_before_circuit_right_dressed.rx(3 * np.pi / 8, qubits)
        self.noisy_before_circuit_right_dressed.sdg(qubits)
        self.noisy_before_circuit_right_dressed.append(ple_12, qargs=qargs[1:3])
        self.noisy_before_circuit_right_dressed.cz(1, 2)
        self.noisy_before_circuit_right_dressed.rx(3 * np.pi / 8, qubits[1:3])
        self.noisy_before_circuit_right_dressed.sdg(qubits[1:3])

        # expected right dressed noisy (after) circuit
        self.noisy_after_circuit_right_dressed = QuantumCircuit(num_qubits)
        qargs = self.noisy_after_circuit_right_dressed.qubits
        self.noisy_after_circuit_right_dressed.cz(0, 1)
        self.noisy_after_circuit_right_dressed.cz(2, 3)
        self.noisy_after_circuit_right_dressed.append(ple_01_23, qargs=qargs)
        self.noisy_after_circuit_right_dressed.rx(3 * np.pi / 8, qubits)
        self.noisy_after_circuit_right_dressed.sdg(qubits)
        self.noisy_after_circuit_right_dressed.cz(1, 2)
        self.noisy_after_circuit_right_dressed.append(ple_12, qargs=qargs[1:3])
        self.noisy_after_circuit_right_dressed.rx(3 * np.pi / 8, qubits[1:3])
        self.noisy_after_circuit_right_dressed.sdg(qubits[1:3])

    def test_manual_boxed_circuit_is_set_correctly(self):
        """[SANITY CHECK] Check that the manually boxed circuit is set correctly by
        comparing it to Samplomatic's boxing pass manager.
        """
        # left dressed circuit
        circuit_left_dressed = QuantumCircuit(4)
        circuit_left_dressed.rx(3 * np.pi / 8, [0, 1, 2, 3])
        circuit_left_dressed.sdg([0, 1, 2, 3])
        circuit_left_dressed.cz(0, 1)
        circuit_left_dressed.cz(2, 3)
        circuit_left_dressed.barrier([0, 1, 2, 3])
        circuit_left_dressed.rx(3 * np.pi / 8, [1, 2])
        circuit_left_dressed.sdg([1, 2])
        circuit_left_dressed.cz(1, 2)
        circuit_left_dressed.barrier([1, 2])

        # box the circuit
        pm = generate_boxing_pass_manager(
            enable_gates=True,
            enable_measures=False,
            measure_annotations="all",
            twirling_strategy="active",
            inject_noise_targets="gates",
            inject_noise_strategy="uniform_modification",
            remove_barriers=True,
        )
        boxed_circuit_left_dressed = pm.run(circuit_left_dressed)
        boxed_circuit_left_dressed.data.pop()  # remove final empty box by undoing AddTerminalRightDressedBoxes

        self.assertEqual(boxed_circuit_left_dressed, self.manually_boxed_circuit_left_dressed)

    def test_pauli_lindblad_map_to_layer_error(self):
        pl_sparse_list = [
            ("X", [0], 0.1),
            ("X", [1], 0.11),
            ("Z", [2], 0.2),
            ("Z", [3], 0.21),
            ("ZX", [2, 3], 0.3),
            ("ZX", [0, 1], 0.31),
            ("ZX", [1, 2], 0.32),
        ]
        self.assertEqual(self.pl.to_sparse_list(), pl_sparse_list)

        ple = _pauli_lindblad_map_to_layer_error(self.pl)
        ple_expected = PauliLindbladError(
            generators=[
                Pauli("IIIX"),
                Pauli("IIXI"),
                Pauli("IZII"),
                Pauli("ZIII"),
                Pauli("XZII"),
                Pauli("IIXZ"),
                Pauli("IXZI"),
            ],
            rates=[0.1, 0.11, 0.2, 0.21, 0.3, 0.31, 0.32],
        )
        self.assertEqual(ple, ple_expected)

    def test_inject_learned_noise_to_boxed_circuit(self):
        ### test noise injection to a circuit with only left dressed boxes
        # inject BEFORE 2q layers
        noisy_before_circuit = _inject_learned_noise_to_boxed_circuit(
            self.manually_boxed_circuit_left_dressed,
            self.refs_to_noise_models_left_dressing,
            inject_noise_before=True,
        )
        self.assertEqual(noisy_before_circuit, self.noisy_before_circuit_left_dressed)

        # inject AFTER 2q layers
        noisy_after_circuit = _inject_learned_noise_to_boxed_circuit(
            self.manually_boxed_circuit_left_dressed,
            self.refs_to_noise_models_left_dressing,
            inject_noise_before=False,
        )
        self.assertEqual(noisy_after_circuit, self.noisy_after_circuit_left_dressed)

        ### test noise injection to a circuit with only right dressed boxes
        # inject BEFORE 2q layers
        noisy_before_circuit = _inject_learned_noise_to_boxed_circuit(
            self.manually_boxed_circuit_right_dressed,
            self.refs_to_noise_models_right_dressing,
            inject_noise_before=True,
        )
        self.assertEqual(noisy_before_circuit, self.noisy_before_circuit_right_dressed)

        # inject AFTER 2q layers
        noisy_after_circuit = _inject_learned_noise_to_boxed_circuit(
            self.manually_boxed_circuit_right_dressed,
            self.refs_to_noise_models_right_dressing,
            inject_noise_before=False,
        )
        self.assertEqual(noisy_after_circuit, self.noisy_after_circuit_right_dressed)
