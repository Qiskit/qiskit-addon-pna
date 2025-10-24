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

from qiskit.circuit import BoxOp, ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_addon_pna.utils import (
    _unbox_box,
    debox_circuit,
    inject_learned_noise_to_boxed_circuit,
)
from samplomatic.annotations import Twirl


class TestUnboxBox(unittest.TestCase):
    def setUp(self):
        qubits = 3
        qr = QuantumRegister(qubits)

        #       ┌───┐
        # q0_0: ┤ X ├─■──■────
        #       ├───┤ │  │
        # q0_1: ┤ X ├─■──┼──■─
        #       ├───┤    │  │
        # q0_2: ┤ X ├────■──■─
        #       └───┘
        qc1 = QuantumCircuit(qr)
        qc1.x(qr)
        qc1.cz(0, 1)
        qc1.cz(0, 2)
        qc1.cz(1, 2)

        self.box_op = BoxOp(body=qc1, annotations=[Twirl()])
        self.box_op_no_twirl = BoxOp(body=qc1)
        self.circ_w_no_barriers = qc1

        #       ┌───┐ twirled_layer           twirled_layer
        # q0_0: ┤ X ├───────░────────■──■───────────░───────
        #       ├───┤       ░        │  │           ░
        # q0_1: ┤ X ├───────░────────■──┼──■────────░───────
        #       ├───┤       ░           │  │        ░
        # q0_2: ┤ X ├───────░───────────■──■────────░───────
        #       └───┘       ░                       ░
        qc2 = QuantumCircuit(qr)
        qc2.x(qr)
        qc2.barrier(label="twirled_layer")
        qc2.cz(0, 1)
        qc2.cz(0, 2)
        qc2.cz(1, 2)
        qc2.barrier(label="twirled_layer")
        self.circ_w_barriers = qc2

        #        twirled_layer ┌───┐          twirled_layer
        # q0_0: ───────░───────┤ X ├─■──■───────────░───────
        #              ░       ├───┤ │  │           ░
        # q0_1: ───────░───────┤ X ├─■──┼──■────────░───────
        #              ░       ├───┤    │  │        ░
        # q0_2: ───────░───────┤ X ├────■──■────────░───────
        #              ░       └───┘                ░
        qc3 = QuantumCircuit(qr)
        qc3.barrier(label="twirled_layer")
        qc3.x(qr)
        qc3.cz(0, 1)
        qc3.cz(0, 2)
        qc3.cz(1, 2)
        qc3.barrier(label="twirled_layer")
        self.circ_w_barriers_outside = qc3

    def test_unbox_box(self):
        new_circ = QuantumCircuit.copy_empty_like(self.circ_w_no_barriers)
        new_circ = _unbox_box(
            new_circ, box=self.box_op, only_2q_gates=False, include_barriers=False
        )
        self.assertEqual(self.circ_w_no_barriers, new_circ)

        new_circ = QuantumCircuit.copy_empty_like(self.circ_w_no_barriers)
        new_circ = _unbox_box(new_circ, box=self.box_op, only_2q_gates=True, include_barriers=False)
        self.assertEqual(self.circ_w_no_barriers, new_circ)

        new_circ = QuantumCircuit.copy_empty_like(self.circ_w_barriers)
        new_circ = _unbox_box(new_circ, box=self.box_op, only_2q_gates=True, include_barriers=True)
        self.assertEqual(self.circ_w_barriers, new_circ)

        new_circ = QuantumCircuit.copy_empty_like(self.circ_w_barriers)
        new_circ = _unbox_box(new_circ, box=self.box_op, only_2q_gates=False, include_barriers=True)
        self.assertEqual(self.circ_w_barriers_outside, new_circ)

        for only_2q_gates, include_barriers in [
            (True, True),
            (False, True),
            (True, False),
            (False, False),
        ]:
            new_circ = QuantumCircuit.copy_empty_like(self.circ_w_no_barriers)
            new_circ = _unbox_box(
                new_circ,
                box=self.box_op_no_twirl,
                only_2q_gates=only_2q_gates,
                include_barriers=include_barriers,
            )
            self.assertEqual(self.circ_w_no_barriers, new_circ)


class TestDebox(unittest.TestCase):
    def setUp(self):
        qubits = 3
        reps = 3

        #            ┌─────── ┌───┐    ───────┐       ┌─────── ┌───┐    ───────┐       ┌─────── ┌───┐    ───────┐
        # q_0: ──────┤        ┤ X ├─■─        ├───────┤        ┤ X ├─■─        ├───────┤        ┤ X ├─■─        ├─
        #            │        ├───┤ │         │       │        ├───┤ │         │       │        ├───┤ │         │
        # q_1: ──────┤ Box-0  ┤ X ├─■─  End-0 ├───────┤ Box-0  ┤ X ├─■─  End-0 ├───────┤ Box-0  ┤ X ├─■─  End-0 ├─
        #      ┌────┐│        └───┘           │ ┌────┐│        └───┘           │ ┌────┐│        └───┘           │
        # q_2: ┤ √X ├┤        ────────        ├─┤ √X ├┤        ────────        ├─┤ √X ├┤        ────────        ├─
        #      └────┘└───────          ───────┘ └────┘└───────          ───────┘ └────┘└───────          ───────┘
        boxed_qc = QuantumCircuit(qubits)
        for _ in range(reps):
            boxed_qc.sx(2)
            with boxed_qc.box():
                boxed_qc.x(0)
                boxed_qc.x(1)
                boxed_qc.cz(0, 1)
                boxed_qc.noop(2)

        #      ┌───┐       ┌───┐    ┌───┐
        # q_0: ┤ X ├───■───┤ X ├──■─┤ X ├─■─
        #      ├───┤   │   ├───┤  │ ├───┤ │
        # q_1: ┤ X ├───■───┤ X ├──■─┤ X ├─■─
        #      ├───┴┐┌────┐├───┴┐   └───┘
        # q_2: ┤ √X ├┤ √X ├┤ √X ├───────────
        #      └────┘└────┘└────┘
        qc = QuantumCircuit(qubits)
        for _ in range(reps):
            qc.sx(2)
            qc.x(0)
            qc.x(1)
            qc.cz(0, 1)
            qc.noop(2)

        self.boxed_circ = boxed_qc
        self.circ = qc

    def test_debox_circuit(self):
        deboxed_circ = debox_circuit(self.boxed_circ)
        self.assertEqual(self.circ, deboxed_circ)


class TestRemoveFinalMeasurements(unittest.TestCase):
    """Test for final measurements removal."""

    def setUp(self):
        num_qubits = 5

        #       ┌─────── ┌───┐      ───────┐ ┌─────── ┌───┐      ───────┐ ┌─────── ┌────┐┌─┐             ───────┐
        # q0_0: ┤        ┤ X ├──■──        ├─┤        ┤ X ├─────        ├─┤        ┤ √X ├┤M├────────────        ├─
        #       │        ├───┤┌─┴─┐        │ │        ├───┤             │ │        ├────┤└╥┘┌─┐                 │
        # q0_1: ┤        ┤ X ├┤ X ├        ├─┤        ┤ X ├──■──        ├─┤        ┤ √X ├─╫─┤M├─────────        ├─
        #       │        ├───┤└───┘        │ │        ├───┤┌─┴─┐        │ │        ├────┤ ║ └╥┘┌─┐              │
        # q0_2: ┤ Box-0  ┤ X ├──■──  End-0 ├─┤ Box-0  ┤ X ├┤ X ├  End-0 ├─┤ Box-0  ┤ √X ├─╫──╫─┤M├──────  End-0 ├─
        #       │        ├───┤┌─┴─┐        │ │        ├───┤└───┘        │ │        ├────┤ ║  ║ └╥┘┌─┐           │
        # q0_3: ┤        ┤ X ├┤ X ├        ├─┤        ┤ X ├──■──        ├─┤        ┤ √X ├─╫──╫──╫─┤M├───        ├─
        #       │        ├───┤└───┘        │ │        ├───┤┌─┴─┐        │ │        ├────┤ ║  ║  ║ └╥┘┌─┐        │
        # q0_4: ┤        ┤ X ├─────        ├─┤        ┤ X ├┤ X ├        ├─┤        ┤ √X ├─╫──╫──╫──╫─┤M├        ├─
        #       └─────── └───┘      ───────┘ └─────── └───┘└───┘ ───────┘ └─────── └────┘ ║  ║  ║  ║ └╥┘ ───────┘
        # c0: 5/══════════════════════════════════════════════════════════════════════════╩══╩══╩══╩══╩═══════════
        #                                                                                 0  1  2  3  4
        qr = QuantumRegister(num_qubits)
        cr = ClassicalRegister(num_qubits)
        qc = QuantumCircuit(qr, cr)

        with qc.box():
            qc.x(qr)
            qc.cx(qr[0], qr[1])
            qc.cx(qr[2], qr[3])

        with qc.box():
            qc.x(qr)
            qc.cx(qr[1], qr[2])
            qc.cx(qr[3], qr[4])

        with qc.box():
            qc.sx(qr)
            qc.measure(qr, cr)

        self.qr = qr
        self.cr = cr
        self.boxed_qc = qc

        #       ┌───┐     ┌───┐┌────┐
        # q0_0: ┤ X ├──■──┤ X ├┤ √X ├──────
        #       ├───┤┌─┴─┐├───┤└────┘┌────┐
        # q0_1: ┤ X ├┤ X ├┤ X ├──■───┤ √X ├
        #       ├───┤└───┘├───┤┌─┴─┐ ├────┤
        # q0_2: ┤ X ├──■──┤ X ├┤ X ├─┤ √X ├
        #       ├───┤┌─┴─┐├───┤└───┘ ├────┤
        # q0_3: ┤ X ├┤ X ├┤ X ├──■───┤ √X ├
        #       ├───┤├───┤└───┘┌─┴─┐ ├────┤
        # q0_4: ┤ X ├┤ X ├─────┤ X ├─┤ √X ├
        #       └───┘└───┘     └───┘ └────┘
        # c0: 5/═══════════════════════════
        no_meas_qc = QuantumCircuit(qr, cr)
        no_meas_qc.x(qr)
        no_meas_qc.cx(qr[0], qr[1])
        no_meas_qc.cx(qr[2], qr[3])
        no_meas_qc.x(qr)
        no_meas_qc.cx(qr[1], qr[2])
        no_meas_qc.cx(qr[3], qr[4])
        no_meas_qc.sx(qr)

        self.no_meas_qc = no_meas_qc

    def test_remove_final_measurements(self):
        """Test that final measurements are removed if requested"""
        unboxed_with_no_meas_qc = inject_learned_noise_to_boxed_circuit(
            boxed_circuit=self.boxed_qc,
            refs_to_pauli_lindblad_maps={},
            include_barriers=False,
            remove_final_measurements=True,
        )
        self.assertEqual(unboxed_with_no_meas_qc, self.no_meas_qc)

    def test_not_removed_if_not_final(self):
        """Test that measurements which are not final are not removed"""
        # add a single x gate after the measurements box
        self.boxed_qc.x(self.qr[0])

        # try to remove measurements
        unboxed_with_no_meas_qc = inject_learned_noise_to_boxed_circuit(
            boxed_circuit=self.boxed_qc,
            refs_to_pauli_lindblad_maps={},
            include_barriers=False,
            remove_final_measurements=True,
        )

        # create the expected.
        # circuit stays the same (with measurements and x gate at the end) just unboxed
        self.no_meas_qc.measure(self.qr, self.cr)
        self.no_meas_qc.x(self.qr[0])

        self.assertEqual(unboxed_with_no_meas_qc, self.no_meas_qc)

    def test_do_not_remove_final_measurements(self):
        """Test that final measurements are not removed if not requested"""
        unboxed_with_meas_qc = inject_learned_noise_to_boxed_circuit(
            boxed_circuit=self.boxed_qc,
            refs_to_pauli_lindblad_maps={},
            include_barriers=False,
            remove_final_measurements=False,
        )
        self.no_meas_qc.measure(self.qr, self.cr)
        self.assertEqual(unboxed_with_meas_qc, self.no_meas_qc)

    def test_do_not_remove_unboxed_final_measurements(self):
        """Test that final measurements in a circuit without boxes are not removed in any way"""
        self.no_meas_qc.measure(self.qr, self.cr)
        for flag in [False, True]:
            unboxed_with_meas_qc = inject_learned_noise_to_boxed_circuit(
                boxed_circuit=self.no_meas_qc,
                refs_to_pauli_lindblad_maps={},
                include_barriers=False,
                remove_final_measurements=flag,
            )
            self.assertEqual(unboxed_with_meas_qc, self.no_meas_qc)
