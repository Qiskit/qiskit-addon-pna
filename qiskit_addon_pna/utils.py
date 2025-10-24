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

# Reminder: update the RST file in docs/apidocs when adding new interfaces.
"""Utilities for representing Pauli-Lindblad noise in quantum circuits."""

import numpy as np
from qiskit.circuit import BoxOp, CircuitInstruction, QuantumCircuit
from qiskit.quantum_info import PauliLindbladMap, SparsePauliOp
from qiskit_aer.noise.errors import PauliLindbladError as PLEAer
from samplomatic.annotations import InjectNoise, Twirl
from samplomatic.utils import get_annotation, undress_box


def inject_learned_noise_to_boxed_circuit(
    boxed_circuit: QuantumCircuit,
    refs_to_pauli_lindblad_maps: dict[str, tuple[PauliLindbladMap, CircuitInstruction]],
    include_barriers: bool = False,
    remove_final_measurements: bool = True,
) -> QuantumCircuit:
    """Generate an unboxed circuit with the noise injected as ``PauliLindbladError`` instructions.

    Args:
        boxed_circuit: A `QuantumCircuit` with boxes and `InjectNoise` annotations for 2-qubit layers.
        refs_to_pauli_lindblad_maps: A dictionary mapping `InjectNoise.ref` to corresponding `PauliLindbladMap` and `CircuitInstruction`.
        include_barriers: A boolean to decide whether or not to insert barriers around `LayerError` instructions.
        remove_final_measurements: If `True` remove any boxed final measure instructions from the circuit.

    Returns:
        A `QuantumCircuit` without boxes and with `PauliLindbladError` instructions inserted according to the given mapping.
    """
    unboxed_noisy_circuit = QuantumCircuit.copy_empty_like(boxed_circuit)
    last_instruction_idx = len(boxed_circuit.data) - 1
    for idx, inst in enumerate(boxed_circuit.data):
        if inst.name == "box":
            box = inst.operation

            # getting the qargs of the circuit instruction in an ordered fashion,
            # so they can be used for mapping the box instruction to the correct qubits
            # in the new unboxed circuit
            ordered_instruction_qargs = [
                q for q in unboxed_noisy_circuit.qubits if q in inst.qubits
            ]

            injected_noise = get_annotation(box, InjectNoise)
            if injected_noise is not None:
                assert injected_noise.ref in refs_to_pauli_lindblad_maps
                pauli_lindblad_map, _ = refs_to_pauli_lindblad_maps[injected_noise.ref]

                if include_barriers:
                    unboxed_noisy_circuit.barrier()

                # A noise model exists for that layer --> inject it before the 2q gates in the layer
                # Creating a PLE instruction
                noise_instruction = pauli_lindblad_map_to_layer_error(pauli_lindblad_map)

                if include_barriers:
                    unboxed_noisy_circuit.barrier()

                # the undressed box is needed in order to know where to inject the noise
                undressed_box = undress_box(box)

                # The noise needs to be injected right before the 2q-gate corresponding
                # instructions. Therefore, if the original box is starting with a 1q-gate instruction
                # it means the box was 'left-dressed' and we should start be adding the 1q-gate instructions
                # before injecting the noise. Essentially ending up with the following instructions order:
                #   1. the box's left dressing 1q-gate instructions
                #   2. the injected noise instruction
                #   3. the box's 2q-gate instructions
                # If the box is staring with a 2q-gate instruction it means it was originally `right-dressed`
                # so we should start with injecting the noise and then add the rest instruction. Essentially
                # ending up with the following instructions order:
                #   1. the injected noise instruction
                #   2. the box's 2q-gate instructions
                #   3. the box's right dressing 1q-gate instructions
                if box.body.data[0].operation.num_qubits == 1:
                    # add the 1q-gates that were removed from the undressed box first
                    for internal_instruction in box.body:
                        if internal_instruction not in undressed_box.body:
                            unboxed_noisy_circuit.append(
                                instruction=internal_instruction,
                                qargs=ordered_instruction_qargs,
                            )

                    # inject the noise second
                    unboxed_noisy_circuit.append(
                        noise_instruction,
                        qargs=ordered_instruction_qargs,
                    )
                    # add the 2q-gates
                    for internal_instruction in box.body:
                        if internal_instruction in undressed_box.body:
                            unboxed_noisy_circuit.append(
                                instruction=internal_instruction,
                                qargs=ordered_instruction_qargs,
                            )
                else:
                    # inject the noise first (because there are no 1q-gates to add before)
                    unboxed_noisy_circuit.append(
                        noise_instruction,
                        qargs=ordered_instruction_qargs,
                    )
                    # add the 2q-gate and 1q-gate instructions in order
                    for internal_instruction in box.body:
                        unboxed_noisy_circuit.append(
                            instruction=internal_instruction,
                            qargs=ordered_instruction_qargs,
                        )

            # add the inner box instructions as is (the box does not have relevant InjectNoise annotation)
            # we assume that measurements does not have the InjectNoise annotation.
            else:
                # this is to close with a barrier on the previous box
                if include_barriers:
                    unboxed_noisy_circuit.barrier()

                # removing any measurements if it is the last box in the circuit
                if remove_final_measurements and idx == last_instruction_idx:
                    # removing measures
                    for internal_instruction in box.body:
                        if internal_instruction.name == "measure":
                            continue
                        else:
                            unboxed_noisy_circuit.append(
                                internal_instruction,
                                qargs=ordered_instruction_qargs,
                            )
                # add all other instructions in order
                else:
                    for internal_instruction in box.body:
                        unboxed_noisy_circuit.append(
                            instruction=internal_instruction,
                            qargs=ordered_instruction_qargs,
                        )

        # add the original instruction as is (it does not have a box)
        else:
            unboxed_noisy_circuit.append(instruction=inst)

    return unboxed_noisy_circuit


def _unbox_twirl_box(
    circuit: QuantumCircuit, box: BoxOp, undressed_box: BoxOp, include_barriers: bool
):
    """Adds to the circuit all the instructions that are part of the undressed part of the box, sandwiched with barriers.

    Args:
        circuit: The circuit to append instruction to.
        box: The box to copy instructions from.
        undressed_box: A box to compare instructions with.
        include_barriers: If True, include named barriers around the unboxed copied instructions.

    """
    empty_box = undressed_box.body.data == []
    if include_barriers and not empty_box:
        circuit.barrier(label="twirled_layer")
    for box_inst in box.body:
        if box_inst in undressed_box.body:
            circuit.append(box_inst)
    if include_barriers and not empty_box:
        circuit.barrier(label="twirled_layer")


def _unbox_twirl_dressing(
    circuit: QuantumCircuit, box: BoxOp, undressed_box: BoxOp
) -> QuantumCircuit:
    """Append all instructions in ``box`` to ``circuit`` if they are not in ``undressed_box``.

    Args:
        circuit: The circuit to append instruction to.
        box: The box to copy instructions from.
        undressed_box: A box containing the undressed part of the circuit.
    """
    for box_inst in box.body:
        if box_inst not in undressed_box.body:
            circuit.append(box_inst)


def _unbox_box(
    circuit: QuantumCircuit,
    box: BoxOp,
    only_2q_gates: bool,
    include_barriers: bool,
) -> QuantumCircuit:
    """Pull the contents out of a box into a new circuit containing no boxes.

    Args:
        circuit: A QuantumCircuit to store the unboxed instructions in.
        box: The box to unbox.
        only_2q_gates: Whether to add the barriers only before and after the 2-qubit
            gates in the box, or before and after the whole box.
        include_barriers: If True, include named barriers around twirled boxes.

    Returns:
        A QuantumCircuit with the instructions of the given box instead of the box.
    """
    if twirl_annotation := get_annotation(box, Twirl):
        if only_2q_gates:
            undressed_box = undress_box(box)
            if twirl_annotation.dressing == "left":
                _unbox_twirl_dressing(circuit, box, undressed_box)
                _unbox_twirl_box(circuit, box, undressed_box, include_barriers)
            else:
                _unbox_twirl_box(circuit, box, undressed_box, include_barriers)
                _unbox_twirl_dressing(circuit, box, undressed_box)
        else:
            # Dressed and undressed box are the same
            _unbox_twirl_box(circuit, box, box, include_barriers)
    else:
        # Dressed and undressed box are the same and no barriers should be added if there isn't a Twirl annotation
        _unbox_twirl_box(circuit, box, box, include_barriers=False)
    return circuit


def debox_circuit(
    boxed_circuit: QuantumCircuit,
    include_named_barriers: bool = False,
    only_2q_gates: bool = True,
) -> QuantumCircuit:
    """A method that return a striped-down box-wise version of a given `QuantumCircuit`.

    Args:
        boxed_circuit: A `QuantumCircuit` with boxes.
        include_named_barriers: Whether to add named barriers before and after 2 qubit layers with twirl annotation
        only_2q_gates: whether to add the barriers only before and after the 2-qubit gates in each box,
                    or before and after each entire box.

    Returns:
        A functionally identical `QuantumCircuit` without boxes.
    """
    deboxed_circuit = QuantumCircuit.copy_empty_like(boxed_circuit)
    for inst in boxed_circuit.data:
        if inst.name == "box":
            deboxed_circuit = _unbox_box(
                deboxed_circuit,
                inst.operation,
                only_2q_gates=only_2q_gates,
                include_barriers=include_named_barriers,
            )

        # add the original instruction
        else:
            deboxed_circuit.append(instruction=inst)
    return deboxed_circuit


def pauli_lindblad_map_to_layer_error(pauli_lindblad_map: PauliLindbladMap) -> PLEAer:
    """Creates a PauliLindbladError instruction from a PauliLindbladMap.

    Args:
        pauli_lindblad_map: A PauliLindbladMap.

    Returns:
        A PauliLindbladError circuit instruction
    """
    sparse_list = pauli_lindblad_map.to_sparse_list()
    spare_pauli_op = SparsePauliOp.from_sparse_list(sparse_list, pauli_lindblad_map.num_qubits)
    noise_instruction = PLEAer(spare_pauli_op.paulis, spare_pauli_op.coeffs)
    return noise_instruction


def keep_k_largest(
    operator: SparsePauliOp,
    k: int | None = None,
    normalize: bool = False,
    ignore_pauli_phase=False,
    copy=True,
) -> tuple[SparsePauliOp, float]:
    """Keep the ``k`` terms in ``operator`` that have the largest coefficient magnitude.

    Args:
        operator: The Sparse Pauli Operator to truncate.
        k: The number of terms to keep after truncation.
        normalize: Should the operator's coefficients be normalized, defaults to False.
        ignore_pauli_phase: Ignoring the operator's Pauli phase, defaults to False.
        copy: Copy the data if possible, defaults to True.

    Returns:
        A tuple of the truncated `SparsePauliOp` and its norm.
    """
    init_onenorm = np.abs(operator.coeffs).sum()

    if k == 0:
        return 0 * operator, init_onenorm

    if k is not None and len(operator) > k:
        ordering = np.argpartition(np.abs(operator.coeffs), kth=-k)[-k:]
    else:
        ordering = np.arange(len(operator))

    kept = SparsePauliOp(
        operator.paulis[ordering],
        operator.coeffs[ordering],
        ignore_pauli_phase=ignore_pauli_phase,
        copy=copy,
    )

    if normalize:
        kept.coeffs *= np.linalg.norm(operator.coeffs) / np.linalg.norm(kept.coeffs)

    trunc_onenorm = init_onenorm - np.abs(kept.coeffs).sum()

    return kept, trunc_onenorm
