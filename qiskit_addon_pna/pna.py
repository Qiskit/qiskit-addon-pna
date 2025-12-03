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
"""Functions for performing propagated noise absorption (PNA)."""

import multiprocessing as mp
import multiprocessing.sharedctypes
import time
from collections import deque

import numpy as np
from pauli_prop.propagation import (
    RotationGates,
    circuit_to_rotation_gates,
    evolve_through_cliffords,
    propagate_through_operator,
    propagate_through_rotation_gates,
)
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Pauli, PauliLindbladMap, PauliList, SparsePauliOp
from qiskit_aer.noise.errors import PauliLindbladError
from samplomatic.annotations import InjectNoise
from samplomatic.utils import get_annotation, undress_box

circuit_as_rot_gates: RotationGates
obs_ready_for_generator_idx: multiprocessing.sharedctypes.Synchronized
z_shared_np: np.ndarray
x_shared_np: np.ndarray
obs_current_length: multiprocessing.sharedctypes.Synchronized
coeffs_shared_np: np.ndarray


def generate_noise_mitigating_observable(
    noisy_circuit: QuantumCircuit,
    observable: SparsePauliOp | Pauli,
    refs_to_noise_model_map: dict[str, PauliLindbladMap] | None = None,
    *,
    max_err_terms: int,
    max_obs_terms: int,
    search_step: int = 4,
    num_processes: int = 1,
    print_progress: bool = False,
    atol: float = 1e-8,
    batch_size: int = 1,
    inject_noise_before: bool = True,
    mp_start_method: str | None = "spawn",
) -> SparsePauliOp:
    r"""Generate a noise-mitigating observable by propagating it through the inverse of a learned noise channel.

    .. note::

       This function uses the Python ``multiprocessing`` module for parallel execution. This function should be called from within
       an ``if __name__ == "__main__"`` guard to prevent unintended process spawning.

    Starting from the beginning of the circuit, the noise affecting each entangling layer is inverted and each Pauli anti-noise
    generator is then propagated forward through the remainder of the circuit and applied to the observable. The propagation
    routines used to implement this method are available in the `pauli-prop <https://qiskit.github.io/pauli-prop/>`_ package.

    As each anti-noise generator is propagated forward through the circuit under the action of :math:`N` Pauli rotation gates of an
    $M$-qubit circuit, the number of terms will grow as :math:`O(2^N)` towards a maximum of :math:`4^M` unique Pauli components. To control
    the computational cost, terms with small coefficients must be truncated, which will result in some error in the evolved
    anti-noise channel.

    In addition to the truncation of the evolved anti-noise channel, :math:`\Lambda^{-1}`, :math:`\tilde{O}` is also truncated as it is
    propagated through :math:`\Lambda^{-1}`. This is also a source of bias in the final mitigated expectation value.

    While letting :math:`\tilde{O}` grow larger during propagation will increase its accuracy, measuring it requires taking many more
    shots on the QPU. Typically this increases the coefficients of the original Pauli terms in :math:`O`, along with creating many new
    Pauli terms with smaller coefficients. Both the rescaling of the original coefficients and the creation of new terms can
    increase sampling overhead. In practice, we truncate once more by measuring only the largest terms in :math:`\tilde{O}`.

    Args:
        noisy_circuit: A circuit with associated Pauli-Lindblad gate noise.

            If this circuit is boxed, there are expected to be ``InjectNoise`` annotations associated with each box for which gate
            noise should be mitigated. Additionally, the ``refs_to_noise_model_map`` should provide a mapping from the reference
            ID of each ``InjectNoise`` annotation to the associated noise model for that layer, specified as a ``PauliLindbladMap``.

            If this circuit is not boxed, the noise model is expected to be embedded as ``PauliLindbladError`` instructions adjacent
            to each circuit layer for which gate noise should be mitigated. In this case, ``refs_to_noise_model_map`` may be None.
        observable: The observable which will absorb the anti-noise.
        refs_to_noise_model_map: A dictionary mapping noise injection referencs IDs to their corresponding noise models as
            ``PauliLindbladMap``. If ``noisy_circuit`` is not boxed and contains ``PauliLindbladError`` instructions from `qiskit-aer`,
            this mapping is not needed.
        max_err_terms: The maximum number of terms each anti-noise generator may contain as it evolves through the circuit
        max_obs_terms: The maximum number of terms the noise-mitigating observable may contain
        search_step: A parameter that can speed up the approximate application of each error to the observable. The
            relevant subroutine searches a very large 3D space to identify the ``max_obs_terms`` largest terms in a product.
            Setting this step size >1 accelerates that search by a factor of ``search_step**3``, at a potential cost
            in accuracy. This inaccuracy is expected to be small for ``search_step**3 << max_obs_terms``.
        num_processes: The number of processes for parallelization. These may be used for forward evolution of generators,
            and for applying evolved generators to the observable. If ``batch_size`` is ``1`` (default), all are used for evolving
            generators. Otherwise, ``max(min(batch_size, num_processes // 2), 1)`` of these will be allocated for applying evolved
            generators to the observable.
        print_progress: Whether to print progress to stdout
        atol: Terms below this threshold will not be added to operators as they evolve
        batch_size: Setting this to a value > 1 allows batches of noise generators to be applied to the observable in parallel.
            This coarse-grain application of anti-noise to the observable comes at a loss of accuracy related to the probability
            that more than one error in the batch occurs when the circuit is run. This should usually not be set higher than
            ``max(1, num_processes // 2)``.
        inject_noise_before: If ``True``, the Pauli Lindblad noise instruction will be inserted before its
            corresponding 2q-gate layer. Otherwise, it will be inserted after it, defaults to ``True``.
        mp_start_method: The method to use when starting new parallel processes. Valid values are ``fork``, ``spawn``,
            ``forkserver``, and ``None``. If ``None``, the default method will be used.

    Returns:
        The noise-mitigating observable

    Raises:
        ValueError: The circuit and observable have mismatching sizes
        ValueError: num_processes and batch_size must be >= 1
        ValueError: ``max_obs_terms`` should be larger than the length of ``observable``
        ValueError: Incompatible noisy circuit and refs_to_noise_model_map
        ValueError: The observable must only contain real-valued coefficients
    """
    if observable.num_qubits != noisy_circuit.num_qubits:
        raise ValueError(f"{observable.num_qubits = } does not match {noisy_circuit.num_qubits = }")
    if batch_size < 1:
        raise ValueError("batch_size must be integer greater than or equal to 1.")
    # Default num_processes is all cores minus one
    if num_processes < 1:
        raise ValueError("num_processes must be integer greater than or equal to 1.")
    if max_obs_terms < len(observable):
        raise ValueError("max_obs_terms must be larger than the length of observable.")

    observable = SparsePauliOp(observable)
    original_obs_length = len(observable)

    z = observable.paulis.z
    ctype = np.ctypeslib.as_ctypes_type(z.dtype)
    z_max_shape = (max_obs_terms, observable.num_qubits)
    z_shared = mp.RawArray(ctype, int(np.prod(z_max_shape)))
    z_shared_np: np.ndarray = np.ndarray(z_max_shape, dtype=z.dtype, buffer=z_shared)
    np.copyto(z_shared_np[: z.shape[0], : z.shape[1]], z)

    x = observable.paulis.x
    ctype = np.ctypeslib.as_ctypes_type(x.dtype)
    x_max_shape = (max_obs_terms, observable.num_qubits)
    x_shared = mp.RawArray(ctype, int(np.prod(x_max_shape)))
    x_shared_np: np.ndarray = np.ndarray(x_max_shape, dtype=x.dtype, buffer=x_shared)
    np.copyto(x_shared_np[: x.shape[0], : x.shape[1]], x)

    if not np.allclose(observable.coeffs.imag, 0):
        raise ValueError("Coeffs must be real.")
    coeffs = observable.coeffs.real
    ctype = np.ctypeslib.as_ctypes_type(coeffs.dtype)
    coeffs_max_shape = (max_obs_terms,)
    coeffs_shared = mp.RawArray(ctype, int(np.prod(coeffs_max_shape)))
    coeffs_shared_np: np.ndarray = np.ndarray(
        coeffs_max_shape, dtype=coeffs.dtype, buffer=coeffs_shared
    )
    np.copyto(coeffs_shared_np[: coeffs.shape[0]], coeffs)

    obs_current_length = mp.RawValue("i", original_obs_length)
    obs_ready_for_generator_idx = mp.RawValue("i", batch_size)

    # Strip boxes and inject Aer PauliLindbladError instructions
    noisy_circuit = _inject_learned_noise_to_boxed_circuit(
        noisy_circuit,
        refs_to_noise_model_map,
        include_barriers=False,
        remove_final_measurements=True,
        inject_noise_before=inject_noise_before,
    )

    # Evolve any known Clifford gates to the front of the circuit
    _, noisy_circuit = evolve_through_cliffords(noisy_circuit)

    noiseless_circuit = QuantumCircuit.from_instructions(
        [circ_inst for circ_inst in noisy_circuit if circ_inst.name != "quantum_channel"],
        qubits=noisy_circuit.qubits,
        clbits=noisy_circuit.clbits,
    )
    circuit_as_rot_gates = circuit_to_rotation_gates(noiseless_circuit)

    generator_jobs: deque = deque()

    # evolve all antinoise channels forwards:
    channels = [inst for inst in noisy_circuit if inst.name == "quantum_channel"]
    num_generators = sum([len(channel.operation._quantum_error.generators) for channel in channels])
    latest_generator_job = None
    num_unfinished_this_batch = batch_size
    num_unfinished_total = num_generators
    new_terms_this_batch = []
    num_started = 0
    num_consumed = 0
    last_update = 0.0
    global_scale_factor = 1.0
    gen_gen = _generator_generator(noisy_circuit)

    ctx = mp.get_context(mp_start_method)
    with ctx.Pool(
        processes=num_processes,
        initializer=_initialize_pool,
        initargs=(
            # Dynamic:
            z_shared,
            x_shared,
            coeffs_shared,
            obs_current_length,
            obs_ready_for_generator_idx,
            # Static:
            circuit_as_rot_gates,
            original_obs_length,
            max_err_terms,
            max_obs_terms,
            atol,
            observable.num_qubits,
        ),
    ) as pool:
        while True:
            if print_progress and (time.time() - last_update > 0.1):  # pragma: no cover
                print(
                    f"\r{num_consumed} / {num_generators} generators propagated",
                    end="",
                    flush=True,
                )
                last_update = time.time()

            # Approach: Avoid backlogs by prioritizing more expensive calculations
            # High priority: Propagate observable through any evolved antinoise in the queue.
            if latest_generator_job is None and len(generator_jobs) > 0:
                latest_generator_job = generator_jobs.popleft()
            if (latest_generator_job is not None) and latest_generator_job.ready():
                new_terms_this_batch.append(latest_generator_job.get())
                latest_generator_job = None
                num_consumed += 1
                num_unfinished_this_batch -= 1
                num_unfinished_total -= 1
                # if entire batch is done, update the observable:
                if num_unfinished_this_batch == 0 or num_unfinished_total == 0:
                    observable += SparsePauliOp.sum(new_terms_this_batch)
                    observable = observable.simplify(atol=0)
                    observable = _keep_k_largest(
                        observable, max_obs_terms, ignore_pauli_phase=True, copy=False
                    )[0]

                    z = observable.paulis.z
                    np.copyto(z_shared_np[: z.shape[0], : z.shape[1]], z)
                    x = observable.paulis.x
                    np.copyto(x_shared_np[: x.shape[0], : x.shape[1]], x)
                    coeffs = observable.coeffs.real
                    np.copyto(coeffs_shared_np[: coeffs.shape[0]], coeffs)

                    new_terms_this_batch = []
                    num_unfinished_this_batch = batch_size
                    obs_current_length.value = len(observable)
                    obs_ready_for_generator_idx.value = (
                        obs_ready_for_generator_idx.value + batch_size
                    )

                if num_unfinished_total == 0:
                    break
                continue

            # Low priority: Forward-evolve any remaining generators through circuit.
            if num_started - num_consumed < 2 * num_processes:
                next_gen = next(gen_gen, None)
                if next_gen is not None:
                    generator_pauli, quasiprob, generator_idx, gate_idx = next_gen
                    err_mag_squared = np.abs(quasiprob / (1 - quasiprob))
                    global_scale_factor *= 1 - quasiprob
                    generator = SparsePauliOp([generator_pauli], [np.sqrt(err_mag_squared)])
                    generator_jobs.append(
                        pool.apply_async(
                            _evolve_and_apply_generator,
                            args=(
                                generator,
                                generator_idx,
                                gate_idx,
                                quasiprob,
                                max_err_terms,
                                max_obs_terms,
                                search_step,
                                atol,
                                original_obs_length,
                            ),
                        )
                    )
                    num_started += 1
                    continue

            # If nothing to do, sleep before checking again
            time.sleep(0.001)

    if print_progress:  # pragma: no cover
        msg = f"\rFinished! {num_generators} / {num_generators} generators propagated."
        print(f"\r{msg:<70}", end="", flush=True)
    observable *= global_scale_factor

    return observable


def _initialize_pool(
    z_shared,
    x_shared,
    coeffs_shared,
    _obs_current_length,
    _obs_ready_for_generator_idx,
    _circuit_as_rot_gates,
    _original_obs_length,
    _max_err_terms,
    _max_obs_terms,
    _atol,
    _num_qubits,
):
    # Dynamic (multiprocessing objects that parent process will update):
    global \
        z_shared_np, \
        x_shared_np, \
        coeffs_shared_np, \
        obs_current_length, \
        obs_ready_for_generator_idx
    z_shared_np = np.ndarray((_max_obs_terms, _num_qubits), dtype=bool, buffer=z_shared)
    x_shared_np = np.ndarray((_max_obs_terms, _num_qubits), dtype=bool, buffer=x_shared)
    coeffs_shared_np = np.ndarray((_max_obs_terms,), dtype=float, buffer=coeffs_shared)
    obs_current_length = _obs_current_length
    obs_ready_for_generator_idx = _obs_ready_for_generator_idx

    # Static (never updated):
    global circuit_as_rot_gates, original_obs_length, max_err_terms, max_obs_terms, atol, num_qubits
    circuit_as_rot_gates = _circuit_as_rot_gates
    original_obs_length = _original_obs_length
    max_err_terms = _max_err_terms
    max_obs_terms = _max_obs_terms
    atol = _atol
    num_qubits = _num_qubits


def _generator_generator(noisy_circuit):
    # start with earliest channels:
    gate_idx = 0
    generator_idx = 0
    for circ_inst in noisy_circuit:
        if circ_inst.name == "quantum_channel":
            err = circ_inst.operation._quantum_error
            err = err.inverse()
            for generator, quasiprob in zip(
                err.generators, (1 - np.exp(-2 * err.rates)) / 2, strict=True
            ):
                yield generator, quasiprob, generator_idx, gate_idx
                generator_idx += 1
        elif circ_inst.name != "barrier":
            gate_idx += 1


def _evolve_and_apply_generator(
    generator: SparsePauliOp,
    generator_idx: int,
    gate_idx: int,
    quasiprob: complex,
    max_error_terms: int,
    max_obs_terms: int,
    search_step: int,
    atol: float,
    original_obs_length: int,
) -> SparsePauliOp:
    """Forward-propagate a generator through a circuit and normalize after truncation if requested."""
    if quasiprob == 0:
        num_qb_in_obs = z_shared_np.shape[1]
        return SparsePauliOp("I" * num_qb_in_obs, [0])

    rot_gates = RotationGates(
        circuit_as_rot_gates.gates[gate_idx:],
        circuit_as_rot_gates.qargs[gate_idx:],
        circuit_as_rot_gates.thetas[gate_idx:],
    )
    evolved, _ = propagate_through_rotation_gates(
        generator, rot_gates, max_terms=max_error_terms, atol=atol, frame="s"
    )

    norm_reduction = float(np.linalg.norm(evolved.coeffs) / np.linalg.norm(generator.coeffs))
    if norm_reduction > 1 + max(atol, float(np.finfo(np.float64).resolution)):
        norm_reduction = 1.0

    while True:
        if generator_idx < obs_ready_for_generator_idx.value:
            paulis = PauliList.from_symplectic(
                z_shared_np[: obs_current_length.value],
                x_shared_np[: obs_current_length.value],
            )
            observable = SparsePauliOp(
                paulis,
                np.sign(quasiprob) * coeffs_shared_np[: obs_current_length.value],
                ignore_pauli_phase=True,
                copy=False,
            )
            new_terms = propagate_through_operator(
                op1=observable,
                op2=evolved,
                max_terms=max_obs_terms,
                coerce_op1_traceless=True,
                num_leading_terms=original_obs_length,
                frame="h",
                atol=atol,
                search_step=search_step,
            )
            break
        time.sleep(0.001)

    return new_terms


def _inject_learned_noise_to_boxed_circuit(
    boxed_circuit: QuantumCircuit,
    refs_to_pauli_lindblad_maps: dict[str, PauliLindbladMap] | None,
    include_barriers: bool = False,
    remove_final_measurements: bool = True,
    inject_noise_before: bool = True,
) -> QuantumCircuit:
    """Generate an unboxed circuit with the noise injected as ``PauliLindbladError`` instructions.

    Args:
        boxed_circuit: A `QuantumCircuit` with boxes and `InjectNoise` annotations for 2-qubit layers.
        refs_to_pauli_lindblad_maps: A dictionary mapping `InjectNoise.ref` to corresponding `PauliLindbladMap`.
        include_barriers: A boolean to decide whether or not to insert barriers around `LayerError` instructions.
        remove_final_measurements: If `True` remove any boxed final measure instructions from the circuit.
        inject_noise_before: If `True`, the Pauli Lindblad noise instruction will be inserted before its
         corresponding 2q-gate layer. Otherwise, it will be inserted after it, defaults to `True`.

    Returns:
        A `QuantumCircuit` without boxes and with `PauliLindbladError` instructions inserted according to the given mapping.
    """
    unboxed_noisy_circuit = QuantumCircuit.copy_empty_like(boxed_circuit)
    last_instruction_idx = len(boxed_circuit.data) - 1
    for idx, inst in enumerate(boxed_circuit.data):
        if inst.name == "box":
            box = inst.operation

            # Collect the circuit's qargs which are used in the instruction.
            # Needed for mapping the box instruction to the correct qubits
            # in the new unboxed circuit.
            qargs = [q for q in unboxed_noisy_circuit.qubits if q in inst.qubits]

            injected_noise = get_annotation(box, InjectNoise)
            if injected_noise is not None:
                if refs_to_pauli_lindblad_maps is None:
                    raise ValueError(
                        "The circuit contains a noisy box, but refs_to_pauli_lindeblad_maps is None."
                    )
                if injected_noise.ref not in refs_to_pauli_lindblad_maps:
                    raise ValueError(
                        f"ref: {injected_noise.ref} is missing from Pauli Lindblad Map."
                    )
                pauli_lindblad_map = refs_to_pauli_lindblad_maps[injected_noise.ref]

                if include_barriers:
                    unboxed_noisy_circuit.barrier()

                # A noise model exists for that instruction, creating a Pauli Lindblad Error instruction.
                noise_instruction = _pauli_lindblad_map_to_layer_error(pauli_lindblad_map)

                if include_barriers:
                    unboxed_noisy_circuit.barrier()

                # The undressed box is needed in order to know where to inject the noise.
                undressed_box = undress_box(box)

                # The noise needs to be injected in proximity to the 2q-gate corresponding instructions.
                # If the original box is 'left-dressed', start by adding the 1q-gate instructions.
                # Then, handle noise injection and 2q-gates (order dependent on `place_noise_before`).
                # If the box is `right-dressed`, first handle noise injections and 2q-gates (order
                # dependent on `place_noise_before`), then add the 1q-gate instructions.
                if box.body.data[0].operation.num_qubits == 1:
                    # First instruction is a 1q-gate => box is left dressed.
                    # Add the 1q-gates first.
                    for internal_instruction in box.body:
                        if internal_instruction not in undressed_box.body:
                            unboxed_noisy_circuit.append(
                                instruction=internal_instruction,
                                qargs=qargs,
                            )
                    # Inject noise (before)
                    if inject_noise_before:
                        unboxed_noisy_circuit.append(noise_instruction, qargs=qargs)

                    # Add the 2q-gates
                    for internal_instruction in box.body:
                        if internal_instruction in undressed_box.body:
                            unboxed_noisy_circuit.append(
                                instruction=internal_instruction,
                                qargs=qargs,
                            )
                    # Inject noise (after)
                    if not inject_noise_before:
                        unboxed_noisy_circuit.append(noise_instruction, qargs=qargs)
                else:
                    # First instruction is NOT a 1q-gate => box is right dressed.
                    # Inject noise (before)
                    if inject_noise_before:
                        unboxed_noisy_circuit.append(
                            noise_instruction,
                            qargs=qargs,
                        )
                        # Add rest of 2q-gate and 1q-gate instructions in order
                        for internal_instruction in box.body:
                            unboxed_noisy_circuit.append(
                                instruction=internal_instruction,
                                qargs=qargs,
                            )
                    # Inject noise (after)
                    if not inject_noise_before:
                        # Add the 2q-gate instructions in order
                        for internal_instruction in box.body:
                            if internal_instruction in undressed_box.body:
                                unboxed_noisy_circuit.append(
                                    instruction=internal_instruction,
                                    qargs=qargs,
                                )

                        # Inject noise
                        unboxed_noisy_circuit.append(
                            noise_instruction,
                            qargs=qargs,
                        )

                        # Add rest of 1q-gate instructions in order
                        for internal_instruction in box.body:
                            if internal_instruction not in undressed_box.body:
                                unboxed_noisy_circuit.append(
                                    instruction=internal_instruction,
                                    qargs=qargs,
                                )

            # Add the boxed instructions as is (not injecting any noise).
            # We assume that measurements do not have InjectNoise annotation.
            else:
                # this is to close with a barrier on the previous box
                if include_barriers:
                    unboxed_noisy_circuit.barrier()

                # If requested by the user, remove any measurements from the last box.
                if remove_final_measurements and idx == last_instruction_idx:
                    # Remove measure instructions INDEPENDENTLY of
                    # their order within the box.
                    for internal_instruction in box.body:
                        if internal_instruction.name == "measure":
                            continue
                        else:
                            unboxed_noisy_circuit.append(
                                internal_instruction,
                                qargs=qargs,
                            )
                # Add instructions in order.
                else:
                    for internal_instruction in box.body:
                        unboxed_noisy_circuit.append(
                            instruction=internal_instruction,
                            qargs=qargs,
                        )

        # Add the instruction as is (it does not have a box),
        # mapping qargs is not needed in that case.
        else:
            unboxed_noisy_circuit.append(instruction=inst)

    return unboxed_noisy_circuit


def _pauli_lindblad_map_to_layer_error(pauli_lindblad_map: PauliLindbladMap) -> PauliLindbladError:
    """Creates a PauliLindbladError instruction from a PauliLindbladMap.

    Args:
        pauli_lindblad_map: A PauliLindbladMap.

    Returns:
        A PauliLindbladError circuit instruction
    """
    sparse_list = pauli_lindblad_map.to_sparse_list()
    spare_pauli_op = SparsePauliOp.from_sparse_list(sparse_list, pauli_lindblad_map.num_qubits)
    noise_instruction = PauliLindbladError(spare_pauli_op.paulis, spare_pauli_op.coeffs)
    return noise_instruction


def _keep_k_largest(
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
