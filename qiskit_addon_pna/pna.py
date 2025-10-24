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
import warnings
from collections import deque

import numpy as np
from pauli_prop.propagation import (
    RotationGates,
    circuit_to_rotation_gates,
    evolve_through_cliffords,
    propagate_through_operator,
    propagate_through_rotation_gates,
)
from qiskit.circuit import CircuitInstruction, QuantumCircuit
from qiskit.quantum_info import Pauli, PauliLindbladMap, PauliList, SparsePauliOp
from qiskit_aer.noise.errors import PauliLindbladError

from .utils import inject_learned_noise_to_boxed_circuit, keep_k_largest

circuit_as_rot_gates: RotationGates
obs_ready_for_generator_idx: multiprocessing.sharedctypes.Synchronized
z_shared_np: np.ndarray
x_shared_np: np.ndarray
obs_current_length: multiprocessing.sharedctypes.Synchronized
coeffs_shared_np: np.ndarray


def generate_noise_mitigated_observable(
    boxed_circuit: QuantumCircuit,
    refs_to_noise_model_map: dict[str, tuple[PauliLindbladMap, CircuitInstruction]],
    observable: SparsePauliOp | Pauli,
    max_err_terms: int = 1000,
    max_obs_terms: int = 1000,
    search_step: int = 4,
    num_processes: int = 1,
    print_progress: bool = False,
    atol: float = 1e-8,
    batch_size: int = 1,
) -> SparsePauliOp:
    """Generate a noise-mitigated observable using Pauli evolution.

    Args:
        boxed_circuit: Boxed circuit containing InjectNoise annotations.
        refs_to_noise_model_map: A dictionary mapping noise injection refs to their corresponding noise models as ``PauliLindbladMap``.
        observable: The observable which will absorb the antinoise.
        max_err_terms: The maximum number of terms each noise generator may contain as it evolves through the circuit
        max_obs_terms: The maximum number of terms the noise-mitigated observable may contain
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

    Returns:
        The noise-mitigated observable

    Raises:
        ValueError: The circuit and observable have mismatching sizes
        ValueError: num_processes and batch_size must be >= 1
        ValueError: ``max_obs_terms`` should be larger than the length of ``observable``
    """
    if observable.num_qubits != boxed_circuit.num_qubits:
        raise ValueError(f"{observable.num_qubits = } does not match {boxed_circuit.num_qubits = }")
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
        raise NotImplementedError("Coeffs must be real.")
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
    noisy_circuit = inject_learned_noise_to_boxed_circuit(
        boxed_circuit,
        refs_to_noise_model_map,
        include_barriers=False,
        remove_final_measurements=True,
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

    with mp.Pool(
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
            if print_progress and (time.time() - last_update > 0.1):
                print(
                    f"{num_consumed} / {num_generators}",
                    end="\r",
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
                    observable = keep_k_largest(
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
            if not isinstance(err, PauliLindbladError):
                raise TypeError(
                    f"Expected PauliLindbladError in noisy_circuit but found {type(err)}"
                )
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
        warnings.warn(
            f"{norm_reduction = } should be <= 1. This can result from truncating before merging duplicate Pauli terms.",
            stacklevel=1,
        )
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
