# Propagated noise absorption (PNA)

PNA is a technique for mitigating errors in observable expectation values by propagating
the observable through the inverse of the learned noise channel. This results in a new observable
that, when measured against the noisy state, mitigates the learned noise.

### Overview
Executing entangling gates on modern QPUs results in a substantial amount of noise. Until fault
tolerant devices are available, ideal entangling gates, $\mathcal{U}$, will not be available.
They will instead be affected by some noise channel, $\Lambda$.

![Noisy experiment](docs/images/noisy_expt.png)

It is possible to learn and efficiently characterize this gate noise as a Pauli-Lindblad model, and
as shown in probabilistic error cancellation (PEC), we can mitigate the error by implementing the
anti-noise, $\Lambda^{-1}$, with a QPU sampling protocol [cite pec]. Other techniques, such as
tensor-network error mitigation (TEM), implement the inverse noise channel as a classical
post-processing step [cite TEM].

![Noise-mitigated picture](docs/images/noise_mitigated_expt.png)

Like TEM, PNA implements the inverse noise channel in a classical processing step. While TEM uses
tensor networks to describe and apply the noise-mitigating map to a set of informationally complete
measurements, PNA uses Pauli propagation to propagate the observable, $O$, through the inverse noise
channel. This results in a new observable, $\tilde{O}$, that when measured against the noisy state, mitigates the
learned noise.

![PNA picture](docs/images/pna_overview.png)

##### Sources of bias

1. This implementation propagates each Pauli error generator within each anti-noise channel, $\Lambda^{-1}_i$,
to the end of the circuit. As each anti-noise generator is propagated forward through the circuit
under the action of $N$ Pauli rotation gates of an $M$-qubit circuit, the number of terms will grow
as $O(2^N)$ towards a maximum of $4^M$ unique Pauli components. To control the computational cost,
terms with small coefficients must be truncated, which results in some error in the evolved
anti-noise channel.

2. In addition to the truncation of the evolved anti-noise channel, $\Lambda^{-1}$, $\tilde{O}$ must also
be truncated as it is propagated through $\Lambda^{-1}$. Of course, this is also a source of
bias in the final mitigated expectation value.

3. While letting $\tilde{O}$ grow larger during propagation will increase its accuracy, measuring it
requires taking many more shots on the QPU. It is often practical and sufficient to only measure the
largest terms in $\tilde{O}$; however, one does not generally know the optimal number of terms to
measure for a given $\tilde{O}$ calculation.

----------------------------------------------------------------------------------------------------

### Documentation

All documentation is available at https://qiskit.github.io/qiskit-addon-pna/.

----------------------------------------------------------------------------------------------------

### Installation

We encourage installing this package via `pip`, when possible:

```bash
pip install 'qiskit-addon-pna'
```

For more installation information refer to these [installation instructions](docs/install.rst).

----------------------------------------------------------------------------------------------------

### Deprecation Policy

We follow [semantic versioning](https://semver.org/) and are guided by the principles in
[Qiskit's deprecation policy](https://github.com/Qiskit/qiskit/blob/main/DEPRECATION.md).
We may occasionally make breaking changes in order to improve the user experience.
When possible, we will keep old interfaces and mark them as deprecated, as long as they can co-exist with the
new ones.
Each substantial improvement, breaking change, or deprecation will be documented in the
[release notes](https://qiskit.github.io/qiskit-addon-pna/release-notes.html).

----------------------------------------------------------------------------------------------------

### Contributing

The source code is available [on GitHub](https://github.com/Qiskit/qiskit-addon-pna).

The developer guide is located at [CONTRIBUTING.md](https://github.com/Qiskit/qiskit-addon-pna/blob/main/CONTRIBUTING.md)
in the root of this project's repository.
By participating, you are expected to uphold Qiskit's [code of conduct](https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md).

----------------------------------------------------------------------------------------------------

### License

[Apache License 2.0](LICENSE.txt)

----------------------------------------------------------------------------------------------------

### References
