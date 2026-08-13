# array-api-rng

Random number generation for the [Array API standard](https://data-apis.org/array-api/latest/).

## Motivation

The Array API standard has no concept of random number generation, which
makes it hard to write array-library-agnostic code that needs randomness
(e.g. statistical libraries, probabilistic models, Monte Carlo methods,
machine learning). This project grew out of
[data-apis/array-api-extra#308](https://github.com/data-apis/array-api-extra/issues/308),
where it became clear that RNG support didn't fit the scope of
`array-api-extra` (which avoids depending on any specific array library and
sticks to pure compositions of standard Array API functions). RNG needs
library-specific handling — NumPy, PyTorch, and JAX all have different
random-state models — so it's being developed here instead, as its own
project under the same umbrella of "make backend-independent array code
possible."

If this reaches a working prototype that multiple projects want to adopt,
it may eventually move under the
[data-apis](https://github.com/data-apis) organization.

## Design goals

- A **stateful, mutable `Generator`** per array namespace (NumPy, PyTorch,
  JAX, ...), analogous to `numpy.random.Generator`, created from a seed
  (and optionally a device).
- Works with JAX despite `jax.random`'s immutable, purely-functional key
  model: the generator is mutable *outside* of `jax.jit`, and exposes a
  `.key()` (or similar) method to hand a fresh, forked `KeyArray` into a
  jitted function, so users don't accidentally reuse entropy across calls.
  This mirrors how libraries like Flax and Equinox already manage JAX PRNG
  keys under the hood.
- Common sampling API across backends (`uniform`, `normal`, etc.), falling
  back to composed implementations for distributions a given backend
  doesn't provide natively (e.g. PyTorch lacks a native Gumbel sampler but
  JAX has one).
- Accepts anything seed-like as input — an integer seed, `None`, or a
  backend-native generator/key — so callers don't need to special-case each
  library.
- No dependency on any specific array library; backend support is
  implemented on a case-by-case, opt-in basis.

## Status

Early stage / pre-prototype. Design is still being discussed — see the
issue history linked above for the design conversation this project is
based on, and open an issue here to continue the discussion.

## Related discussions

- [data-apis/array-api-extra#308](https://github.com/data-apis/array-api-extra/issues/308) — the original proposal and design discussion (now converted to a discussion thread)
- [data-apis/array-api#431](https://github.com/data-apis/array-api/issues/431) and [data-apis/array-api#874](https://github.com/data-apis/array-api/issues/874) — related discussion in the Array API standard itself
- [glass-dev/rng-jax](https://github.com/glass-dev/rng-jax) — a proof-of-concept JAX RNG wrapper with a similar stateful design

## Contributing

Contributions and design discussion are welcome — please open an issue.
