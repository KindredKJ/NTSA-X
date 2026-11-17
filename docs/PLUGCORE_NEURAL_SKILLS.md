# PlugCore by Kindred Labs — neural module contract v0.2.0

**Product:** PlugCore. **Owner:** Kindred Labs. **Authoritative intelligence:** Auxiliary/NTSA-X. **Primary requirement:** all mandatory inference, training, skill weights and capability execution must operate on Kindred-owned, offline-capable infrastructure.

## Architecture

Existing `CogFuModel` -> **existing `NativeOrbitModel.cortex`** -> existing `OrbitCell` tuple `(x, X/Y/Z state)` -> **reviewed PlugCore differentiable residual** -> same upstream `NativeOrbitModel` state update -> same CogFu recurrent+masked joint heads -> same answer path. PlugCore is never an independent model or decoder.

`PlugCoreBank` and `TrainableRecurrentSkill` are additive and *not registered as submodules of the original model*, so existing model `state_dict` keys and original checkpoint loading remain unchanged. A bank attaches verified forward hooks to existing cells after checkpoint loading, and can be removed without modifying any base weights. The bank holds its own trainable weight state and optimizer. Identity initialization ensures the output is initially identical to the unmodified host. A module consumes and emits one shared X/Y/Z state; multi-module composition uses stable sorted IDs.

### Skill manifest

```
{
  "id": "kindred_rule_skill",
  "version": "0.2.0",
  "width": 16,
  "rank": 8,
  "layer": 0,
  "scale": 1.0,
  "schema": "kindred.plugcore.neural-skill/v1",
  "architecture": "ntsa-x-orbit-cell-hook/v1"
}
```

### Operations

- `register(spec)`: register exactly one locally reviewed `TrainableRecurrentSkill` implementation.
- `attach(existing_model)`: bind a bank to one real native cortex in the current process.
- `enable(id, bool)` / `remove(id)` / `detach()`: control or fully unplug skills without altering base weights.
- `PlugCoreTrainer.step(lambda: existing_model.joint_loss(examples)["loss"])`: optimize only plug-in weights against the **existing model loss**.
- `save(directory)` / `load(directory)`: checkpoint the plug-in bank, retaining the existing model checkpoint separately. SHA-256 verifies file integrity; a manifest is not a publisher signature.

## Evaluation

Synthetic training proof: 24 symbol IDs with 14 held-out IDs, binary feature-rule transfer; measured baseline holdout loss ~0.693 and after 90 adapter-only steps ~0.00055, base weights unchanged, detachment restores baseline. This is a small representational-learning proof, **not** a broad AGI result. Real CogFu integration tests are included, but still need to be run with the source and actual weights in the upstream repository.

## Remaining deployment work

Actual source integration and CI passing, real pretrained Auxiliary checkpoint validation, performance benchmarks, memory/goal/policy coordination, learned experience applicability and negative transfer, offline model training data and hardware profiling, signed publisher chain, local distributed authoritative checkpoint replicas, and mobile native application APIs. Not every device is verified merely by cross-compilation. iOS host embedding is unimplemented.
