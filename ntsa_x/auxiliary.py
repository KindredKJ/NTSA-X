"""Auxiliary: NTSA-X projection-native diffusive cognition (research-stage).

The Auxiliary core scans raw byte/control-symbol input directly through its
native recurrent X/Y/Z cells. Those same evolving states seed and condition
the projection-diffusion workspace; no separate semantic encoder runs before
PDC. The integrated response denoiser reads the resolved projections.

Adaptive inference allocates a bounded number of projection-refinement cycles
based on latent convergence. This version scales *inference compute*, not the
model's parameter count. It has a fixed configured projection width and does
not claim autonomous network growth or general intelligence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .cog_fu import CogFuConfig, CogFuModel
from .native_model import ByteCodec


AUXILIARY_ARCH = "kindred-auxiliary-ntsa-x-pdc-v0.2-native-scanner"
LEGACY_AUXILIARY_ARCHES = {"kindred-auxiliary-ntsa-x-pdc-v0.1"}


@dataclass(frozen=True)
class AuxiliaryConfig(CogFuConfig):
    pdc_max_cycles: int = 12
    pdc_min_cycles: int = 2
    pdc_tolerance: float = 0.015
    pdc_weight: float = 0.25

    def __post_init__(self):
        super().__post_init__()
        if not 2 <= self.pdc_max_cycles <= 64:
            raise ValueError("pdc_max_cycles must be in [2,64]")
        if not 1 <= self.pdc_min_cycles <= self.pdc_max_cycles:
            raise ValueError("pdc_min_cycles must be within the cycle budget")
        if not 0.0 <= self.pdc_tolerance <= 1.0:
            raise ValueError("pdc_tolerance must be in [0,1]")
        if not 0.0 <= self.pdc_weight <= 10.0:
            raise ValueError("pdc_weight must be in [0,10]")


@dataclass(frozen=True)
class ProjectionReceipt:
    cycles: int
    max_cycles: int
    final_relative_residual: float
    converged: bool
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycles": self.cycles,
            "max_cycles": self.max_cycles,
            "final_relative_residual": self.final_relative_residual,
            "converged": self.converged,
            "seed": self.seed,
        }


class ProjectionDenoiser(nn.Module):
    """Conditional continuous denoiser over three internal projection lanes."""

    def __init__(self, width: int):
        super().__init__()
        self.width = width
        self.state_norm = nn.LayerNorm(width * 3)
        self.condition_norm = nn.LayerNorm(width)
        self.time = nn.Sequential(nn.Linear(1, width), nn.SiLU(),
                                  nn.Linear(width, width), nn.SiLU())
        self.network = nn.Sequential(
            nn.Linear(width * 5, width * 4), nn.SiLU(),
            nn.LayerNorm(width * 4),
            nn.Linear(width * 4, width * 3), nn.Tanh())

    def forward(self, noisy: Tensor, condition: Tensor,
                noise_level: Tensor) -> Tensor:
        if (noisy.ndim != 3 or noisy.shape[1:] != (3, self.width)
                or condition.shape != (noisy.shape[0], self.width)
                or noise_level.shape != (noisy.shape[0],)):
            raise ValueError("Invalid PDC projection shapes")
        if not bool(torch.isfinite(noisy).all().item()):
            raise ValueError("PDC state must be finite")
        if not bool(torch.isfinite(condition).all().item()):
            raise ValueError("PDC condition must be finite")
        if bool(torch.any((noise_level < 0) | (noise_level > 1)).item()):
            raise ValueError("PDC noise levels must be in [0,1]")
        features = torch.cat((
            self.state_norm(noisy.flatten(1)),
            self.condition_norm(condition),
            self.time(noise_level[:, None])), dim=-1)
        return self.network(features).reshape(-1, 3, self.width)


class AuxiliaryModel(CogFuModel):
    """Single direct-scanning recurrent + PDC + response intelligence core.

    Raw experience is read by the native X/Y/Z recurrence itself. Its terminal
    state is the starting projection for PDC, rather than an encoder-produced
    condition passed into a separate cognitive stage. PDC refinement and the
    answer denoiser are jointly trained parts of this model.
    """

    architecture = AUXILIARY_ARCH

    def __init__(self, config: AuxiliaryConfig):
        if not isinstance(config, AuxiliaryConfig):
            raise TypeError("AuxiliaryModel requires AuxiliaryConfig")
        super().__init__(config)
        self.config: AuxiliaryConfig
        self.projection_denoiser = ProjectionDenoiser(config.width)
        self.projection_readout = nn.Sequential(
            nn.LayerNorm(config.width * 3),
            nn.Linear(config.width * 3, config.width), nn.Tanh())
        self.pdc_context_gate = nn.Parameter(torch.tensor(0.5))
        self.last_projection_receipt: ProjectionReceipt | None = None

    @staticmethod
    def _generator(device: torch.device, seed: int | None) -> tuple[torch.Generator, int]:
        actual_seed = int(torch.randint(0, 2**31 - 1, (1,)).item()) if seed is None else seed
        if type(actual_seed) is not int or not 0 <= actual_seed < 2**63:
            raise ValueError("seed must be an integer in [0,2^63)")
        generator = torch.Generator(device=device)
        generator.manual_seed(actual_seed)
        return generator, actual_seed

    def scan_experience(self, prefix: Tensor) -> tuple[Tensor, Tensor]:
        """Read raw experience in the core and return live X/Y/Z state + signal.

        This deliberately calls the native recurrent cells directly instead of
        running an encoder and handing its pooled output to PDC. The input
        scanner is the core's own byte embedding and recurrent state transition;
        its state is simultaneously cognition and the initial PDC projection.
        """
        if (prefix.ndim != 2 or prefix.dtype != torch.long
                or not 1 <= prefix.shape[1] <= self.config.max_context
                or bool(torch.any((prefix < 0) |
                                  (prefix >= self.cortex.config.vocab_size)).item())):
            raise ValueError("Invalid Auxiliary experience stream")
        projection, signal, _ = self._scan_chunk(prefix)
        return projection, signal

    def _scan_chunk(self, prefix: Tensor, states=None):
        """Advance the core scanner over a chunk and retain recurrent state."""
        batch = prefix.shape[0]
        states = (list(self.cortex.initial_state(batch, prefix.device))
                  if states is None else list(states))
        embedded = self.cortex.dropout(self.cortex.embedding(prefix))
        for position in range(prefix.shape[1]):
            value = embedded[:, position, :]
            following = []
            for cell, prior in zip(self.cortex.cells, states):
                value, state = cell(value, prior)
                following.append(state)
            states = following
        # Keep the X/Y/Z axes intact while folding recurrence depth into one
        # shared cognitive state, matching the established native state shape.
        projection = torch.stack(states, dim=0).mean(dim=0)
        return projection, value, tuple(states)

    def _target_projections(self, prefix: Tensor) -> tuple[Tensor, Tensor]:
        return self.scan_experience(prefix)

    def _differentiable_projection(self, prefix: Tensor) -> Tensor:
        """Fixed-budget PDC path used by joint training and validation loss."""
        target, condition = self.scan_experience(prefix)
        # The core's live scan is the experience-bearing projection. Noise is
        # applied inside the same workspace; there is no encoder handoff.
        state = target + torch.randn_like(target)
        for index in range(self.config.pdc_max_cycles):
            current = 1.0 - index / self.config.pdc_max_cycles
            following = max(0.0, 1.0 - (index + 1) /
                            self.config.pdc_max_cycles)
            level = torch.full((prefix.shape[0],), current,
                               device=prefix.device, dtype=state.dtype)
            next_level = torch.full_like(level, following)
            estimate = self.projection_denoiser(state, condition, level)
            if following == 0.0:
                state = estimate
                break
            noise = ((state - (1.0 - level[:, None, None]).sqrt() * estimate)
                     / level[:, None, None].sqrt().clamp_min(1e-6))
            state = ((1.0 - next_level[:, None, None]).sqrt() * estimate
                     + next_level[:, None, None].sqrt() * noise)
        return state

    def cognitive_context(self, prefix: Tensor) -> Tensor:
        """Use the resolved projection as the same cortex's denoiser context."""
        projected = self.projection_readout(
            self._differentiable_projection(prefix).flatten(1))
        # This gate is trained by the response objective as well as PDC's
        # reconstruction objective, so the response path can learn how much
        # the resolved projection should influence its denoising context.
        return torch.sigmoid(self.pdc_context_gate) * projected

    def fuse_context(self, recurrent_context: Tensor,
                     auxiliary_context: Tensor | None) -> Tensor:
        if auxiliary_context is None:
            return recurrent_context
        if auxiliary_context.shape != recurrent_context.shape:
            raise ValueError("PDC and recurrent context shapes differ")
        return recurrent_context + torch.sigmoid(self.pdc_context_gate) * auxiliary_context

    def joint_loss(self, messages: list[dict], *, noise_fraction: float = 0.5,
                   seed: int | None = None) -> dict[str, Tensor]:
        """Jointly train causal response, byte diffusion, and PDC projections."""
        # Keep the stochastic PDC trajectory paired with the fixed byte-mask
        # seed, so held-out measurements are reproducible.
        actual_seed = (int(torch.randint(0, 2**31 - 1, (1,)).item())
                       if seed is None else seed)
        cuda_devices = ([self.cortex.embedding.weight.device.index or 0]
                        if self.cortex.embedding.weight.is_cuda else [])
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(actual_seed)
            if cuda_devices:
                torch.cuda.manual_seed_all(actual_seed)
            base = super().joint_loss(messages, noise_fraction=noise_fraction,
                                      seed=actual_seed)
        prefix, _ = self.codec.conversation(
            messages[:-1], generation=True, max_tokens=self.config.max_context)
        device = self.cortex.embedding.weight.device
        tokens = torch.tensor([prefix], dtype=torch.long, device=device)
        target, condition = self._target_projections(tokens)
        generator, actual_seed = self._generator(device, actual_seed)
        level = torch.rand((1,), generator=generator, device=device).clamp_(
            min=1e-4, max=1.0)
        noise = torch.randn(target.shape, generator=generator, device=device,
                            dtype=target.dtype)
        noisy = ((1.0 - level[:, None, None]).sqrt() * target
                 + level[:, None, None].sqrt() * noise)
        estimate = self.projection_denoiser(noisy, condition, level)
        pdc_loss = F.mse_loss(estimate, target)
        loss = base["loss"] + self.config.pdc_weight * pdc_loss
        return {**base, "pdc_loss": pdc_loss, "loss": loss,
                "pdc_seed": torch.tensor(actual_seed, device=device)}

    @torch.inference_mode()
    def resolve_projections(self, prefix: list[int], *,
                            max_cycles: int | None = None,
                            adaptive: bool = True, seed: int = 0
                            ) -> tuple[Tensor, ProjectionReceipt]:
        """Resolve X/Y/Z projections with a bounded reverse-diffusion rollout."""
        if (not isinstance(prefix, list) or not prefix
                or len(prefix) > self.config.max_context
                or any(type(token) is not int or not 0 <= token < 262
                       for token in prefix)):
            raise ValueError("Invalid Auxiliary input stream")
        cycles = self.config.pdc_max_cycles if max_cycles is None else max_cycles
        if type(cycles) is not int or not 1 <= cycles <= self.config.pdc_max_cycles:
            raise ValueError("max_cycles exceeds Auxiliary PDC budget")
        if not adaptive and cycles < self.config.pdc_min_cycles:
            # Fixed-cycle ablations may use one pass by design.
            pass
        device = self.cortex.embedding.weight.device
        generator, actual_seed = self._generator(device, seed)
        was_training = self.training
        self.eval()
        tokens = torch.tensor([prefix], dtype=torch.long, device=device)
        experience, condition = self.scan_experience(tokens)
        state = experience + torch.randn(
            experience.shape, generator=generator, device=device,
            dtype=experience.dtype)
        previous_estimate = None
        final_residual = math.inf
        converged = False
        used = 0
        for index in range(cycles):
            t_value = 1.0 - index / cycles
            next_value = max(0.0, 1.0 - (index + 1) / cycles)
            t = torch.full((1,), t_value, device=device, dtype=state.dtype)
            next_t = torch.full((1,), next_value, device=device,
                                dtype=state.dtype)
            estimate = self.projection_denoiser(state, condition, t)
            used = index + 1
            if previous_estimate is not None:
                numerator = torch.linalg.vector_norm(
                    estimate - previous_estimate)
                denominator = torch.linalg.vector_norm(previous_estimate).clamp_min(1e-8)
                final_residual = float((numerator / denominator).item())
                if (adaptive and used >= self.config.pdc_min_cycles
                        and final_residual <= self.config.pdc_tolerance):
                    converged = True
                    state = estimate
                    break
            if next_value == 0.0:
                state = estimate
                final_residual = 0.0 if previous_estimate is None else final_residual
                break
            eps_estimate = ((state - (1.0 - t[:, None, None]).sqrt() * estimate)
                            / t[:, None, None].sqrt().clamp_min(1e-6))
            state = ((1.0 - next_t[:, None, None]).sqrt() * estimate
                     + next_t[:, None, None].sqrt() * eps_estimate)
            previous_estimate = estimate
        receipt = ProjectionReceipt(
            cycles=used, max_cycles=cycles,
            final_relative_residual=(0.0 if not math.isfinite(final_residual)
                                     else final_residual),
            converged=converged, seed=actual_seed)
        self.train(was_training)
        return state, receipt

    @torch.inference_mode()
    def resolve_projections_overlapped(self, prefix: list[int], *,
                                       max_cycles: int | None = None,
                                       seed: int = 0
                                       ) -> tuple[Tensor, ProjectionReceipt]:
        """Overlap PDC refinement with direct scanning of the remaining input.

        The core scans a short input prefix to create a live X/Y/Z projection.
        PDC refines that state in a worker while the same recurrent core
        continues scanning the remaining experience. At the dependency boundary,
        the PDC result is fused with the final live scan state and receives one
        final denoising update. Total PDC updates remain equal to max_cycles.
        This is an inference schedule inside one model, not a second PDC
        service. It is an ablation: its early PDC branch sees a prefix snapshot
        and may trade task quality for overlap.
        """
        from concurrent.futures import ThreadPoolExecutor

        if (not isinstance(prefix, list) or not prefix
                or len(prefix) > self.config.max_context
                or any(type(token) is not int or not 0 <= token < 262
                       for token in prefix)):
            raise ValueError("Invalid Auxiliary input stream")
        cycles = self.config.pdc_max_cycles if max_cycles is None else max_cycles
        if type(cycles) is not int or not 2 <= cycles <= self.config.pdc_max_cycles:
            raise ValueError("overlapped PDC requires 2..pdc_max_cycles updates")
        if len(prefix) < 4:
            raise ValueError("overlapped PDC requires at least four input symbols")

        device = self.cortex.embedding.weight.device
        generator, actual_seed = self._generator(device, seed)
        was_training = self.training
        self.eval()
        try:
            tokens = torch.tensor([prefix], dtype=torch.long, device=device)
            seed_length = min(
                len(prefix) - 1,
                max(3, min(16, (len(prefix) + 7) // 8)))
            first_projection, first_signal, recurrent_state = self._scan_chunk(
                tokens[:, :seed_length])
            pdc_generator, _ = self._generator(device, actual_seed)
            pdc_state = first_projection + torch.randn(
                first_projection.shape, generator=pdc_generator,
                device=device, dtype=first_projection.dtype)

            def refine_prefix_snapshot():
                for index in range(cycles - 1):
                    t_value = 1.0 - index / (cycles - 1)
                    next_value = max(0.0, 1.0 - (index + 1) / (cycles - 1))
                    t = torch.full((1,), t_value, device=device,
                                   dtype=pdc_state.dtype)
                    next_t = torch.full_like(t, next_value)
                    estimate = self.projection_denoiser(
                        pdc_state, first_signal, t)
                    if next_value == 0.0:
                        return estimate
                    eps_estimate = (
                        pdc_state - (1.0 - t[:, None, None]).sqrt() * estimate
                    ) / t[:, None, None].sqrt().clamp_min(1e-6)
                    pdc_state.copy_(
                        (1.0 - next_t[:, None, None]).sqrt() * estimate
                        + next_t[:, None, None].sqrt() * eps_estimate)
                raise RuntimeError("PDC overlap branch did not resolve")

            pdc_stream = (torch.cuda.Stream(device=device)
                          if device.type == "cuda" else None)
            scan_stream = (torch.cuda.current_stream(device)
                           if device.type == "cuda" else None)

            def run_refinement():
                if pdc_stream is None:
                    with torch.inference_mode():
                        return refine_prefix_snapshot()
                pdc_stream.wait_stream(scan_stream)
                with torch.cuda.stream(pdc_stream):
                    with torch.inference_mode():
                        result = refine_prefix_snapshot()
                return result

            with ThreadPoolExecutor(max_workers=1,
                                    thread_name_prefix="auxiliary-pdc") as pool:
                future = pool.submit(run_refinement)
                if tokens.shape[1] > seed_length:
                    live_projection, final_signal, _ = self._scan_chunk(
                        tokens[:, seed_length:], recurrent_state)
                else:
                    live_projection, final_signal = (
                        first_projection, first_signal)
                refined_snapshot = future.result()
            if pdc_stream is not None:
                torch.cuda.current_stream(device).wait_stream(pdc_stream)

            fused = (refined_snapshot + live_projection) * 0.5
            final_level = torch.zeros((1,), device=device, dtype=fused.dtype)
            resolved = self.projection_denoiser(
                fused, final_signal, final_level)
            relative_residual = float((
                torch.linalg.vector_norm(resolved - refined_snapshot)
                / torch.linalg.vector_norm(refined_snapshot).clamp_min(1e-8)
            ).item())
            receipt = ProjectionReceipt(
                cycles=cycles, max_cycles=cycles,
                final_relative_residual=relative_residual,
                converged=relative_residual <= self.config.pdc_tolerance,
                seed=actual_seed)
            return resolved, receipt
        finally:
            self.train(was_training)

    @torch.inference_mode()
    def projected_context(self, projections: Tensor) -> Tensor:
        if projections.shape != (1, 3, self.config.width):
            raise ValueError("Expected one resolved X/Y/Z projection triplet")
        return self.projection_readout(projections.flatten(1))

    def draft(self, prefix: list[int], *, max_new_tokens: int = 128,
              temperature: float = 0.0, steps: int | None = None,
              cached_context: Tensor | None = None,
              pdc_cycles: int | None = None, adaptive_pdc: bool = True,
              seed: int = 0, pdc_schedule: str = "serial") -> list[int]:
        if cached_context is None:
            if pdc_schedule == "serial":
                projection, receipt = self.resolve_projections(
                    prefix, max_cycles=pdc_cycles, adaptive=adaptive_pdc,
                    seed=seed)
            elif pdc_schedule == "overlap":
                if adaptive_pdc:
                    raise ValueError(
                        "overlap schedule currently requires fixed PDC cycles")
                projection, receipt = self.resolve_projections_overlapped(
                    prefix, max_cycles=pdc_cycles, seed=seed)
            else:
                raise ValueError("pdc_schedule must be serial or overlap")
            cached_context = self.projected_context(projection)
            self.last_projection_receipt = receipt
        return super().draft(
            prefix, max_new_tokens=max_new_tokens, temperature=temperature,
            steps=steps, cached_context=cached_context)

    def unified_draft(self, prefix: list[int], *, max_new_tokens: int = 48,
                      temperature: float = 0.0, diffusion_mix: float = 0.5,
                      pdc_cycles: int | None = None, adaptive_pdc: bool = True,
                      seed: int = 0,
                      pdc_schedule: str = "serial") -> list[int]:
        if pdc_schedule == "serial":
            projection, receipt = self.resolve_projections(
                prefix, max_cycles=pdc_cycles, adaptive=adaptive_pdc,
                seed=seed)
        elif pdc_schedule == "overlap":
            if adaptive_pdc:
                raise ValueError(
                    "overlap schedule currently requires fixed PDC cycles")
            projection, receipt = self.resolve_projections_overlapped(
                prefix, max_cycles=pdc_cycles, seed=seed)
        else:
            raise ValueError("pdc_schedule must be serial or overlap")
        self.last_projection_receipt = receipt
        context = self.projected_context(projection)
        return super().unified_draft(
            prefix, max_new_tokens=max_new_tokens, temperature=temperature,
            diffusion_mix=diffusion_mix, cached_context=context)


def load_checkpoint(folder: str, *, device: str = "cpu") -> AuxiliaryModel:
    from pathlib import Path
    import json

    path = Path(folder)
    info = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if info.get("architecture") not in ({AUXILIARY_ARCH}
                                         | LEGACY_AUXILIARY_ARCHES):
        raise ValueError("Not an Auxiliary PDC checkpoint")
    config = AuxiliaryConfig(**info["config"])
    model = AuxiliaryModel(config)
    state = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
