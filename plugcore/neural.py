"""PlugCore trainable recurrent skills for existing Auxiliary/NTSA-X cortex.

This module does NOT create a language model, decoder, independent state store,
training corpus, or alternate inference route. A bank attaches forward hooks to
reviewed OrbitCell instances, sharing the same X/Y/Z recurrent state, and can
be detached without changing the original model's state_dict/ABI.

Only trusted Python code should be imported; snapshot loading is weights-only.
This is not a sandbox for adversarial Python extensions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Callable, Mapping
import weakref

import torch
from torch import Tensor, nn
from torch.nn import functional as F

PLUGCORE_SCHEMA = "kindred.plugcore.neural-skill/v1"
CORE_ARCHITECTURE = "ntsa-x-orbit-cell-hook/v1"
_NAME = re.compile(r"^[a-z][a-z0-9_-]{2,63}$")
_SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


@dataclass(frozen=True)
class SkillSpec:
    id: str
    version: str
    width: int
    rank: int
    layer: int
    scale: float = 0.25
    schema: str = PLUGCORE_SCHEMA
    architecture: str = CORE_ARCHITECTURE

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not _NAME.fullmatch(self.id):
            raise ValueError("Skill id must be lowercase, portable and 3-64 chars")
        if not isinstance(self.version, str) or not _SEMVER.fullmatch(self.version):
            raise ValueError("Skill version must be strict SemVer")
        if type(self.width) is not int or not 16 <= self.width <= 4096:
            raise ValueError("Skill width out of range")
        if type(self.rank) is not int or not 1 <= self.rank <= min(256, self.width):
            raise ValueError("Skill rank out of range")
        if type(self.layer) is not int or not 0 <= self.layer <= 23:
            raise ValueError("Skill layer out of range")
        if not isinstance(self.scale, (float, int)) or not 0 < self.scale <= 1:
            raise ValueError("Skill scale out of range")
        if self.schema != PLUGCORE_SCHEMA or self.architecture != CORE_ARCHITECTURE:
            raise ValueError("Unsupported skill protocol")

    @classmethod
    def from_mapping(cls, obj: Mapping) -> "SkillSpec":
        if not isinstance(obj, dict) or set(obj) != set(cls.__dataclass_fields__):
            raise ValueError("Skill manifest fields mismatch")
        return cls(**obj)


def _safe_key(skill_id: str) -> str:
    return "s" + hashlib.sha256(skill_id.encode("utf-8")).hexdigest()[:24]


class TrainableRecurrentSkill(nn.Module):
    """Identity-initialized, gated low-rank residual on ONE recurrent X/Y/Z stream.

    The skill modifies the existing cell's output and existing X/Y/Z state;
    neither logits nor answers are computed by this module. Gradients flow via
    ordinary backpropagation from the host model's own native loss.
    """
    def __init__(self, spec: SkillSpec):
        super().__init__()
        self.spec = spec
        self.down = nn.Linear(2 * spec.width, spec.rank, bias=False)
        self.up_output = nn.Linear(spec.rank, spec.width, bias=False)
        self.up_state = nn.Linear(spec.rank, 3 * spec.width, bias=False)
        self.gate = nn.Linear(2 * spec.width, 1)
        nn.init.zeros_(self.up_output.weight)
        nn.init.zeros_(self.up_state.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, output: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        if (not isinstance(output, tuple) or len(output) != 2):
            raise TypeError("Orbit cell must return (output, shared_state)")
        vector, state = output
        w = self.spec.width
        if vector.ndim != 2 or state.shape != (vector.shape[0], 3, w) or vector.shape[1] != w:
            raise ValueError("Incompatible recurrent output and X/Y/Z state")
        joined = torch.cat((vector, state.mean(dim=1)), dim=-1)
        low_rank = F.silu(self.down(joined))
        gating = torch.sigmoid(self.gate(joined)) * self.spec.scale
        next_output = vector + gating * self.up_output(low_rank)
        next_state = state + gating.unsqueeze(-1) * self.up_state(low_rank).reshape(-1, 3, w)
        return next_output, next_state


class PlugCoreBank(nn.Module):
    """One detachable bank of trainable skills on a single native cortex.

    Model checkpoints remain *byte-level key compatible*: this bank is NOT
    registered as a child of the base model, and callbacks are removed by
    detach(). Optimizers may train bank.parameters() while core weights stay
    frozen. The caller manages concurrency at skill-replacement boundaries.
    """
    def __init__(self):
        super().__init__()
        self.skills = nn.ModuleDict()
        self._specs: dict[str, SkillSpec] = {}
        self._enabled: dict[str, bool] = {}
        self._handles = []
        self._core_ref = None

    @property
    def specs(self) -> tuple[SkillSpec, ...]:
        return tuple(self._specs[k] for k in sorted(self._specs))

    def register(self, spec: SkillSpec) -> TrainableRecurrentSkill:
        if not isinstance(spec, SkillSpec):
            raise TypeError("Expected reviewed SkillSpec")
        if spec.id in self._specs:
            raise ValueError("Duplicate skill id; use a versioned replacement")
        core = self._core_ref() if self._core_ref else None
        if core is not None:
            self._assert_compatible(core, spec)
        key = _safe_key(spec.id)
        if key in self.skills:
            raise ValueError("Skill storage key collision")
        module = TrainableRecurrentSkill(spec)
        if core is not None:
            module.to(device=next(core.parameters()).device,
                      dtype=next(core.parameters()).dtype)
        self.skills[key] = module
        self._specs[spec.id] = spec
        self._enabled[spec.id] = True
        return module

    def remove(self, skill_id: str) -> None:
        if skill_id not in self._specs:
            raise KeyError(skill_id)
        del self.skills[_safe_key(skill_id)]
        del self._specs[skill_id]
        del self._enabled[skill_id]

    def enable(self, skill_id: str, enabled: bool) -> None:
        if skill_id not in self._specs:
            raise KeyError(skill_id)
        if type(enabled) is not bool:
            raise TypeError("Enabled must be bool")
        self._enabled[skill_id] = enabled

    @staticmethod
    def _resolve_core(model):
        core = getattr(model, "cortex", model)
        if (not isinstance(core, nn.Module) or not isinstance(getattr(core, "cells", None), nn.ModuleList)
                or not hasattr(core, "config") or not hasattr(core.config, "width")):
            raise TypeError("Expected existing NativeOrbitModel or CogFuModel.cortex")
        return core

    @staticmethod
    def _assert_compatible(core, spec: SkillSpec) -> None:
        if spec.width != core.config.width or spec.layer >= len(core.cells):
            raise ValueError("Skill incompatible with existing cortex dimensions")
        if not isinstance(core.cells[spec.layer], nn.Module):
            raise TypeError("Invalid host cortex cell")

    def attach(self, model) -> None:
        if self._core_ref is not None:
            raise RuntimeError("Bank already attached; detach first")
        core = self._resolve_core(model)
        current = getattr(core, "_plugcore_owner_ref", None)
        if current is not None and current() is not None:
            raise RuntimeError("Another PlugCore bank owns this cortex")
        for spec in self.specs:
            self._assert_compatible(core, spec)
        ref_param = next(core.parameters())
        self.to(device=ref_param.device, dtype=ref_param.dtype)
        bank_ref = weakref.ref(self)

        def make_hook(layer: int):
            def update(_layer, _input, output):
                bank = bank_ref()
                if bank is None:
                    return output
                # Stable order independent of registration sequence.
                for skill_id in sorted(bank._specs):
                    spec = bank._specs[skill_id]
                    if spec.layer == layer and bank._enabled[skill_id]:
                        output = bank.skills[_safe_key(skill_id)](output)
                return output
            return update

        installed = []
        try:
            for layer in range(len(core.cells)):
                installed.append(core.cells[layer].register_forward_hook(make_hook(layer)))
        except Exception:
            for handle in installed:
                handle.remove()
            raise
        self._handles = installed
        self._core_ref = weakref.ref(core)
        # The owner marker is not registered as part of nn.Module state_dict.
        object.__setattr__(core, "_plugcore_owner_ref", bank_ref)

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        core = self._core_ref() if self._core_ref is not None else None
        if core is not None:
            owner = getattr(core, "_plugcore_owner_ref", None)
            if owner is not None and owner() is self:
                delattr(core, "_plugcore_owner_ref")
        self._core_ref = None

    def __enter__(self):
        if self._core_ref is None:
            raise RuntimeError("Attach bank before entering context")
        return self

    def __exit__(self, *_):
        self.detach()

    def save(self, target: str | Path) -> Path:
        """Write portable weights-only skill files plus hash-checked manifest."""
        path = Path(target).resolve()
        path.mkdir(parents=True, exist_ok=True)
        entries = []
        for spec in self.specs:
            module = self.skills[_safe_key(spec.id)]
            file_name = f"skill-{_safe_key(spec.id)}.pt"
            final = path / file_name
            with tempfile.NamedTemporaryFile(prefix=".plugcore-", suffix=".pt",
                                             dir=path, delete=False) as f:
                temporary = Path(f.name)
            try:
                torch.save({k: v.detach().to("cpu").contiguous() for k, v in module.state_dict().items()}, temporary)
                data = temporary.read_bytes()
                if len(data) > 128 * 1024 * 1024:
                    raise ValueError("Skill weights exceed maximum size")
                os.replace(temporary, final)
            finally:
                temporary.unlink(missing_ok=True)
            entries.append({"spec": asdict(spec), "file": file_name,
                            "sha256": hashlib.sha256(data).hexdigest(),
                            "enabled": self._enabled[spec.id]})
        manifest = {"format": "kindred.plugcore-bank/v1", "entries": entries,
                    "integrity": "sha256-data-integrity-not-publisher-authenticity"}
        serialized = json.dumps(manifest, sort_keys=True, indent=2).encode()
        with tempfile.NamedTemporaryFile(prefix=".plugcore-", dir=path,
                                         delete=False) as f:
            f.write(serialized)
            tmp = Path(f.name)
        os.replace(tmp, path / "plugcore.json")
        return path / "plugcore.json"

    @classmethod
    def load(cls, target: str | Path) -> "PlugCoreBank":
        """Load ONLY approved built-in kernels with verified weights-only data."""
        path = Path(target).resolve()
        manifest_path = path / "plugcore.json"
        if not manifest_path.is_file() or manifest_path.stat().st_size > 128 * 1024:
            raise ValueError("Missing/oversized PlugCore manifest")
        manifest = json.loads(manifest_path.read_text("utf-8"))
        if (not isinstance(manifest, dict) or set(manifest) != {"format", "entries", "integrity"}
                or manifest["format"] != "kindred.plugcore-bank/v1"
                or not isinstance(manifest["entries"], list)
                or len(manifest["entries"]) > 128):
            raise ValueError("Invalid PlugCore bank manifest")
        bank = cls()
        for entry in manifest["entries"]:
            if not isinstance(entry, dict) or set(entry) != {"spec", "file", "sha256", "enabled"}:
                raise ValueError("Malformed skill entry")
            spec = SkillSpec.from_mapping(entry["spec"])
            name = f"skill-{_safe_key(spec.id)}.pt"
            if entry["file"] != name or not re.fullmatch(r"[0-9a-f]{64}", str(entry["sha256"])):
                raise ValueError("Unsafe skill file or digest")
            filename = path / name
            if not filename.is_file() or filename.is_symlink() or filename.stat().st_size > 128 * 1024 * 1024:
                raise ValueError("Missing/unsafe skill weights file")
            if hashlib.sha256(filename.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError("Skill weights failed SHA-256 integrity check")
            module = bank.register(spec)
            state = torch.load(filename, map_location="cpu", weights_only=True)
            if not isinstance(state, dict) or not all(isinstance(v, Tensor) for v in state.values()):
                raise ValueError("Invalid weights-only skill state")
            module.load_state_dict(state, strict=True)
            bank.enable(spec.id, entry["enabled"])
        return bank


class PlugCoreTrainer:
    """Opt-in gradient training of attached skills using the base model loss.

    Use `step(lambda: model.joint_loss(messages, seed=...)['loss'])`
    for CogFuModel, or `step(lambda: model.loss(tokens, labels))` for native
    cortex. Base weights remain fixed unless user trains them separately.
    """
    def __init__(self, model: nn.Module, bank: PlugCoreBank, *, lr: float = 0.002,
                 weight_decay: float = 0.0, grad_clip: float = 1.0):
        if bank._core_ref is None or bank._core_ref() is not PlugCoreBank._resolve_core(model):
            raise RuntimeError("Attach bank to this model before training")
        if not bank.skills:
            raise ValueError("No trainable skills registered")
        if not 0 < lr <= 0.1 or not 0 <= weight_decay <= 1 or not 0 < grad_clip <= 100:
            raise ValueError("Invalid optimizer hyperparameters")
        self.model, self.bank, self.grad_clip = model, bank, grad_clip
        self._training = model.training
        self._flags = [(p, p.requires_grad) for p in model.parameters()]
        for p, _ in self._flags:
            p.requires_grad_(False)
        model.eval()
        bank.train()
        self.optimizer = torch.optim.AdamW(bank.parameters(), lr=lr, weight_decay=weight_decay)
        self.steps = 0

    def step(self, loss_fn: Callable[[], Tensor]) -> float:
        if not callable(loss_fn):
            raise TypeError("Loss callback required")
        self.optimizer.zero_grad(set_to_none=True)
        value = loss_fn()
        if not isinstance(value, Tensor) or value.ndim != 0 or not bool(torch.isfinite(value).item()):
            raise ValueError("Expected finite scalar native model loss")
        value.backward()
        torch.nn.utils.clip_grad_norm_(self.bank.parameters(), self.grad_clip,
                                       error_if_nonfinite=True)
        self.optimizer.step()
        self.steps += 1
        return float(value.detach().item())

    def close(self) -> None:
        for param, required in self._flags:
            param.requires_grad_(required)
        self.model.train(self._training)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
