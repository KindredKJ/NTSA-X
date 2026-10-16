"""COG-FU: coupled NTSA-X cognitive recurrence + discrete absorbing-mask diffusion.

Research-stage, from-scratch native architecture: no Transformer, attention,
external teacher or externally loaded foundation-model weights. Both branches
are optimized jointly. Diffusion is *discrete masked-token denoising*, not a
continuous image diffusion model, and generated dialogue is not verified truth.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterator
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .native_model import ByteCodec, NativeConfig, NativeOrbitModel


MASK_ID = 262
COG_FU_ARCH = "ntsa-x-cog-fu-discrete-v1"


@dataclass(frozen=True)
class CogFuConfig:
    width: int = 96
    layers: int = 2
    max_context: int = 2048
    max_answer: int = 256
    diffusion_layers: int = 3
    denoise_steps: int = 8
    dropout: float = 0.0
    diffusion_weight: float = 1.0

    def __post_init__(self):
        NativeConfig(width=self.width, layers=self.layers,
                     max_context=self.max_context, dropout=self.dropout)
        if not 8 <= self.max_answer <= 1024:
            raise ValueError("max_answer must be in [8,1024]")
        if not 1 <= self.diffusion_layers <= 16:
            raise ValueError("diffusion_layers must be in [1,16]")
        if not 1 <= self.denoise_steps <= 32:
            raise ValueError("denoise_steps must be in [1,32]")
        if not 0.0 <= self.diffusion_weight <= 10.0:
            raise ValueError("Invalid diffusion weight")

    def to_dict(self) -> dict:
        return asdict(self)


class ResidualDenoiseBlock(nn.Module):
    """Bidirectional local token interactions with cognitive-state FiLM."""

    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.mix = nn.Conv1d(width, width * 2, kernel_size=3, padding=1)
        self.project = nn.Conv1d(width, width, kernel_size=3, padding=1)
        self.film = nn.Linear(width, width * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        scale, bias = self.film(context).chunk(2, dim=-1)
        hidden = self.norm(x) * (1 + torch.tanh(scale[:, None, :])) + bias[:, None, :]
        hidden = F.glu(self.mix(hidden.transpose(1, 2)), dim=1)
        hidden = self.project(hidden).transpose(1, 2)
        return x + self.dropout(hidden)


class CogFuModel(nn.Module):
    """One shared trainable cognitive/diffusion model.

    The native recurrent cortex encodes prior conversation and is also trained
    on causal assistant tokens. The denoiser conditions on that same cortex's
    X/Y/Z states, making its masked-byte loss differentiable into the cortex.
    """

    architecture = COG_FU_ARCH

    def __init__(self, config: CogFuConfig):
        super().__init__()
        self.config = config
        self.cortex = NativeOrbitModel(NativeConfig(
            width=config.width, layers=config.layers,
            max_context=config.max_context, dropout=config.dropout))
        self.noisy_embedding = nn.Embedding(MASK_ID + 1, config.width)
        self.position_embedding = nn.Embedding(config.max_answer, config.width)
        self.time_embedding = nn.Sequential(
            nn.Linear(1, config.width), nn.SiLU(),
            nn.Linear(config.width, config.width))
        self.context_projection = nn.Sequential(
            nn.LayerNorm(config.layers * 3 * config.width),
            nn.Linear(config.layers * 3 * config.width, config.width),
            nn.Tanh())
        self.blocks = nn.ModuleList([
            ResidualDenoiseBlock(config.width, config.dropout)
            for _ in range(config.diffusion_layers)])
        self.final_norm = nn.LayerNorm(config.width)
        self.diffusion_head = nn.Linear(config.width, MASK_ID)
        self.codec = ByteCodec()
        # Serving-only choice, not stored in checkpoint. Compare the causal
        # cortex against diffusion without retraining or changing weights.
        self.inference_decoder = "diffusion"

    def cognitive_context(self, prefix: Tensor) -> Tensor:
        """Encode a prompt ONCE into cross-conditioned recurrent X/Y/Z state."""
        _, states = self.cortex(prefix)
        joined = torch.cat([lane.flatten(1) for lane in states], dim=-1)
        return self.context_projection(joined)

    def denoise_logits(self, prefix: Tensor, noisy: Tensor,
                       noise_level: Tensor, *,
                       cached_context: Tensor | None = None) -> Tensor:
        if (prefix.ndim != 2 or noisy.ndim != 2
                or prefix.shape[0] != noisy.shape[0]
                or prefix.dtype != torch.long or noisy.dtype != torch.long
                or not 1 <= prefix.shape[1] <= self.config.max_context
                or not 1 <= noisy.shape[1] <= self.config.max_answer
                or noise_level.shape != (prefix.shape[0],)):
            raise ValueError("Invalid COG-FU conditioning shapes")
        if bool(torch.any((noise_level < 0) | (noise_level > 1)).item()):
            raise ValueError("Noise levels must be in [0,1]")
        context = (self.cognitive_context(prefix) if cached_context is None
                   else cached_context)
        if context.shape != (prefix.shape[0], self.config.width):
            raise ValueError("Invalid cached cognitive state")
        positions = torch.arange(noisy.shape[1], device=noisy.device)
        x = (self.noisy_embedding(noisy)
             + self.position_embedding(positions)[None, :, :]
             + self.time_embedding(noise_level[:, None])
             + context[:, None, :])
        for block in self.blocks:
            x = block(x, context)
        return self.diffusion_head(self.final_norm(x))

    def joint_loss(self, messages: list[dict], *, noise_fraction: float = 0.5,
                   seed: int | None = None) -> dict[str, Tensor]:
        """Concurrent native AR + masked-diffusion training on ONE example.

        A final assistant answer is the target. Only deliberately corrupted
        response positions contribute to diffusion CE, avoiding target leakage.
        """
        if not 0 < noise_fraction <= 1:
            raise ValueError("noise_fraction must be within (0,1]")
        tokens, labels = self.codec.conversation(
            messages, max_tokens=self.config.max_context)
        if len(messages) < 2 or messages[-2]["role"] != "user":
            raise ValueError("Expected a user turn before the target assistant")
        prefix, _ = self.codec.conversation(
            messages[:-1], generation=True, max_tokens=self.config.max_context)
        answer = [*messages[-1]["content"].encode("utf-8"), ByteCodec.EOS]
        if len(answer) > self.config.max_answer:
            raise ValueError("Answer exceeds COG-FU maximum")
        device = self.cortex.embedding.weight.device
        full = torch.tensor([tokens], dtype=torch.long, device=device)
        target = torch.tensor([labels], dtype=torch.long, device=device)
        prefix_tensor = torch.tensor([prefix], dtype=torch.long, device=device)
        answer_tensor = torch.tensor([answer], dtype=torch.long, device=device)
        rng = torch.Generator(device=device)
        if seed is None:
            # Draw from PyTorch's global generator, so torch.manual_seed works.
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        rng.manual_seed(seed)
        corrupted = (torch.rand(answer_tensor.shape, device=device,
                                generator=rng) < noise_fraction)
        if not corrupted.any():
            corrupted[0, int(torch.randint(0, len(answer), (1,), generator=rng,
                                           device=device).item())] = True
        masked_answer = answer_tensor.masked_fill(corrupted, MASK_ID)
        levels = torch.tensor([noise_fraction], device=device, dtype=torch.float)
        logits = self.denoise_logits(prefix_tensor, masked_answer, levels)
        diffusion_loss = F.cross_entropy(
            logits[corrupted], answer_tensor[corrupted])
        cognitive_loss = self.cortex.loss(full, target)
        loss = cognitive_loss + self.config.diffusion_weight * diffusion_loss
        return {"loss": loss, "cognitive_loss": cognitive_loss,
                "diffusion_loss": diffusion_loss}

    def language_loss(self, prefix: list[int], continuation: list[int], *,
                      noise_fraction: float = 0.5,
                      seed: int | None = None) -> dict[str, Tensor]:
        """Self-supervised causal + masked reconstruction of authorized text.

        The recurrent objective predicts continuation bytes from the preceding
        text; the denoiser reconstructs ONLY masked positions, conditioned on
        the recurrent prefix state. No assistant answer or external teacher is
        required. The next-byte target is shifted to prevent future leakage.
        """
        if (not 1 <= len(prefix) or not 1 <= len(continuation)
                or len(continuation) > self.config.max_answer
                or len(prefix) + len(continuation) > self.config.max_context
                or any(not 0 <= b < 262 for b in prefix)
                or any(not 0 <= b < 262 for b in continuation)
                or not 0 < noise_fraction <= 1):
            raise ValueError("Invalid language pretraining window")
        device = self.cortex.embedding.weight.device
        raw = torch.tensor([prefix + continuation], device=device,
                           dtype=torch.long)
        # A position predicts the *next* byte; only continuation targets count.
        ar_logits, _ = self.cortex(raw)
        predicted = ar_logits[:, len(prefix) - 1:-1]
        expected = raw[:, len(prefix):]
        recurrent_loss = F.cross_entropy(
            predicted.reshape(-1, self.cortex.config.vocab_size),
            expected.reshape(-1))
        clean = torch.tensor([continuation], device=device, dtype=torch.long)
        generator = torch.Generator(device=device)
        if seed is None:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        generator.manual_seed(seed)
        mask = torch.rand(clean.shape, generator=generator,
                          device=device) < noise_fraction
        if not bool(mask.any().item()):
            mask[0, 0] = True
        noisy = clean.masked_fill(mask, MASK_ID)
        context = torch.tensor([prefix], device=device, dtype=torch.long)
        t = torch.tensor([noise_fraction], device=device)
        diffusion_logits = self.denoise_logits(context, noisy, t)
        diffusion_loss = F.cross_entropy(diffusion_logits[mask], clean[mask])
        return {
            "loss": recurrent_loss + self.config.diffusion_weight * diffusion_loss,
            "cognitive_loss": recurrent_loss,
            "diffusion_loss": diffusion_loss,
        }

    def draft(self, prefix: list[int], *, max_new_tokens: int = 128,
              temperature: float = 0.0, steps: int | None = None) -> list[int]:
        """Iteratively reveal highest-confidence masked bytes, no teacher calls.

        An absorbing-mask schedule reveals additional positions each iteration.
        Positions once committed are not modified. EOS stops the final answer;
        this version is research-grade and does not implement dynamic length.
        """
        if (not prefix or len(prefix) > self.config.max_context
                or not 1 <= max_new_tokens <= self.config.max_answer
                or not 0 <= temperature <= 2):
            raise ValueError("Invalid generation request")
        count = steps if steps is not None else self.config.denoise_steps
        if type(count) is not int or not 1 <= count <= 32:
            raise ValueError("Invalid number of diffusion steps")
        self.eval()
        device = self.cortex.embedding.weight.device
        with torch.inference_mode():
            prompt = torch.tensor([prefix], dtype=torch.long, device=device)
            draft = torch.full((1, max_new_tokens), MASK_ID,
                               device=device, dtype=torch.long)
            cached_context = self.cognitive_context(prompt)
            for step in range(count):
                remaining = draft.eq(MASK_ID)
                n_remain = int(remaining.sum().item())
                if not n_remain:
                    break
                fraction = n_remain / max_new_tokens
                t = torch.tensor([fraction], device=device)
                logits = self.denoise_logits(
                    prompt, draft, t, cached_context=cached_context).clone()
                # Only raw UTF-8 bytes and EOS are legal response tokens.
                logits[:, :, [ByteCodec.PAD, ByteCodec.BOS, ByteCodec.USER,
                              ByteCodec.ASSISTANT, ByteCodec.SYSTEM]] = -torch.inf
                if temperature == 0:
                    predictions = logits.argmax(dim=-1)
                    confidence = F.softmax(logits, dim=-1).gather(
                        -1, predictions.unsqueeze(-1)).squeeze(-1)
                else:
                    probabilities = F.softmax(logits / temperature, dim=-1)
                    predictions = torch.multinomial(
                        probabilities.reshape(-1, probabilities.shape[-1]), 1
                    ).reshape_as(draft)
                    confidence = probabilities.gather(
                        -1, predictions.unsqueeze(-1)).squeeze(-1)
                # Monotonic unmasking; final pass reveals all remaining bytes.
                reveal = max(1, math.ceil(max_new_tokens * (step + 1) / count)
                             - (max_new_tokens - n_remain))
                reveal = min(reveal, n_remain)
                ranked = confidence.masked_fill(~remaining, -1.0)
                indices = ranked.topk(reveal, dim=1).indices
                draft.scatter_(1, indices, predictions.gather(1, indices))
            output: list[int] = []
            for byte in draft[0].tolist():
                if byte == ByteCodec.EOS:
                    break
                if not 0 <= byte < 256:
                    raise RuntimeError("Incomplete diffusion decode")
                output.append(byte)
            return output

    def generate(self, prefix: list[int], *, max_new_tokens=128,
                 temperature=0.0) -> Iterator[int]:
        # Match the native_server generator contract. Denoising is full-window;
        # actual bytes are emitted AFTER iterative refinement completes.
        # Absorbing-mask decoding can place partial/illegal UTF-8 sequences
        # in any slot. Sanitize AFTER refinement but BEFORE streaming. This
        # only guarantees valid wire text, not grammatical/semantic quality.
        decoder = self.inference_decoder
        if decoder == "diffusion":
            raw = bytes(self.draft(prefix, max_new_tokens=max_new_tokens,
                                   temperature=temperature))
        elif decoder == "cortex":
            # The pretrained recurrent head predicts the next byte. This
            # provides a meaningful decoding ablation, NOT fluent chat:
            # source text pretraining did not teach assistant-turn framing.
            raw = bytes(self.cortex.generate(
                prefix, max_new_tokens=max_new_tokens,
                temperature=temperature))
        else:
            raise ValueError("Unknown COG-FU inference decoder")
        repaired = raw.decode("utf-8", errors="replace").replace("\ufffd", "?")
        yield from repaired.encode("utf-8")
