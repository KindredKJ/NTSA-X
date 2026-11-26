"""Composable, input-independent GF(2) rules; no nonlinear inference claim."""
from dataclasses import dataclass
from typing import Iterable

from .xor_ciu import XorTransition, _bits


@dataclass(frozen=True)
class AffineTransition:
    """Columns of a GF(2) matrix and a fixed translation, all packed integers."""
    width: int
    columns: tuple[int, ...]
    offset: int = 0
    count: int = 0

    def __post_init__(self):
        if type(self.width) is not int or not 1 <= self.width <= 256:
            raise ValueError("affine width must be 1..256")
        if type(self.columns) is not tuple or len(self.columns) != self.width:
            raise ValueError("one immutable column per input bit required")
        for c in self.columns:
            _bits(c, self.width)
        _bits(self.offset, self.width)
        if type(self.count) is not int or self.count < 0:
            raise ValueError("invalid operation count")

    @classmethod
    def identity(cls, width: int):
        if type(width) is not int or not 1 <= width <= 256:
            raise ValueError("affine width must be 1..256")
        return cls(width, tuple(1 << i for i in range(width)))

    @classmethod
    def from_xor(cls, transition: XorTransition):
        base = cls.identity(transition.width)
        return cls(base.width, base.columns, transition.mask, transition.count)

    def linear(self, state: int) -> int:
        _bits(state, self.width)
        result = 0
        while state:
            bit = state & -state
            result ^= self.columns[bit.bit_length() - 1]
            state ^= bit
        return result

    def apply(self, state: int) -> int:
        return self.linear(state) ^ self.offset

    def then(self, other: 'AffineTransition') -> 'AffineTransition':
        if not isinstance(other, AffineTransition) or self.width != other.width:
            raise ValueError("incompatible affine transformations")
        return AffineTransition(self.width,
                                tuple(other.linear(c) for c in self.columns),
                                other.apply(self.offset), self.count + other.count)

    @classmethod
    def compile(cls, width: int, instructions: Iterable[tuple[str, int]]):
        """Compile explicit affine operations. Never infer arbitrary black boxes."""
        base = cls.identity(width)
        columns, offset, count = base.columns, 0, 0
        limit = (1 << width) - 1
        for op, arg in instructions:
            if op == 'xor':
                offset ^= _bits(arg, width)
            elif op in ('rotl', 'xorshl', 'xorshr'):
                if type(arg) is not int or not 0 < arg < width:
                    raise ValueError("shift must be strictly inside width")
                if op == 'rotl':
                    def linear(v):
                        return ((v << arg) | (v >> (width - arg))) & limit
                elif op == 'xorshl':
                    def linear(v):
                        return v ^ ((v << arg) & limit)
                else:
                    def linear(v):
                        return v ^ (v >> arg)
                columns = tuple(linear(c) for c in columns)
                offset = linear(offset)
            else:
                raise ValueError("unsupported or non-affine instruction")
            count += 1
        return cls(width, columns, offset, count)
