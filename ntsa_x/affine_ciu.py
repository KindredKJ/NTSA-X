"""Composable, input-independent GF(2) rules; no nonlinear inference claim."""
from dataclasses import dataclass
from typing import Iterable

from .xor_ciu import XorTransition, _bits


@dataclass(frozen=True)
class GF2Solution:
    """Exact solution set for a square packed GF(2) linear system.

    Every solution is the particular value XOR a subset of the independent
    vectors in nullspace. Rank is the coefficient-matrix rank.
    """
    width: int
    particular: int
    nullspace: tuple[int, ...]
    rank: int

    def __post_init__(self):
        if type(self.width) is not int or not 1 <= self.width <= 256:
            raise ValueError("solution width must be 1..256")
        if type(self.rank) is not int or not 0 <= self.rank <= self.width:
            raise ValueError("invalid matrix rank")
        _bits(self.particular, self.width)
        if type(self.nullspace) is not tuple or len(self.nullspace) != self.width - self.rank:
            raise ValueError("invalid nullspace basis")
        basis = {}
        for vector in self.nullspace:
            _bits(vector, self.width)
            if vector == 0:
                raise ValueError("nullspace vectors must be nonzero")
            reduced = vector
            while reduced:
                pivot = reduced.bit_length() - 1
                if pivot not in basis:
                    basis[pivot] = reduced
                    break
                reduced ^= basis[pivot]
            if reduced == 0:
                raise ValueError("nullspace vectors must be independent")

    @property
    def unique(self) -> bool:
        return not self.nullspace

    @property
    def solution_count(self) -> int:
        return 1 << len(self.nullspace)

    def solutions(self, *, max_solutions: int = 65536) -> tuple[int, ...]:
        """Enumerate the complete solution set under an explicit output bound."""
        if type(max_solutions) is not int or not 1 <= max_solutions <= 65536:
            raise ValueError("invalid solution enumeration limit")
        count = self.solution_count
        if count > max_solutions:
            raise ValueError("solution set exceeds enumeration limit")
        results = []
        for selector in range(count):
            value = self.particular
            for index, vector in enumerate(self.nullspace):
                if selector & (1 << index):
                    value ^= vector
            results.append(value)
        return tuple(results)


def _solve_columns(width: int, columns: tuple[int, ...], rhs: int) -> GF2Solution | None:
    """Solve A*x=rhs using packed rows and full GF(2) elimination."""
    rows = []
    for output_bit in range(width):
        output_mask = 1 << output_bit
        coefficients = 0
        for input_bit, column in enumerate(columns):
            if column & output_mask:
                coefficients |= 1 << input_bit
        rows.append((coefficients, (rhs >> output_bit) & 1))

    pivot_columns = []
    pivot_row = 0
    for column in range(width):
        column_mask = 1 << column
        selected = next((row for row in range(pivot_row, width)
                         if rows[row][0] & column_mask), None)
        if selected is None:
            continue
        rows[pivot_row], rows[selected] = rows[selected], rows[pivot_row]
        pivot_coefficients, pivot_value = rows[pivot_row]
        for row in range(width):
            if row != pivot_row and rows[row][0] & column_mask:
                coefficients, value = rows[row]
                rows[row] = (coefficients ^ pivot_coefficients,
                             value ^ pivot_value)
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == width:
            break

    if any(coefficients == 0 and value for coefficients, value in rows):
        return None

    particular = 0
    for row, column in enumerate(pivot_columns):
        if rows[row][1]:
            particular |= 1 << column

    pivot_set = set(pivot_columns)
    nullspace = []
    for free_column in range(width):
        if free_column in pivot_set:
            continue
        vector = 1 << free_column
        free_mask = 1 << free_column
        for row, pivot_column in enumerate(pivot_columns):
            if rows[row][0] & free_mask:
                vector |= 1 << pivot_column
        nullspace.append(vector)

    return GF2Solution(width, particular, tuple(nullspace), pivot_row)


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

    def solve(self, target: int) -> GF2Solution | None:
        """Return every x satisfying apply(x)==target, or None if impossible."""
        _bits(target, self.width)
        return _solve_columns(self.width, self.columns, target ^ self.offset)

    def solve_xor_fixed_point(self, rhs: int) -> GF2Solution | None:
        """Solve x = apply(x) XOR rhs over GF(2), including singular systems."""
        _bits(rhs, self.width)
        difference_columns = tuple((1 << bit) ^ column
                                   for bit, column in enumerate(self.columns))
        return _solve_columns(self.width, difference_columns,
                              rhs ^ self.offset)

    @property
    def rank(self) -> int:
        result = _solve_columns(self.width, self.columns, 0)
        return result.rank

    @property
    def is_invertible(self) -> bool:
        return self.rank == self.width

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
