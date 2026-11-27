"""Composable, input-independent GF(2) rules; no nonlinear inference claim."""
from dataclasses import dataclass
from typing import Iterable

from .xor_ciu import XorTransition, _bits


@dataclass(frozen=True)
class GF2Solution:
    """Exact solution set for a packed GF(2) linear system.

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


@dataclass(frozen=True)
class GF2ConstraintSystem:
    """A bounded set of simultaneous affine bit constraints over one input word.

    Rows may outnumber the input width, so independent components can be
    combined horizontally before solving. Each row is a packed coefficient
    vector and each RHS entry is one bit.
    """
    width: int
    rows: tuple[int, ...]
    rhs: tuple[int, ...]

    def __post_init__(self):
        if type(self.width) is not int or not 1 <= self.width <= 256:
            raise ValueError("constraint width must be 1..256")
        if type(self.rows) is not tuple or type(self.rhs) is not tuple:
            raise TypeError("constraint rows and right sides must be immutable tuples")
        if len(self.rows) != len(self.rhs) or len(self.rows) > 65536:
            raise ValueError("invalid constraint count")
        for row in self.rows:
            _bits(row, self.width)
        if any(type(value) is not int or value not in (0, 1) for value in self.rhs):
            raise ValueError("right sides must be binary")

    @classmethod
    def from_affine(cls, constraints):
        """Stack (transition, target) observations that share the same input."""
        items = tuple(constraints)
        if not 1 <= len(items) <= 256:
            raise ValueError("expected 1..256 affine constraints")
        if any(type(item) not in (tuple, list) or len(item) != 2
               for item in items):
            raise TypeError("each constraint must be a transition/target pair")
        first, target = items[0]
        if not isinstance(first, AffineTransition):
            raise TypeError("AffineTransition required")
        width = first.width
        rows = []
        values = []
        for transition, target in items:
            if not isinstance(transition, AffineTransition) or transition.width != width:
                raise ValueError("incompatible affine constraint widths")
            _bits(target, width)
            rows.extend(_matrix_rows(width, transition.columns))
            bound = target ^ transition.offset
            values.extend((bound >> bit) & 1 for bit in range(width))
        return cls(width, tuple(rows), tuple(values))

    def combine(self, other):
        """Add another component's constraints over the same unknown word."""
        if not isinstance(other, GF2ConstraintSystem) or other.width != self.width:
            raise ValueError("incompatible constraint systems")
        if len(self.rows) + len(other.rows) > 65536:
            raise ValueError("combined constraint limit exceeded")
        return GF2ConstraintSystem(self.width, self.rows + other.rows,
                                   self.rhs + other.rhs)

    def solve(self) -> GF2Solution | None:
        """Return a complete exact solution space, or None if inconsistent."""
        return _solve_rows(self.width, self.rows, self.rhs)

    def complete(self, known_mask: int, known_values: int) -> GF2Solution | None:
        """Complete a partially known input while preserving every known bit."""
        _bits(known_mask, self.width)
        _bits(known_values, self.width)
        if known_values & ~known_mask:
            raise ValueError("known values contain bits outside known mask")
        rows = list(self.rows)
        rhs = list(self.rhs)
        for bit in range(self.width):
            if known_mask & (1 << bit):
                rows.append(1 << bit)
                rhs.append((known_values >> bit) & 1)
        if len(rows) > 65536:
            raise ValueError("completed constraint limit exceeded")
        return _solve_rows(self.width, tuple(rows), tuple(rhs))

    def verifies(self, solution: GF2Solution) -> bool:
        """Check the particular solution and every nullspace direction."""
        if not isinstance(solution, GF2Solution) or solution.width != self.width:
            return False
        system_rank = _solve_rows(self.width, self.rows,
                                  (0,) * len(self.rows)).rank
        if solution.rank != system_rank:
            return False
        if any(((row & solution.particular).bit_count() & 1) != value
               for row, value in zip(self.rows, self.rhs)):
            return False
        return all((row & vector).bit_count() % 2 == 0
                   for row in self.rows for vector in solution.nullspace)


def _matrix_rows(width: int, columns: tuple[int, ...]) -> tuple[int, ...]:
    rows = []
    for output_bit in range(width):
        output_mask = 1 << output_bit
        coefficients = 0
        for input_bit, column in enumerate(columns):
            if column & output_mask:
                coefficients |= 1 << input_bit
        rows.append(coefficients)
    return tuple(rows)


def _solve_rows(width: int, rows: tuple[int, ...],
                rhs: tuple[int, ...]) -> GF2Solution | None:
    """Solve a rectangular packed GF(2) system by exact Gauss-Jordan reduction."""
    work = list(zip(rows, rhs))
    row_count = len(work)
    pivot_columns = []
    pivot_row = 0
    for column in range(width):
        column_mask = 1 << column
        selected = next((row for row in range(pivot_row, row_count)
                         if work[row][0] & column_mask), None)
        if selected is None:
            continue
        work[pivot_row], work[selected] = work[selected], work[pivot_row]
        pivot_coefficients, pivot_value = work[pivot_row]
        for row in range(row_count):
            if row != pivot_row and work[row][0] & column_mask:
                coefficients, value = work[row]
                work[row] = (coefficients ^ pivot_coefficients,
                             value ^ pivot_value)
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == row_count:
            break

    if any(coefficients == 0 and value for coefficients, value in work):
        return None

    particular = 0
    for row, column in enumerate(pivot_columns):
        if work[row][1]:
            particular |= 1 << column

    pivot_set = set(pivot_columns)
    nullspace = []
    for free_column in range(width):
        if free_column in pivot_set:
            continue
        vector = 1 << free_column
        free_mask = 1 << free_column
        for row, pivot_column in enumerate(pivot_columns):
            if work[row][0] & free_mask:
                vector |= 1 << pivot_column
        nullspace.append(vector)

    return GF2Solution(width, particular, tuple(nullspace), pivot_row)


def _solve_columns(width: int, columns: tuple[int, ...], rhs: int) -> GF2Solution | None:
    values = tuple((rhs >> bit) & 1 for bit in range(width))
    return _solve_rows(width, _matrix_rows(width, columns), values)

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

    def constraints(self, target: int) -> GF2ConstraintSystem:
        """Represent apply(x)==target as one exact GF(2) constraint system."""
        _bits(target, self.width)
        rows = _matrix_rows(self.width, self.columns)
        bound = target ^ self.offset
        rhs = tuple((bound >> bit) & 1 for bit in range(self.width))
        return GF2ConstraintSystem(self.width, rows, rhs)

    def solve(self, target: int) -> GF2Solution | None:
        """Return every x satisfying apply(x)==target, or None if impossible."""
        return self.constraints(target).solve()

    def solve_xor_fixed_point(self, rhs: int) -> GF2Solution | None:
        """Solve x = apply(x) XOR rhs over GF(2), including singular systems."""
        _bits(rhs, self.width)
        difference_columns = tuple((1 << bit) ^ column
                                   for bit, column in enumerate(self.columns))
        rows = _matrix_rows(self.width, difference_columns)
        bound = rhs ^ self.offset
        values = tuple((bound >> bit) & 1 for bit in range(self.width))
        return GF2ConstraintSystem(self.width, rows, values).solve()

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


def compose_affine_chain(transitions):
    """Compose ordered XOR/affine transitions vertically into one exact map."""
    items = tuple(transitions)
    if not 1 <= len(items) <= 4096:
        raise ValueError("expected 1..4096 transitions")

    def affine(item):
        if isinstance(item, AffineTransition):
            return item
        if isinstance(item, XorTransition):
            return AffineTransition.from_xor(item)
        raise TypeError("XorTransition or AffineTransition required")

    result = affine(items[0])
    for item in items[1:]:
        result = result.then(affine(item))
    return result


def solve_affine_constraints(constraints) -> GF2Solution | None:
    """Solve horizontal observations that constrain the same unknown input."""
    return GF2ConstraintSystem.from_affine(constraints).solve()


def solve_affine_chain(transitions, target: int) -> GF2Solution | None:
    """Solve the complete preimage through an ordered vertical transition chain."""
    return compose_affine_chain(transitions).solve(target)
