from .classify import classify_problem, CATEGORIES
from .base_solver import BaseSolver, SolverResult
from .bit_manipulation import BitManipulationSolver
from .cipher import CipherSolver
from .numeral import NumeralSolver
from .unit_conversion import UnitConversionSolver
from .gravity import GravitySolver
from .equation_transform import EquationTransformSolver

_SOLVERS = {
    "bit_manipulation": BitManipulationSolver(),
    "cipher": CipherSolver(),
    "numeral": NumeralSolver(),
    "unit_conversion": UnitConversionSolver(),
    "gravity": GravitySolver(),
    "equation": EquationTransformSolver(),
}


def solve(prompt: str) -> "SolverResult":
    """Classify and solve a problem. Returns a SolverResult."""
    category = classify_problem(prompt)
    solver = _SOLVERS.get(category)
    if solver is None:
        from .base_solver import SolverResult
        return SolverResult(answer=None, category=category, confidence=0.0, reasoning="No solver for category")
    return solver.solve(prompt)
