from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    xs_prev = list(vals).copy()
    xs_prev[arg] -= epsilon
    xs_next = list(vals).copy()
    xs_next[arg] += epsilon
    return (f(*xs_next) - f(*xs_prev)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    g = dict()
    vars = dict()
    vars[variable.unique_id] = variable
    new = [variable]
    i = 0
    while i < len(new):
        v = new[i]
        for par in v.parents:
            if par.is_constant():
                continue
            if par.unique_id not in g:
                g[par.unique_id] = []
            g[par.unique_id].append(v.unique_id)
            if par.unique_id not in vars:
                vars[par.unique_id] = par
                new.append(par)
        i += 1
    ord = []
    used = set()
    leaves = []
    for i in vars.keys():
        if vars[i].is_leaf():
            leaves.append(i)
    def dfs(v):
        if v in used:
            return
        used.add(v)
        for u in g.get(v, []):
            dfs(u)
        ord.append(v)
    for i in leaves:
        dfs(i)
    result = []
    for i in ord:
        result.append(vars[i])
    return result


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    if variable.is_constant():
        return
    order = topological_sort(variable)
    d = dict()
    d[variable.unique_id] = deriv
    for var in order:
        if var.is_leaf():
            var.accumulate_derivative(d[var.unique_id])
        else:
            for par, deriv in var.chain_rule(d[var.unique_id]):
                if not par.is_constant():
                    if par.unique_id not in d:
                        d[par.unique_id] = 0
                    d[par.unique_id] += deriv


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
