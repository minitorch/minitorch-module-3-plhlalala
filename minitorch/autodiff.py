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
    # TODO: Implement for Task 1.1.
    vals_diff = list(vals).copy()
    vals_diff[arg] += epsilon
    f1 = f(*vals_diff)
    vals_diff[arg] -= 2 * epsilon
    f2 = f(*vals_diff)
    return (f1 - f2) / (2 * epsilon)


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
    # TODO: Implement for Task 1.4.
    PermanentMarked = []
    TemporaryMarked = []
    result = []

    def visit(n):
        if n.is_constant():
            return
        if n.unique_id in PermanentMarked:
            return
        elif n.unique_id in TemporaryMarked:
            raise (RuntimeError("Not a DAG"))

        TemporaryMarked.append(n.unique_id)

        if n.is_leaf():
            pass
        else:
            for input in n.parents:
                visit(input)
        TemporaryMarked.remove(n.unique_id)
        PermanentMarked.append(n.unique_id)
        result.insert(0, n)

    visit(variable)

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
    # TODO: Implement for Task 1.4.
    nodes = topological_sort(variable)
    dict = {}
    dict[variable.unique_id] = deriv
    for node in nodes:
        deriv = dict[node.unique_id]
        if node.is_leaf():
            node.accumulate_derivative(dict[node.unique_id])
        else:
            for p, d in node.chain_rule(deriv):
                if p.unique_id in dict:
                    dict[p.unique_id] += d
                else:
                    dict[p.unique_id] = d
    return


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
