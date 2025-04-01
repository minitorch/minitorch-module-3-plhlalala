"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(x: float, y: float) -> float:
    return float(x * y)


def id(x: float) -> float:
    return float(x)


def add(x: float, y: float) -> float:
    return float(x + y)


def neg(x: float) -> float:
    return float(-1.0 * x)


def lt(x: float, y: float) -> float:
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    return 1.0 if abs(x - y) < 1e-2 else 0.0


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    return x if x > 0 else 0.0  # 不能用max，会超出递归深度


def log(x: float) -> float:
    return math.log(x)


def exp(x: float) -> float:
    return math.exp(x)


def log_back(x: float, y: float) -> float:
    return 1.0 / x * y


def inv(x: float) -> float:
    return 1.0 / x


def inv_back(x: float, y: float) -> float:
    return -1.0 / (x**2) * y


def relu_back(x: float, y: float) -> float:
    return y if x > 0 else 0.0


def sigmoid_back(x: float, y: float) -> float:
    sig = 0
    if x >= 0:
        sig =  1.0 / (1.0 + math.exp(-x))
    else:
        sig = math.exp(x) / (1.0 + math.exp(x))
    return sig * (1.0 - sig) * y


def exp_back(x: float, y: float) -> float:
    return math.exp(x) * y

# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    def process(data: Iterable[float]) -> Iterable[float]:
        return [fn(x) for x in data]

    return process


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    def process(data1: Iterable[float], data2: Iterable[float]) -> Iterable[float]:
        return [fn(x, y) for x, y in zip(data1, data2)]

    return process


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    def process(data: Iterable[float]) -> float:
        result = start
        for x in data:
            result = fn(result, x)
        return result

    return process


def negList(data: Iterable[float]) -> Iterable[float]:
    return map(neg)(data)


def addLists(data1: Iterable[float], data2: Iterable[float]) -> Iterable[float]:
    return zipWith(add)(data1, data2)


def sum(data: Iterable[float]) -> float:
    return reduce(add, 0)(data)


def prod(data: Iterable[float]) -> float:
    return reduce(mul, 1)(data)
