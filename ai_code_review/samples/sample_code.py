"""
sample_code.py — Demo file for the AI Code Review Assistant.

This file intentionally contains a variety of code quality issues, potential
bugs, and style problems so that the AI reviewer has something meaningful
to analyse during a demo.

Upload this file through the Streamlit UI and click "Analyze Code" to see
the AI review in action.
"""

import math
import os
import sys

# ── Module-level constants (magic numbers — should be named) ────────────────
RATE = 0.05
MAX = 100


# ── 1. Missing type hints, no docstring, potential ZeroDivisionError ────────
def calculate_discount(price, discount):
    result = price - (price * discount / 100)
    return result


def divide(a, b):
    # No guard for b == 0
    return a / b


# ── 2. Inefficient loop — could use sum() / list comprehension ──────────────
def sum_of_squares(numbers):
    """Return the sum of squares of all numbers in the list."""
    total = 0
    squares = []
    for n in numbers:
        squares.append(n * n)
    for s in squares:
        total = total + s
    return total


# ── 3. Poor naming, deeply nested logic, no early return ────────────────────
def p(x, y, z):
    if x > 0:
        if y > 0:
            if z > 0:
                return x + y + z
            else:
                return x + y
        else:
            return x
    else:
        return 0


# ── 4. Class with several issues ─────────────────────────────────────────────
class userAccount:                          # Should be UserAccount (PascalCase)
    def __init__(self, name, balance):
        self.Name = name                    # Should be snake_case: self.name
        self.Balance = balance              # Should be snake_case: self.balance
        self.Transactions = []              # Should be snake_case

    def Deposit(self, amount):             # Method name should be snake_case
        if amount > 0:
            self.Balance += amount
        self.Transactions.append(("deposit", amount))

    def Withdraw(self, amount):            # No check for insufficient funds
        self.Balance -= amount
        self.Transactions.append(("withdraw", amount))

    def getBalance(self):                  # Should be get_balance (snake_case)
        return self.Balance

    def print_history(self):
        for t in self.Transactions:
            print(t[0], t[1])              # Should use f-string or format


# ── 5. File I/O with no error handling ───────────────────────────────────────
def read_config(filepath):
    f = open(filepath)                     # Not using context manager (with)
    data = f.read()
    f.close()
    return data


# ── 6. Mutable default argument (classic Python bug) ─────────────────────────
def append_item(item, container=[]):       # Mutable default arg is a bug
    container.append(item)
    return container


# ── 7. Unused imports and variables ──────────────────────────────────────────
def unused_example():
    unused_var = 42                        # Variable defined but never used
    result = math.sqrt(16)
    # sys and os are imported at the top but never used in this file
    return result


# ── 8. Recursive function with no base-case guard for negative input ─────────
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)            # Fails for negative n (infinite recursion)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Sum of squares:", sum_of_squares([1, 2, 3, 4, 5]))
    print("Discount:", calculate_discount(200, 10))

    acc = userAccount("Alice", 1000)
    acc.Deposit(500)
    acc.Withdraw(200)
    print("Balance:", acc.getBalance())
    acc.print_history()
