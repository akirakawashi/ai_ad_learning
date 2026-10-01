# AGENTS.md — AI Ad learning

## Rules

1. **Every function and method has a docstring that says what it does, what it takes and what it
   returns.** Google style: a short Russian summary, then `Args:` with every parameter except
   `self` and `cls`, and `Returns:` (`Yields:` for a generator) with the result. Section names stay
   English so editors and linters parse them; the text under them is Russian. A section is left out
   only when it would be empty: no parameters, or the function returns `None`. `tests/` is exempt.
