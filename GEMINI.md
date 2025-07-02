# Repository Rules
1. Commit code frequently with small changes so that experiments can easily be undone
2. Code should only be commited in a working state; do not commit a partial feature
3. Before committing code, you are required to assess and address the impact on documentation. If documentation needs to be updated, do so. If no documentation needs to be updated, you must explicitly state that and provide a brief justification *before* proceeding with the commit.
4. All tests must pass before committing code. Run the test suite and ensure there are no failures before proceeding with a commit.
5. When running any solver, you MUST apply a time limit using the --time_limit argument. Default to a 30-second limit. You may use your judgment to propose a longer time limit if the complexity of the puzzle appears to warrant it, but you must state your proposed time limit and reasoning to the user before running.
6. When writing commit messages, use single quotes so that backticks do not get interpreted by bash.
