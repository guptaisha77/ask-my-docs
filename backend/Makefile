# Makefile — shortcuts for common dev tasks.
# Run with: make lint, make format, make check

format:
	ruff format app/

lint:
	ruff check --fix app/

# "check" does both, the way CI will: format then lint.
check: format lint

# A read-only version that fixes nothing — safe to run anytime.
verify:
	ruff format --check app/
	ruff check app/