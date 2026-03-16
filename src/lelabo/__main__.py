"""Allow ``python -m lelabo`` to execute the LeLabo CLI."""

from .cli.main import main


if __name__ == "__main__":
    raise SystemExit(main())
