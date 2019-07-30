from .cli import parse_args


def main() -> None:
    """CLI entry point of tmexp."""
    handler, kw_args = parse_args()
    handler(**kw_args)


if __name__ == "__main__":
    main()
