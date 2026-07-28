from .pipeline import parse_test_args, run_test


def main() -> None:
    run_test(parse_test_args())


if __name__ == "__main__":
    main()
