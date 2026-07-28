from .pipeline import parse_test_all_args, run_test_all


def main() -> None:
    run_test_all(parse_test_all_args())


if __name__ == "__main__":
    main()
