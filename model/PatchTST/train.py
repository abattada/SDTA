from .pipeline import parse_train_args, run_train


def main() -> None:
    run_train(parse_train_args())


if __name__ == "__main__":
    main()
