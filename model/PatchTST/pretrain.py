from .pipeline import parse_pretrain_args, run_pretrain


def main() -> None:
    run_pretrain(parse_pretrain_args())


if __name__ == "__main__":
    main()
