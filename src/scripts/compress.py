if __name__ == "__main__":
    from src.utils.utils import make_default_argparse, make_cfg
    from src.models.utils import make_model

    parser = make_default_argparse()
    parser.add_argument("-s", "--source", type=str, required=True)
    parser.add_argument("-t", "--target", type=str, required=True)
    args = parser.parse_args()
    cfg = make_cfg(args)

    model = make_model(cfg)
    model.compress(source_name=args.source, target_name=args.target)
