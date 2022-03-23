import json
import sys

def parse_with_config(parser):
    args = parser.parse_args()
    if args.config_ttml_paths is not None:
        config_args = json.load(open(args.config_ttml_paths))
        override_keys = {arg[2:].split('=')[0] for arg in sys.argv[1:]
                         if arg.startswith('--')}
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
    del args.config_ttml_paths
    return args