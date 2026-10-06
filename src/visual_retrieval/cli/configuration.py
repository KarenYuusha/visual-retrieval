"""Optional YAML defaults with explicit CLI values taking precedence."""
from pathlib import Path


def parse_configured(parser, argv=None):
    parser.add_argument('--config', type=Path, help='Optional model profile YAML; CLI arguments override its values.')
    known, _ = parser.parse_known_args(argv)
    if known.config:
        import yaml
        values = yaml.safe_load(known.config.read_text(encoding='utf-8'))
        if not isinstance(values,dict):
            parser.error('Configuration must be a YAML mapping.')
        actions = {a.dest:a for a in parser._actions}
        for key,value in values.items():
            if key not in actions or key in ('help','config'):
                parser.error(f'Unknown configuration key {key!r}.')
            action=actions[key]
            sequence=value if isinstance(value,list) else [value]
            if action.choices and any(v not in action.choices for v in sequence):
                parser.error(f'Invalid value for {key}: {value!r}.')
            if action.type and value is not None:
                values[key] = [action.type(v) for v in value] if isinstance(value,list) else action.type(value)
        parser.set_defaults(**values)
    return parser.parse_args(argv)
