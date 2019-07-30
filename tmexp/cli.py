from argparse import ArgumentParser
from collections import OrderedDict
from logging import _nameToLevel as logging_name_to_level
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

from .utils import check_create_default, SUPPORTED_LANGUAGES


class CommandDescription(NamedTuple):
    help: str
    parser_definer: Callable[[ArgumentParser], None]
    handler: Callable[..., Any]


_commands: Dict[str, CommandDescription] = OrderedDict()


def register_command(
    *,
    parser_definer: Callable[[ArgumentParser], None],
    name: Optional[str] = None,
    help: Optional[str] = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def wrapper(handler: Callable[..., Any]) -> Callable[..., Any]:
        if help is None:
            final_help = handler.__doc__.strip()
            if not final_help:
                raise RuntimeError(
                    "Could not register the command %s, please provide a help "
                    "or a docstring." % name
                )
            final_help = final_help.splitlines()[0].strip()
        else:
            final_help = help
        final_name = handler.__name__.replace("_", "-") if name is None else name
        _commands[final_name] = CommandDescription(
            help=final_help, parser_definer=parser_definer, handler=handler
        )
        return handler

    return wrapper


def _define_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=logging_name_to_level,
        help="Logging verbosity.",
    )
    subparsers = parser.add_subparsers(help="Commands")
    for name, (help, parser_definer, handler) in _commands.items():
        subparser = subparsers.add_parser(name, help=help)
        subparser.set_defaults(handler=handler)
        parser_definer(subparser)
    return parser


def parse_args() -> Tuple[Callable[..., Any], Dict[str, Any]]:
    parser = _define_parser()
    args = parser.parse_args()
    args.log_level = logging_name_to_level[args.log_level]
    handler = args.handler
    delattr(args, "handler")
    return handler, vars(args)


class CLIBuilder:
    def __init__(self, parser: ArgumentParser) -> None:
        self.parser = parser

    def add_lang_args(self) -> None:
        lang_group = self.parser.add_mutually_exclusive_group()
        lang_group.add_argument(
            "--select-langs",
            help="To select a perticular set of languages, defaults to all.",
            nargs="*",
            dest="langs",
            choices=SUPPORTED_LANGUAGES,
        )
        lang_group.add_argument(
            "--exclude-langs",
            help="To exclude a perticular set of languages, defaults to none.",
            nargs="*",
            choices=SUPPORTED_LANGUAGES,
        )

    def add_feature_arg(self) -> None:
        from .preprocess import COMMENTS, IDENTIFIERS, LITERALS

        self.parser.add_argument(
            "--features",
            help="To select which tokens to use as words, defaults to all.",
            nargs="*",
            choices=[COMMENTS, IDENTIFIERS, LITERALS],
            default=[COMMENTS, IDENTIFIERS, LITERALS],
        )

    def add_force_arg(self) -> None:
        self.parser.add_argument(
            "-f",
            "--force",
            help="Delete and replace existing output(s).",
            action="store_true",
        )

    def add_dataset_arg(self, required: bool) -> None:
        help = "Name of the dataset created by `preprocess`%s."
        if required:
            self._add_required("--dataset-name", help % "")
        else:
            self._add_default(
                "--dataset-name", help % ", defaults to '%(default)s'", "dataset"
            )

    def add_bow_arg(self, required: bool) -> None:
        help = "Name of the BoW created by `create_bow`%s."
        if required:
            self._add_required("--bow-name", help % "")
        else:
            self._add_default("--bow-name", help % ", defaults to '%(default)s'", "bow")

    def add_experiment_arg(self, required: bool) -> None:
        help = "Name of the experiment created by `train_$`%s."
        if required:
            self._add_required("--exp-name", help % "")
        else:
            self._add_default(
                "--exp-name", help % ", defaults to '%(default)s'", "experiment"
            )

    def _add_default(self, flag: str, help: str, out_type: str) -> None:
        self.parser.add_argument(
            flag, help=help, default=check_create_default(out_type)
        )

    def _add_required(self, flag: str, help: str) -> None:
        self.parser.add_argument(flag, help=help, required=True)
