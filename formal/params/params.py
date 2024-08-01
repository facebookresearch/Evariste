# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Any, Set, Dict, Type, TypeVar, get_origin
from argparse import ArgumentParser
from logging import getLogger
from dataclasses import (
    field,
    fields,
    asdict,
    dataclass,
    is_dataclass,
)
import abc
import copy
import argparse
import typeguard
import dataclasses

from pathlib import Path

from evariste import json as json
import yaml


FALSY_STRINGS = {"off", "false", "0"}
TRUTHY_STRINGS = {"on", "true", "1"}

SEPARATOR = "."


logger = getLogger()


class DeepCopyDict(dict):
    def __getitem__(self, item):
        res = super(DeepCopyDict, self).__getitem__(item)
        if callable(res):
            return res()
        return copy.deepcopy(res)

    def __setitem__(self, key, value):
        if key in self:
            raise ValueError(f"{key} already in ConfStore")

        super(DeepCopyDict, self).__setitem__(key, value)


ConfStore = DeepCopyDict()


MISSING: Any = "???"


class NOTSET:
    pass


class MissingArg(Exception):
    pass


class WrongConfName(Exception):
    pass


class WrongConfType(Exception):
    pass


class WrongArgType(Exception):
    pass


class WrongFieldType(Exception):
    pass


class OptionalDataClass(Exception):
    pass


class DefaultDataClassValue(Exception):
    pass


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    elif s == NOTSET:
        return MISSING
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")


def flatten_dict(to_flatten, prefix=""):
    flattened = {}
    for x, y in to_flatten.items():
        if isinstance(y, dict):
            flattened.update(flatten_dict(y, prefix=f"{prefix}{x}{SEPARATOR}"))
        else:
            flattened[f"{prefix}{x}"] = y
    return flattened


def is_optional(some_type):
    return some_type == Optional[some_type]


def get_opt_type(some_type):
    if is_optional(some_type):
        # allowing Optional dataclasses now: we only recurse in the cli if the name appears in the args
        # if is_dataclass(some_type.__args__[0]):
        #     raise OptionalDataClass("Optional dataclasses not supported")
        return some_type.__args__[0]
    return some_type


def NOCLI(default: Any = None):
    return field(default=default, metadata={"NOCLI": True})


def is_nocli(some_field):
    return (
        isinstance(some_field, dataclasses.Field)
        and some_field.metadata.get("NOCLI", False) is True
    )


@dataclass
class Params(abc.ABC):
    def __new__(cls, *args, **kwargs):
        """
        Verify that:
            1) a Params dataclass does not have a field of type Params with a
               default value (as it would result in silent configs bugs)
            2) default_factory arguments build dataclasses of expected types
        """
        for field in fields(cls):
            ftype = get_opt_type(field.type)
            if get_origin(ftype) is dict:
                raise RuntimeError(
                    f"Field {field.name} of class {cls.__name__} "
                    f"is of type {ftype}, which breaks our Params. If you need it, fix Params."
                )
            if is_dataclass(ftype) and isinstance(field.default, Params):
                raise DefaultDataClassValue(
                    f"Field {field.name} of class {cls.__name__} is set to a shared "
                    f"default config. Setting a shared default value is dangerous "
                    f"and can lead to unexpected changes across configs. "
                    f"Use `default_factory` instead."
                )
            if is_dataclass(ftype) and field.default_factory != dataclasses.MISSING:
                try:
                    value = field.default_factory()
                    typeguard.check_type(field.name, value, field.type)
                except TypeError:
                    raise DefaultDataClassValue(
                        f"`default_factory` for field {field.name} of class {cls.__name__} "
                        f"should build objects of type {field.type.__name__}."
                    )
        return super().__new__(cls)

    @classmethod
    def to_cli(
        cls, prefix: str = "", parser: Optional[ArgumentParser] = None
    ) -> ArgumentParser:

        # initialize parser
        if parser is None:
            parser = ArgumentParser(allow_abbrev=False)
            parser.add_argument("--cfg", type=str)

        # for each field in the dataclass
        for field in fields(cls):

            # sanity check / get field name / field type
            if prefix == "":
                assert field.name != "cfg", "'cfg' field is reserved for cli parser"
            fullname = f"{prefix}{field.name}"
            field_type = get_opt_type(field.type)

            # this field is not allowed in the CLI -> nothing to do
            if is_nocli(field):
                pass

            # dataclass -> recursively add arguments to the CLI
            elif is_dataclass(field_type):
                field_type.to_cli(prefix=f"{fullname}.", parser=parser)
                parser.add_argument(f"--{fullname}", type=str, default=NOTSET)

            # standard parameter
            else:
                parser.add_argument(
                    f"--{fullname}",
                    type=bool_flag if field_type == bool else field_type,
                    help="" if field.metadata is None else field.metadata.get("help"),
                    default=NOTSET,
                )

        return parser

    @classmethod
    def from_cli(
        cls,
        param_dict: Dict[str, Any],
        prefix: str = "",
        default_instance: Optional["Params"] = None,
        allow_incomplete: Optional[bool] = False,
    ):
        """
        Converts a flat dot separated dictionary into a class object.
        The dictionary must come from a CLI (`parse_args`) output directly.
        """
        assert default_instance is None or isinstance(default_instance, cls)

        kwargs = {}

        # to crash/or do some warning if some unused args
        used_args: Set[str] = {"cfg"}

        for field in fields(cls):

            fullname = f"{prefix}{field.name}"
            field_type = get_opt_type(field.type)
            assert allow_incomplete or (fullname not in param_dict) == is_nocli(field)
            cli_value = NOTSET
            if fullname in param_dict:
                cli_value = param_dict[fullname]
                used_args |= {k for k in param_dict if k.startswith(fullname)}

            # default value is field is not set in the CLI
            default_value = (
                field.default
                if default_instance is None
                else getattr(default_instance, field.name)
            )
            # this field should not be in the CLI, so we use the default value
            if is_nocli(field):
                kwargs[field.name] = default_value

            # this is a dataclass
            elif is_dataclass(field_type):
                # If optional, default value must be None
                # In this case, we only recurse if some params start with our full name
                must_recurse = any(
                    [
                        x.startswith(fullname) and y != NOTSET
                        for x, y in param_dict.items()
                    ]
                )
                if not must_recurse and is_optional(field.type):
                    # Either default_value is None and nothing else was set in the CLI
                    # Or it might be set in the default instance which is captured by default_value
                    kwargs[field.name] = default_value
                else:
                    # dataclass no specified, use a default value
                    if cli_value == NOTSET:
                        sub_conf = (
                            None
                            if default_value in [MISSING, dataclasses.MISSING]
                            else default_value
                        )

                    # dataclass specified using a named config
                    elif isinstance(cli_value, str) and cli_value != MISSING:
                        if cli_value not in ConfStore:
                            raise WrongConfName(
                                f"Unknown conf key {cli_value} for field {fullname}"
                            )
                        sub_conf = ConfStore[cli_value]

                    # unexpected type (should not be reachable if `param_dict` is the
                    # output of a CLI, as the argument should be of type `str`).
                    else:
                        raise WrongArgType(f"Value for {fullname} should be a string!")

                    # check that the current config has a correct type
                    if sub_conf is not None and not isinstance(sub_conf, field_type):
                        raise WrongConfType(
                            f"Invalid configuration. Provided a configuration of type "
                            f'"{type(sub_conf).__name__}", expected "{field_type.__name__}".'
                        )

                    # if we specified a.b = some_name and a.b.c = 5, we want to overwrite some_name.c
                    kwargs[field.name] = field_type.from_cli(
                        param_dict=param_dict,
                        prefix=f"{fullname}.",
                        default_instance=sub_conf,
                    )

            # this is not a dataclass, with a CLI provided value. try to parse it
            elif cli_value != NOTSET:
                try:
                    kwargs[field.name] = field_type(cli_value)
                except ValueError as e:
                    raise WrongArgType(e)

            # this is not a dataclass, and it does not appear in the CLI
            else:
                kwargs[field.name] = default_value

        # all arguments should be used and not MISSING
        unused_args = {
            k: v
            for k, v in param_dict.items()
            if (k not in used_args) and k.startswith(prefix)
        }
        if unused_args:
            raise RuntimeError(
                f"Some fields in from_cli are unused to instanciate {cls}: {unused_args}"
            )
        for x, y in kwargs.items():
            if y == MISSING:
                raise MissingArg(f"Parameter {prefix}{x} is MISSING")

        return cls(**kwargs)

    @classmethod
    def from_flat(
        cls,
        flat: Dict[str, Any],
        prefix: str = "",
        default_instance: Optional["Params"] = None,
        unused_warning: bool = True,
    ):
        """
        Converts a flat dot separated dictionary into a class object.
        The dictionary must come from a flattened object from the same class,
        and not a CLI.

        NOTE: some issues will happen if a class field is a Union of dataclasses (e.g. DatasetConf),
        as `field_type.from_flat` will not know which class to reload (e.g. MetamathDatasetConf, or
        HolLightDatasetConf in ZMQProverParams.dataset)
        """
        kwargs = {}

        # to crash/or do some warning if some unused args
        used_args: Set[str] = set()

        for field in fields(cls):
            fullname = f"{prefix}{field.name}"
            field_type = get_opt_type(field.type)

            default_value = (
                (
                    field.default_factory()  # type: ignore ## https://github.com/python/mypy/issues/5485
                    if field.default_factory != dataclasses.MISSING  # type: ignore ## https://github.com/python/mypy/issues/5485
                    else field.default
                )
                if default_instance is None
                else getattr(default_instance, field.name)
            )

            if is_dataclass(field_type):
                sub_conf = (
                    None
                    if default_value == MISSING or default_value == dataclasses.MISSING
                    else default_value
                )

                kwargs[field.name] = field_type.from_flat(
                    flat, prefix=f"{fullname}{SEPARATOR}", default_instance=sub_conf
                )
                used_args |= {k for k in flat if k.startswith(f"{fullname}{SEPARATOR}")}
            else:
                try:
                    kwargs[field.name] = flat[fullname]
                    used_args.add(fullname)
                except KeyError:
                    # We only crash if no default is provided
                    if default_value == MISSING or default_value == dataclasses.MISSING:
                        raise MissingArg(
                            f"Parameter {fullname} unspecified and has no default"
                        )
                    else:
                        kwargs[field.name] = default_value

        unused_args = {
            k: v
            for k, v in flat.items()
            if (k not in used_args) and k.startswith(prefix)
        }
        if unused_warning and unused_args:
            logger.warning(
                f"Some fields in flat_dict are unused to instantiate {cls}: {unused_args}"
            )
        for x, y in kwargs.items():
            if y == MISSING:
                raise MissingArg(f"Parameter {prefix}{x} is MISSING")
        return cls(**kwargs)

    @classmethod
    def from_dict(cls: Type["Params"], src: Dict):
        """
        @param src: a nested (or flat) dictionary as exported by dataclass.asdict()
        @return: a cls object with defaults filled in
        """
        flat_dict = flatten_dict(src)
        return cls.from_flat(flat_dict)

    @classmethod
    def from_json(cls: Type["Params"], s: str):
        return cls.from_dict(json.loads(s))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_flat(self):
        return flatten_dict(asdict(self))

    def to_json(self: "Params"):
        return json.dumps(asdict(self), sort_keys=True, indent=4)

    def check_and_mutate_args(self: "Params", avoid: Optional[Set[Type]] = None):
        self._check_and_mutate_args()
        for field in fields(self):
            attr = getattr(self, field.name)
            if isinstance(attr, Params) and (avoid is None or not type(attr) in avoid):
                attr.check_and_mutate_args(avoid=avoid)
            elif avoid is not None and type(attr) in avoid:
                print(f"Not checking {field.name} of type {type(attr)}")

    def get_missing(self) -> Set[str]:
        def traverse(x: "Params", prefix: str, missing: Set[str]):
            for field in fields(x):
                value = getattr(x, field.name)
                if value == MISSING:
                    missing.add(f"{prefix}{field.name}")
                elif is_dataclass(value):
                    traverse(value, f"{prefix}{field.name}.", missing)
            return missing

        return traverse(self, "", set())

    def has_missing(self) -> bool:
        for field in fields(self):
            ftype = get_opt_type(field.type)
            value = getattr(self, field.name)
            if value == MISSING:
                return True
            elif is_dataclass(ftype) and value.has_missing():
                return True
        return False

    def check_type(self, prefix: str = "") -> None:
        for field in fields(self):
            ftype = get_opt_type(field.type)
            value = getattr(self, field.name)
            fullname = f"{prefix}{field.name}"
            try:
                typeguard.check_type(field.name, value, field.type)
            except TypeError:
                raise WrongFieldType(f"Wrong field type for {fullname}: {value}")
            if is_dataclass(ftype) and value is not None:
                value.check_type(f"{fullname}.")

    def _check_and_mutate_args(self):
        pass


T = TypeVar("T", bound=Params)

import itertools
from typing import List, Tuple


def parse_json(grid: Dict, script: str) -> List[Tuple[Dict, Dict, str]]:

    subsweep_keys = [k for k in grid.keys() if k.startswith("SUBSWEEP")]
    if len(subsweep_keys) > 0:
        commands = []
        subsweep_grids = itertools.product(*[grid[k].values() for k in subsweep_keys])
        for k in subsweep_keys:
            del grid[k]
        for subsweep in subsweep_grids:
            new_grid = copy.deepcopy(grid)
            for v1 in subsweep:
                for k2, v2 in v1.items():
                    new_grid[k2] = v2
            # TODO: fix something here if we want swept_params to include params from subsweep
            commands.extend(parse_json(new_grid, script))
        return commands

    perms = list(itertools.product(*grid.values()))
    commands = []

    for p in perms:
        argstr = ""
        grid_params = {}
        swept_params = {}
        for i, k in enumerate(grid.keys()):
            if p[i] is None:  # to avoid setting optional parameters
                continue
            grid_params[k] = p[i]
            if len(grid[k]) > 1:
                swept_params[k] = p[i]
            if isinstance(p[i], (bool, int, float)):
                v = str(p[i])
            elif type(p[i]) is list or type(p[i]) is dict:
                v = f"'{json.dumps(p[i])}'"
            else:
                assert '"' not in p[i]
                v = f'"{p[i]}"'
            argstr += f" --{str(k)} {v}"
        commands.append((grid_params, swept_params, argstr))

    return commands


def cfg_from_cli(
    base_config: Optional[T] = None, schema: Optional[Type[T]] = None
) -> T:

    assert (base_config is None) == (schema is not None)
    if base_config is not None:
        schema = base_config.__class__

    assert schema is not None
    param_dict = vars(schema.to_cli().parse_args())
    if param_dict["cfg"] is not None:
        cfg = param_dict.pop("cfg")
        if cfg in ConfStore:
            base_config = ConfStore[cfg]
            assert isinstance(base_config, schema)
        elif Path(cfg).exists():
            with open(Path(cfg), "r") as f:
                content = f.read()
            if cfg.endswith(".yaml"):
                parsed = yaml.full_load(content)
            else:
                parsed = json.loads(content)
            all_parses = parse_json(parsed, "")
            if len(all_parses) > 1:
                print(f"\033[91m WARNING SELECTING ONE GRID\033[0m")
            command = all_parses[0][-1]
            print(command)
            raise RuntimeError
        else:
            raise NotImplementedError(
                f"Unknown cfg {cfg} neither a file nore an entry in confstore"
            )
    ret = schema.from_cli(param_dict=param_dict, default_instance=base_config)
    assert isinstance(ret, schema)
    return ret
