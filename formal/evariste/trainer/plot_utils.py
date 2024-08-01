# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Tuple, List, Set, Dict, Callable
from collections import defaultdict, OrderedDict
from matplotlib import pyplot as plt
import os
import mpld3
import numpy as np

# Define plot colors
from evariste.trainer.utils import Metrics

cmap = plt.get_cmap("viridis")
# PR1222, fight me

ExpData = Dict
DataPoint = Dict


def get_common_parameters(experiments: List[ExpData]) -> Tuple[Dict, Set[str]]:
    """
    Take a list of name of experiments, and return a list
    of parameters that are common to all of these.
    """
    all_res_for_parameters: Dict[str, List] = defaultdict(list)
    for exp in experiments:
        for k, v in exp["params"].items():
            all_res_for_parameters[k].append(v)
    common_parameters: Dict = {}
    for k, v in all_res_for_parameters.items():
        if all([vv == v[0] for vv in v[1:]]):
            common_parameters[k] = v[0]

    return common_parameters, set(all_res_for_parameters.keys())


def get_categories(
    param_name: str, experiments: List[ExpData], ignore: List[str]
) -> List[List[Dict]]:
    """
    Group experiments by categories.
    """
    categories: List[List[Dict]] = []

    # For each experiment
    for exp in experiments:
        exp_params = exp["params"]
        if len(exp["logs"]) == 0:
            continue
        found_category = False

        # Look whether a category matches
        for cat in categories:
            cat_params = cat[0]["params"]
            is_matching = True

            # Check whether the experiments share the same other parameters
            for k, v in exp_params.items():
                if not is_matching:
                    break
                if k == param_name or k in ignore:
                    continue
                if k not in cat_params or cat_params[k] != v:
                    is_matching = False
            for k, v in cat_params.items():
                if not is_matching:
                    break
                if k == param_name or k in ignore:
                    continue
                if k not in exp_params or exp_params[k] != v:
                    is_matching = False

            if is_matching:
                found_category = True
                cat.append(exp)
                break

        # If no category was found, we create a new one
        if not found_category:
            categories.append([exp])

    return categories


def sort_data(data):
    return OrderedDict(sorted(data.items()))


def plot_experiments(
    x_type: str,
    y_type: List[str],
    experiments: List[ExpData],
    exp_filter: Optional[Callable[[ExpData], bool]] = None,
    higher_better: bool = True,
    n_best: Optional[int] = None,
    n_max_pts: Optional[int] = None,
    scores_only: bool = False,
    clip_max: Optional[float] = None,
    with_legend: Optional[bool] = True,
    filter_data: Optional[Callable[[DataPoint], bool]] = None,
    repeat_print: bool = False,
    just_return_best: bool = False,
    ignore_prefix: Optional[List[str]] = None,
    dump_paths: Optional[List[str]] = None,
    log_scale: bool = False,
    filter_missing_data_points: bool = False,
):
    """
    Plot a set of experiments.
    """
    if ignore_prefix is None:
        ignore_prefix = []
    if dump_paths is None:
        dump_paths = []

    assert isinstance(y_type, list)
    assert all(isinstance(x, str) for x in y_type)
    assert len(y_type) > 0

    # Filter experiments to plot
    experiments = [v for v in experiments if exp_filter is None or exp_filter(v)]
    if not just_return_best:
        print("Found %i experiments." % len(experiments))
    if len(experiments) == 0:
        return

    # Look for common parameters
    common_parameters, _ = get_common_parameters(experiments)
    if not just_return_best:
        print("\n==== Common parameters ====")
        print(common_parameters)
        # print(
        #     "\n".join(
        #         "{: <15}{: <10}".format(str(param[0]), str(param[1]))
        #         for param in sorted(common_parameters.items())
        #     )
        # )

    # Initialize figure and all scores
    if not scores_only:
        plt.figure(figsize=(10, 10))
    all_scores: List[Tuple[int, List[float], str, str]] = []

    # For each experiment
    curves: List = []
    curves_labels: List[str] = []

    diff_params: Set[str] = set(
        sum([list(exp["params"].keys()) for exp in experiments], [])
    ) - set(common_parameters.keys())
    diff_length = OrderedDict(
        {
            k: max(
                [len(str(exp["params"].get(k, None))) for exp in experiments] + [len(k)]
            )
            for k in diff_params
        }
    )

    for exp_num, exp in enumerate(experiments):

        # Experiment main info
        for dump_path in dump_paths:
            location = os.path.join(dump_path, str(exp["name"]), str(exp["id"]))
            if os.path.isdir(location):
                break
        else:
            raise Exception("Exp ID not found in any dump path!")
        exp_id: str = exp["id"]
        params: Dict = exp["params"]
        logs: List[DataPoint] = exp["logs"]

        # Filter checkpoints that contain the relevant value to plot for this experiment
        if filter_data is not None:
            logs = [data for data in logs if filter_data(data)]
        if len(logs) == 0:
            continue

        # Name of the experiment (flattened parameters, excluding common ones)
        params_name = "|".join(
            ("{: >%i}" % v).format(str(params.get(k, None)))
            for k, v in diff_length.items()
            if "exp_id" in k
            or k != "dump_path"
            and not any([k.startswith(prefix) for prefix in ignore_prefix])
        )

        # Define experiment data
        x_values: List[float] = [
            d[x_type] if x_type is not None else i for i, d in enumerate(logs)
        ]

        # Skip some values if there are too many
        assert len(x_values) == len(logs)
        if n_max_pts is not None and False:
            n_skip = int(max(1, len(x_values) // n_max_pts))
            if len(x_values) > n_skip:
                x_values = [x for i, x in enumerate(x_values) if i % n_skip == 0]
                y_data: List[DataPoint] = [
                    y for i, y in enumerate(logs) if i % n_skip == 0
                ]
            else:
                y_data = logs
        else:
            y_data = logs
        assert len(x_values) == len(y_data)

        # Clip experiments values
        y_values: List[List[float]] = [
            [data.get(y, -1.0) for y in y_type] for data in y_data
        ]
        if clip_max is not None:
            y_values = [
                [np.clip(w, -clip_max, clip_max).item() for w in v] for v in y_values
            ]

        # Add the curve to the plot and add experiment scores
        if not scores_only:
            xxx = x_values
            yyy = [v[0] for v in y_values]
            if filter_missing_data_points:
                zzz = [(x, y) for x, y in zip(xxx, yyy) if y != -1]
                xxx = [x for x, _ in zzz]
                yyy = [y for _, y in zzz]
            curves.append(
                plt.plot(xxx, yyy, "k.-", color=cmap(exp_num / len(experiments)))
            )
            # create tooltips
            params_str = "<br>".join(
                "%s: %s" % (k, v)
                for k, v in params.items()
                if k not in common_parameters and k != "dump_path"
            )
            params_str = params_str + "<br>" + "expID: " + exp_id
            # params_str = 'ID: %s<br>Name: %s<br>%s' % (exp_id, exp_name, params_str)
            data_str = [
                (
                    "<br>".join(
                        "%s: %s" % (str(x), str(y)) for x, y in sort_data(data).items()
                    )
                    + "<br>"
                )
                for data in y_data
            ]
            label_values = [
                '<div style="background-color: rgba(255, 255, 255, 0.8); font-size: 10px">%s</div>'
                % (params_str + "<br>========<br>" + x)
                for x in data_str
            ]
            assert len(x_values) == len(y_values) == len(label_values)
            # add tooltips
            tooltips = mpld3.plugins.PointHTMLTooltip(curves[-1], labels=label_values)
            mpld3.plugins.connect(plt.gcf(), tooltips)
            curves_labels.append(params_str.replace("<br>", " - "))
        for _x, _y in zip(
            [data[x_type] if x_type is not None else i for i, data in enumerate(logs)],
            [[data.get(y, -1.0) for y in y_type] for data in logs],
        ):
            all_scores.append((_x, _y, params_name, location))

    if len(all_scores) == 0:
        print("No data point found.")
        return

    # compute best scores
    sorted_scores = sorted(all_scores, key=lambda x: x[1][0])
    if higher_better:
        sorted_scores = sorted_scores[::-1]

    # optionally just return the best score
    if just_return_best:
        return None if len(sorted_scores) == 0 else sorted_scores[0][1][0]

    # Plot all curves
    if not scores_only:
        if with_legend:
            plt.legend(curves, curves_labels)
        if log_scale:
            plt.yscale("log")
        plt.show()

    # Show best experiments
    print("\n" + " - ".join(y_type + [x_type]) + "\n")
    col_width = [
        max([len("%.4f" % v[1][i]) for v in sorted_scores[:n_best]])
        for i in range(len(y_type))
    ] + [max([len(str(v[0])) for v in sorted_scores[:n_best]])]
    pattern = " | ".join(["{: >%i}" % cw for cw in col_width])
    print(
        pattern.format(*(["-"] * (len(y_type) + 1)))
        + "  |"
        + "|".join(
            [
                ("{: <%i}" % v).format(k)
                for k, v in diff_length.items()
                if "exp_id" in k
                or k != "dump_path"
                and not any([k.startswith(prefix) for prefix in ignore_prefix])
            ]
        )
    )

    printed: Set[str] = set()
    for _x, _y, params_name, location in sorted_scores[:n_best]:
        if params_name in printed and not repeat_print:
            continue
        print(
            pattern.format(*(["%.4f" % y for y in _y] + [str(_x)]))
            + "  |"
            + params_name
            + "|"
            + os.path.join(location, "train.log")
        )
        printed.add(params_name)

    print("\n")


def get_curves(
    x_name: str, y_name: str, metrics: Metrics
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper to get x and y array for two keys given a metrics list parsed
    from metrics.jsonl
    """
    x, y = [], []
    for item in metrics:
        if x_name in item and y_name in item:
            x.append(item[x_name])
            y.append(item[y_name])
    return np.array(x), np.array(y)
