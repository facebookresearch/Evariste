# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from pathlib import Path
import sys
import pandas as pd
import numpy as np
from dhash import get_split
from tqdm import tqdm
import json

RAW_TRACED_DATA_DIR = "raw_traced_data"
EXTRACTED_PROOF_DATA_DIR = "extracted_proof_data"
CLEANED_TRAINING_DATA_DIR = "cleaned_training_data"


def gather_data_for_model(
    tactic_state_goal: pd.DataFrame,
    tactic_state: pd.DataFrame,
    tactics: pd.DataFrame,
    no_solve1: bool,
):
    # take first goal in each tactic state
    df = tactic_state_goal.copy()
    df = df[df["ix"] == 0]
    df["tactic_state_key"] = df["filename"] + ":" + df["tactic_state"]
    df = df[["tactic_state_key", "goal_pp"]]
    df = df.set_index("tactic_state_key")

    # only use the tactic states which happen before the tactic is applied
    df2 = tactic_state.copy()
    df2 = df2[df2["before_after"] == "before"]
    df2["tactic_state_key"] = df2["filename"] + ":" + df2["key"]
    df2["tactic_instance_key"] = df2["filename"] + ":" + df2["tactic_instance"]
    df2["tactic_key"] = df2["tactic_instance_key"].apply(
        lambda k: ":".join(k.split(":")[:-1])
    )
    df2 = df2[
        [
            "tactic_state_key",
            "tactic_instance_key",
            "tactic_key",
            "decl_name",
            "open_namespaces",
        ]
    ]
    df2["decl_name"] = df2["decl_name"].str.replace("`", "")
    df2["open_namespaces"] = df2["open_namespaces"].str.replace("`", "")
    df2["open_namespaces"] = df2["open_namespaces"].str.replace(
        r"\[anonymous\] ?", "", regex=True
    )
    df2 = df2.set_index("tactic_state_key")

    df = df.join(df2, how="inner")
    df = df.set_index("tactic_key")

    # join with the tactic commands
    df3 = tactics.copy()
    df3["tactic_key"] = df3["filename"] + ":" + df3["trace_key"]
    df3 = df3[
        [
            "tactic_key",
            "filename",
            "line",
            "column",
            "proof_key",
            "code_string",
            "class",
        ]
    ]
    df3 = df3.rename(
        columns={"class": "tactic_class", "code_string": "human_tactic_code"}
    )
    df3 = df3.set_index("tactic_key")
    df = df.join(df3, how="inner")
    df = df.reset_index()
    df = df.drop(["tactic_key", "tactic_instance_key"], axis="columns")

    # remove solve1 tactics
    if no_solve1:
        df = df[df["tactic_class"] != "solve1"]

    # clean input
    df["cleaned_goal"] = (
        df["goal_pp"]
        # remove tags
        .str.replace(r"^[^:⊢]*\n", "", regex=True)
        # remove pp indenting (including extra line wraps)
        .str.replace(r"\n +", " ", regex=True)
        # replace new lines with tabs
        .str.replace(r"\n", "\t", regex=True)
    )
    # train-test-validate split based on decl_name

    decl_names = df["decl_name"].unique()
    tvt = [get_split(nm) for nm in decl_names]
    tvt_split = pd.Series(tvt, index=decl_names)
    df["split"] = df["decl_name"].map(tvt_split)
    # tvt =

    # # train-test-validate split based on proof_key
    # proof_keys = df['proof_key'].unique()
    # rng = np.random.default_rng(seed=0)
    # tvt = rng.choice(['train', 'valid', 'test'], p=[0.8, 0.1, 0.1], size=len(proof_keys))
    # tvt_split = pd.Series(tvt, index=proof_keys)
    # tvt_split
    # df['split'] = df['proof_key'].map(tvt_split)

    return df


def _parse_main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--no-solve1", action="store_true", dest="no_solve1")
    parser.add_argument("--sexp", action="store_true", dest="sexp")
    return parser.parse_args()


def main():
    opts = _parse_main()
    data_dir = Path(opts.data_dir)
    assert data_dir.exists(), data_dir
    assert data_dir.is_dir(), data_dir

    # load previously extracted data
    print("[extract_training_test_data] LOADING PREVIOUSLY EXTRACTED DATA")
    tactic_state_goal = pd.read_json(
        data_dir / RAW_TRACED_DATA_DIR / "tactic_state_goal.json", orient="records"
    )
    tactic_state = pd.read_json(
        data_dir / RAW_TRACED_DATA_DIR / "tactic_state.json", orient="records"
    )
    tactics = pd.read_json(
        data_dir / EXTRACTED_PROOF_DATA_DIR / "tactics.json", orient="records"
    )

    # process
    print("[extract_training_test_data] PROCESSING TO FULL DATA")
    full_data = gather_data_for_model(
        tactic_state_goal, tactic_state, tactics, no_solve1=opts.no_solve1
    )

    # save to files
    cleaned_data_dir = data_dir / CLEANED_TRAINING_DATA_DIR
    cleaned_data_dir.mkdir(exist_ok=True)

    full_data.to_csv(cleaned_data_dir / "data_and_metadata.csv")

    example_name_set = set()
    for split in ["train", "valid", "test"]:
        print(f"[extract_training_test_data] CREATING {split} SPLIT")
        src_file = cleaned_data_dir / f"{split}.src"
        tgt_file = cleaned_data_dir / f"{split}.tgt"
        name_file = cleaned_data_dir / f"{split}.names"
        name_index_file = cleaned_data_dir / f"{split}.index"
        data_split = full_data[full_data["split"] == split]
        skip_count = 0
        multiple_goal_count = 0
        example_set = set()
        goal_tk = "⊢" if not opts.sexp else "GOAL"
        with open(str(src_file), "w") as src_handle:
            with open(str(tgt_file), "w") as tgt_handle:
                with open(str(name_file), "w") as name_handle:
                    with open(str(name_index_file), "w") as name_index_handle:
                        for idx, row in tqdm(
                            data_split.iterrows(), total=len(data_split.index)
                        ):
                            # discard solve1s applied to only 1 goal to avoid duplication
                            if (
                                row["tactic_class"] == "solve1"
                                and row["cleaned_goal"].count(goal_tk) == 1
                            ):
                                skip_count += 1
                                continue
                            if row["cleaned_goal"].count(goal_tk) > 1:
                                multiple_goal_count += 1
                            example_src = row["cleaned_goal"]
                            example_tgt = row["human_tactic_code"]
                            example_name = (
                                row["decl_name"] + " " + row["open_namespaces"]
                            )

                            if (example_src, example_tgt) in example_set:
                                continue
                            else:
                                example_set.add((example_src, example_tgt))

                            src_handle.write(example_src + "\n")
                            tgt_handle.write(example_tgt + "\n")
                            if (
                                row["decl_name"] in example_name_set
                                or row["decl_name"] == "_example"
                            ):
                                pass
                            else:
                                name_handle.write(example_name + "\n")
                                example_name_set.add(row["decl_name"])
                            name_index_entry = (
                                json.dumps(
                                    dict(
                                        src=row["cleaned_goal"],
                                        decl_nm=row["decl_name"],
                                    )
                                )
                                + "\n"
                            )
                            name_index_handle.write(name_index_entry)

        print(f"SKIPPED {skip_count} FOR SPLIT {split}")
        print(f"MULTIPLE GOAL DATAPOINTS: {multiple_goal_count}")

        # for src_tgt in ['src', 'tgt']:
        #     path = cleaned_data_dir / f"{split}.{src_tgt}"
        #     if src_tgt == "src":
        #         data = data_split['cleaned_goal']
        #     else:
        #         data = data_split['human_tactic_code']
        #     path.touch()
        #     path.write_text("\n".join(data))


if __name__ == "__main__":
    main()
