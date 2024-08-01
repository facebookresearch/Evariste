#!/bin/bash

# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

which python3
time python3 tools/run_lean_and_save_results.py src/all.lean "DATA_DIR"
time python3 tools/extract_trace_data.py "DATA_DIR"
time python3 tools/extract_proof_data.py "DATA_DIR"
time python3 tools/extract_training_testing_data.py "DATA_DIR"
