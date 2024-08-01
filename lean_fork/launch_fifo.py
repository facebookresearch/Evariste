# Copyright (c) Facebook, Inc. and its affiliates.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
import os
import sys
from leanml import get_api, MissingLib
import io
from subprocess import Popen

fifo_folder = Path(sys.argv[1])
assert fifo_folder.exists(), "{fifo_folder} doesn't exist"

cur_dir = Path(os.path.dirname(os.path.abspath(__file__)))
path_to_server = cur_dir / "build" / "release" / "ml_server"
root_dir = path_to_server.parent.parent.parent
if not all([x in os.listdir(root_dir) for x in {'lean', 'mathlib'}]):
    raise MissingLib(f"Both 'lean' and 'mathlib' are expected in folder {root_dir}")
lean_roots={
    'lean': root_dir / "lean/library",
    'mathlib': root_dir / "mathlib/src",
    'others': root_dir / 'others',
    'cleaning_utils': root_dir / 'cleaning_utils',
}
os.environ["LEAN_PATH"] = ':'.join([str(root) for root in lean_roots.values()])


# create FIFOS
if {'stdin', 'stdout'}.intersection(os.listdir(fifo_folder)):
    print("Fifo already exist in this folder. Delete ? [yn]")
    answer = input() 
    assert answer in  {'y', 'n'}, "answer with y (yes) or n (no)"
    if answer == 'n':
        exit()
    else:
        try:
            os.unlink(fifo_folder / 'stdin')
            os.unlink(fifo_folder / 'stdout')
        except FileNotFoundError:
            pass

os.mkfifo(fifo_folder / 'stdin')
os.mkfifo(fifo_folder / 'stdout')

# Launch Server
cmd = f"{path_to_server} < {str(fifo_folder / 'stdin')} > {str(fifo_folder / 'stdout')}"
print(f"Executing : {cmd}")
proc = Popen(
    cmd,
    encoding="utf-8",
    shell=True
)
stdin_maintain_open = io.open(fifo_folder / 'stdin', 'wb', -1)  # not closing prevents EOF

proc.wait()
