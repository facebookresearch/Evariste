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


git submodule update --init
cd mathlib
./scripts/mk_all.sh
cd ..
rm -Rf build
rm -Rf lean/build
mkdir -p lean/build/release
cd lean/build/release
cmake ../../src -DLEAN_EXTRA_CXX_FLAGS=-g3
make -j`nproc` standard_lib
cd ../../../

mkdir -p checkpoint-process/build/release
cd checkpoint-process/build/release
cmake ../.. -DCMAKE_BUILD_TYPE=release
make
cd ../../../

mkdir -p build/release
cd build/release
cmake ../../src
make -j`nproc`


