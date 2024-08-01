// Copyright (c) 2019-present, Facebook, Inc.
// All rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <iostream>
#include <string>
#include "earley.hpp"

using namespace std;

int main(void) {
    auto grammar = Grammar(string("input_2.in"));
    grammar.parse("1 + 1");
//    grammar.parse("class ( 1 + 1 )");
//    grammar.parse("class ( 1 + ( 1 + 1 ) )");
//    grammar.parse("wff ( 1 + ( 1 + 1 ) )");
//    grammar.parse("class ( 1 + ( 1 + 1 )");
//    grammar.parse("class ( 1 + ( 1 1 ) )");
    return 0;
}