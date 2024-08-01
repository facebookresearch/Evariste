/* Copyright (c) 2019-present, Facebook, Inc.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree. */

function getTokenClass(token) {

    // 0-9
    if (token.match(/^[0-9_]+$/)) {
        return "hl_digit";
    }

    // A-Z_
    if (token.match(/^[A-Z_]+$/)) {
        if (token.endsWith("_TAC")) {
            return "hl_tactic";
        } else {
            return "hl_theorem";
        }
    }

    // a-z_
    if (token.match(/^[a-z_]+$/)) {
        if (token.length === 1) {
            return "hl_var";
        } else {
            return "hl_definition";
        }
    }

    // a-z_ 0-9
    if (token.match(/^[a-z_]+[0-9]+$/)) {
        return "hl_var";
    }

    // symbol
    if (token.toLowerCase() === token.toUpperCase()) {
        return "hl_symbol";
    }

    // unknown token
    console.log("unknown token " + token);
    return "hl_normal";
}

let hl_utils = {
    splitTokens(s) {
        let tokens = s.split(" ");
        let result = [];
        tokens.forEach(function (token,) {
            result.push({
                token: token,
                tokenClass: getTokenClass(token),
            });
        });
        return result;
    }
};

export default hl_utils