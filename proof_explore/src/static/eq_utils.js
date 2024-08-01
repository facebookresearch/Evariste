/* Copyright (c) 2019-present, Facebook, Inc.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree. */

function getTokenClass(token) {

  // 0-9
  if (token.match(/^[0-9_]+$/)) {
      return "eq_digit";
  }

  // PI
  if (token === "PI") {
      return "eq_const";
  }

  // x0, x1, ...
  if (token.match(/^x[0-9_]+$/)) {
      return "eq_var";
  }

  // binary operator
  if (token.match(/^[+*-]+$/)) {
      return "eq_bin_operator";
  }

  // comparison operator
  if (token.match(/^(==|<=|<|!=)$/)) {
      return "eq_comp_operator";
  }

  // function
  if (token.match(/^[a-z_]+$/)) {
      return "eq_function";
  }

  // a-z_ 0-9
  if (token.match(/^[a-z]+$/)) {
      return "eq_var";
  }

  // symbol
  if (token.toLowerCase() === token.toUpperCase()) {
      return "eq_symbol";
  }

  // unknown token
  console.log("unknown token " + token);
  return "eq_normal";
}

var splitOrig = String.prototype.split; // Maintain a reference to inbuilt fn
String.prototype.split = function (){
    if(arguments[0].length > 0){
        // Check if our separator is an array
        if(Object.prototype.toString.call(arguments[0]) == "[object Array]"){
            return splitMulti(this, arguments[0]);  // Call splitMulti
        }
    }
    return splitOrig.apply(this, arguments); // Call original split maintaining context
};

let eq_utils = {
  splitTokens(s) {
      // let tokens = s.split(/\(|\)|x0|x1|x2|x3|x4| /) ;
      // var separators = ["(", ")", " "];
      // console.log(separators.join('|'));
      // var tokens = s.split(new RegExp(separators.join('|'), 'g'));
      s = s.replace(/\(/g, ' ( ');
      s = s.replace(/\)/g, ' ) ');
      s = s.replace(/\+/g, ' + ').trim();
      s = s.replace(/-/g, ' - ').trim();
      s = s.replace(/\*/g, ' * ').trim();
      s = s.replace(/\//g, ' / ').trim();
      s = s.replace(/  +/g, ' ').trim();
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

export default eq_utils