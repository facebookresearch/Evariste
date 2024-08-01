/* Copyright (c) 2019-present, Facebook, Inc.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree. */

let utils = {
    handleServerError(data) {
        var message = "Error " + data["code"] + " detected. Reason: " + data["reason"];
        if ("traceback" in data) {
            message = message + "\nTraceback:\n" + data["traceback"];
        }
        window.alert(message);
    }
};

export default utils