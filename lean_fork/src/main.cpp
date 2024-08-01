/*
Copyright (c) Facebook, Inc. and its affiliates.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <dlfcn.h>
#include "init/init.h"
#include "ml_server.h"
#include "kernel/standard_kernel.h"
#include "frontends/lean/pp.h"
#include "frontends/lean/json.h"
#include "library/io_state.h"
#include "checkpoint.h"

int main(int argc, char* argv[]) {
    // Untie cin from cout to avoid having to lock one from the other
    // https://www.cplusplus.com/reference/ios/ios/tie/
    std::cin.tie(nullptr);

    // Unsync cout from stdio
    // https://stackoverflow.com/questions/14991671/cin-causing-cout-to-flush-after-i-untied-the-two-streams
    std::cout.sync_with_stdio(false);
    std::cin.sync_with_stdio(false);

    // Untie cerr as well to avoid flushing cout inconveniently
    std::cerr.tie(nullptr);
    std::cerr.sync_with_stdio(false);


    lean::initializer m_init;
    lean::options opts;
    opts = opts.update(lean::get_verbose_opt_name(), false);
    opts = opts.update("skip_proofs", true);

    auto maybe_paths = lean::get_lean_path_from_env();
    if (!maybe_paths) {
        std::cerr << "[ML Server] " << "ERROR: invalid LEAN_PATH" << std::endl;
        return 1;
    }
    auto paths = maybe_paths.value();
    unsigned num_threads = 16; //lean::hardware_concurrency();
    std::cerr << "[ML Server] " << "Launching lean server with " << num_threads << " threads" << std::endl;
    lean::environment env = lean::mk_environment(LEAN_BELIEVER_TRUST_LEVEL + 1);

    bool fast_start = true;
    if (argc > 1 && std::string("preload").compare(argv[1]) == 0) {
        fast_start = false;
    } else {
        std::cerr << "[ML Server] " << "Using paths :" << std::endl;
        for (auto p: paths) {
            std::cerr << "\t" << p << std::endl;
        }
    }

    lean::io_state ios(opts, lean::mk_pretty_formatter_factory());
    
    lean::ml_server server(num_threads, paths, env, ios, fast_start);
    lean::scope_global_ios scoped_ios(ios); 

    server.clear_handlers();

    std::cerr << "[ML Server] Saving checkpoint" << std::endl;
    checkpoint_save(fast_start ? "fast_ckpt" : "ckpt");
    std::cout << "lean ml server ready" << std::endl;    
    std::string thread_string;
    std::getline(std::cin, thread_string);
    lean::json req = lean::json::parse(thread_string);
    server.set_profile_tactic_process(req.at("profile_tactic_process"));
    unsigned run_num_threads = req.at("num_threads");
    std::string paths_str = req.at("paths");
    std::cout << lean::json{"ok"} << std::endl;

    /* borrowed from lean_path.cpp */
    auto lean_path = lean::normalize_path(paths_str);
    unsigned i  = 0;
    unsigned j  = 0;
    unsigned sz = static_cast<unsigned>(lean_path.size());
    lean::search_path runtime_paths;
    for (; j < sz; j++) {
        if (lean::is_path_sep(lean_path[j])) {
            if (j > i)
                runtime_paths.push_back(lean_path.substr(i, j - i));
            i = j + 1;
        }
    }
    if (j > i)
        runtime_paths.push_back(lean_path.substr(i, j - i));

    std::cerr << "[ML Server] Running with " << run_num_threads << " threads" << std::endl;
    server.setup_handlers(run_num_threads);
    std::cerr << "[ML Server] Checkpoint saved, running" << std::endl;
    server.run(runtime_paths);
    std::cerr << "[ML Server] Exiting normally" << std::endl;
    return 0;
}
