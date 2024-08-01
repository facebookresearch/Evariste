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
#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <queue>
#include <sstream>

#include "kernel/expr_sets.h"
#include "kernel/expr_maps.h"
#include "kernel/environment.h"
#include "library/io_state.h"
#include "library/tactic/tactic_state.h"
#include "library/module_mgr.h"
#include "frontends/lean/json.h"
#include "frontends/lean/parser_state.h"
#include "util/lean_path.h"
#include "util/thread.h"
#include "frontends/lean/pp.h"

namespace lean {

using tp = chrono::time_point<chrono::steady_clock>;
using namespace std::chrono_literals;

struct cancellation_scheduler {
    std::unique_ptr<lean::lthread> m_thread;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::multimap<tp, cancellation_token> m_queue;
    bool m_finished = false;
    cancellation_scheduler();
    ~cancellation_scheduler();
    void cancel_at(tp tp, cancellation_token const & tok);
    template <class T> void cancel_after(T dur, cancellation_token const & tok) {
        cancel_at(chrono::steady_clock::now() + dur, tok);
    }
};

struct session_node {
    mutex node_mutex;
    std::unordered_map<std::string, json> tactics_ran; // tactic_str -> result
    tactic_state ts;
    std::string pp_ts;  // avoid recomputing the pretty-printed ts
    session_node(tactic_state const & ts, std::string&& pp_ts): ts(ts), pp_ts(pp_ts) {};
};

struct session {
    std::string decl_name;
    mutex session_mutex;
    
    std::vector<std::shared_ptr<session_node>> nodes;
    expr_map<unsigned> expr_to_id;
    std::shared_ptr<const snapshot> m_snapshot;
    bool merge_alpha_equiv;
    json opts;
    session(std::string decl_name, std::shared_ptr<const snapshot> snapshot, bool merge_alpha_equiv, json opts):
      decl_name(decl_name), m_snapshot(snapshot), merge_alpha_equiv(merge_alpha_equiv), opts(opts) {};
    std::string print_ts(tactic_state ts, bool all=false){
        auto ts_opts = ts.get_options();
        for (auto it : json::iterator_wrapper(opts)) {
            ts_opts = ts_opts.update(string_to_name(it.key()), bool(it.value()));
        }
        if (all) {
            ts_opts = ts_opts.update(string_to_name("pp.all"), true);
        }
        auto to_print_ts = set_options(ts, ts_opts);
        auto res = (sstream() << to_print_ts.pp()).str();
        return res;
    }
};

struct output_queue;

struct mt_accounting {
    mutex sessions_mutex;
    std::unordered_map<std::string, std::shared_ptr<session>> sessions;  // randomly generated session name -> all intermediate tactic states
    std::shared_ptr<output_queue> m_stderr;
    bool profile_tactic_process = false;
    std::atomic<long int> waiting_for_lock;
    std::atomic<long int> total_processing;
};

struct output_queue {
    std::ostream & m_out;

    std::unique_ptr<lthread> m_out_thread;
    std::mutex m_out_mutex;
    std::condition_variable m_out_cv;
    std::queue<std::string> m_to_print;
    bool m_out_finished = false;

    std::shared_ptr<std::atomic<int>> req_in_queue;

    output_queue(std::ostream &, std::shared_ptr<std::atomic<int>>);
    ~output_queue();
    void send(std::string const & s, std::shared_ptr<std::atomic<int>>);
    void send(std::string const & s);
    void send(sstream const & s) { send(s.str()); }
    void send(sstream const & s, std::shared_ptr<std::atomic<int>> r) { send(s.str(), r); }
    void send(json const & j) { send(sstream() << j); }
    void send(json const & j, std::shared_ptr<std::atomic<int>> r) { send(sstream() << j, r); }
};

class ml_server : public module_vfs {
    search_path m_path;
    environment m_initial_env;
    io_state m_ios;
    log_tree m_lt;
    std::unique_ptr<module_mgr> m_mod_mgr;
    std::unique_ptr<task_queue> m_tq;
    fs_module_vfs m_fs_vfs;

    log_tree & get_log_tree() { return m_lt; }

    std::shared_ptr<std::atomic<int>> handling, req_in_queue;
    std::shared_ptr<mt_accounting> acct;
    std::shared_ptr<output_queue> m_stdout;
    std::shared_ptr<const lean::snapshot> m_snapshot_cmd;

public:
    ml_server(unsigned num_threads, search_path const & path, environment const & initial_env, io_state const & ios, bool fast_start);
    ~ml_server();

    std::shared_ptr<module_info> load_module(module_id const & id, bool can_use_olean) override;

    void pre_load(search_path const & path);
    void run(search_path const & path);
    void test();
    void handle_async_request(json const & req);
    void clear_handlers();
    void setup_handlers(unsigned num_threads);

    void set_profile_tactic_process(bool v) { acct->profile_tactic_process = v; };
};

json handle_ml_request(std::shared_ptr<mt_accounting> acct, json const & jreq);
tactic_state apply(tactic_state& some_ts, snapshot const & snapshot, const std::string& tactic_str);

}
