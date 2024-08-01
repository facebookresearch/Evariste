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

#include <random>
#include <mutex>
#include <filesystem>
#include <thread>
#include <chrono>
#include <unordered_map>

#include "util/timer.h"
#include "kernel/type_checker.h"
#include "kernel/expr_sets.h"
#include "kernel/abstract.h"
#include "kernel/for_each_fn.h"
#include "library/trace.h"
#include "library/tactic/tactic_evaluator.h"
#include "library/tactic/intro_tactic.h"
#include "library/unfold_macros.h"
#include "library/type_context.h"
#include "library/compiler/rec_fn_macro.h"
#include "library/library_task_builder.h"
#include "library/st_task_queue.h"
#include "library/mt_task_queue.h"
#include "frontends/lean/elaborator.h"
#include "frontends/lean/json.h"
#include "frontends/lean/parser.cpp"
#include "frontends/lean/cmd_table.h"
#include "frontends/lean/builtin_cmds.cpp"
#include "frontends/lean/print_cmd.h"
#include "library/check.h"
#include "kernel/expr.h"

#include "ml_server.h"



namespace lean {

cancellation_scheduler::cancellation_scheduler() {
    m_thread.reset(new lthread([&] {
        std::unique_lock<std::mutex> lock(m_mutex);
        while (!m_finished) {
            tp now = chrono::steady_clock::now();
            tp next_wakeup = now + chrono::seconds(1);

            while (m_queue.begin() != m_queue.end()) {
                auto it = m_queue.begin();
                if (it->first <= now) {
                    cancel(it->second);
                    m_queue.erase(it);
                } else {
                    next_wakeup = it->first + chrono::milliseconds(10);
                    break;
                }
            }

            m_cv.wait_until(lock, next_wakeup);
        }
    }));
}

void cancellation_scheduler::cancel_at(tp tp, cancellation_token const & tok) {
    bool need_notify;
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        need_notify = m_queue.begin() == m_queue.end() || m_queue.begin()->first > tp;
        m_queue.insert({tp, tok});
    }
    if (need_notify) m_cv.notify_one();
}

cancellation_scheduler::~cancellation_scheduler() {
    (std::unique_lock<std::mutex>(m_mutex), m_finished = true);
    m_cv.notify_one();
    if (m_thread) m_thread->join();
}

static cancellation_scheduler * g_cancel_sched;

void ml_server::pre_load(search_path const & path) {
    std::vector<std::string> all_module_paths;
    std::cerr << "[ML Server] " << "LEAN_PATH has paths" << std::endl;
    for (auto & root : path) {
        std::cerr << root << std::endl;
    }

    for (auto & root : path) {
        // others folder contains minif2f files which lead to increased ram usage
        // loading at runtime is ~3 secs.
        if (root.find("others") != std::string::npos || root.find("cleaning_utils") != std::string::npos) {
            std::cerr << "[ML Server] not preloading from " << root << std::endl;
            continue;
        }
        const std::string ext = ".lean";
        for (const auto& dir_entry : std::filesystem::recursive_directory_iterator(root)){
            std::string module_path = dir_entry.path().c_str();
            if (module_path.length() >= ext.length()){
                if (0 == module_path.compare(module_path.length() - ext.length(), ext.length(), ext)){
                    all_module_paths.push_back(module_path);
                }
            }
        }
    }

    // Start parsing.
    for (auto & mod_path : all_module_paths)
        m_mod_mgr->get_module(mod_path);

    // Wait for everything to be done by getting absolutely all snapshots.
    for (auto & mod_path : all_module_paths) {
        auto mod = m_mod_mgr->get_module(mod_path);
        auto res = mod->m_snapshots;
        while (res && res->m_next) {
            res = get(res->m_next);
        }
    }
}

expr parse_goal_core(snapshot const & s, json goal) {
    name decl_name = "dummy";
    std::string goal_string = goal.at("goal");
    std::istringstream input_stream(goal_string);
    parser p(s.m_env, get_global_ios(), mk_dummy_loader(), input_stream, "dummy file");
    p.from_snapshot(s);
    p.scan();

    auto pre_expr = (p.no_error_recovery_scope(), p.parse_expr());

    auto e = p.elaborate_type(decl_name, list<expr>(), pre_expr);
    return e.first;
}

expr parse_command_core(snapshot const & s, json command) {
    std::string goal_string = command.at("command");
    std::string decl_name = command.at("decl_name");

    std::istringstream input_stream(goal_string);
    parser p(s.m_env, get_global_ios(), mk_dummy_loader(), input_stream, "dummy file");
    p.from_snapshot(s);
    p.scan();
    (p.no_error_recovery_scope(), p.parse_command(cmd_meta()));
    return p.env().get(decl_name).get_type();
}

tactic_state parse_goal(snapshot const & s, json parsed_goals, bool as_goal) {
    name decl_name = "dummy";
    auto ts = mk_tactic_state_for(s.m_env, s.m_options, decl_name, local_context(), mk_true());

    buffer<expr> goals;
    for (auto & g : parsed_goals) {
        auto goal_expr = as_goal ? parse_goal_core(s, g) : parse_command_core(s, g);
        unsigned n_hyps = g.at("n_hyps");

        auto mctx = ts.mctx();
        auto goal = mctx.mk_metavar_decl(local_context(), goal_expr);
        ts = set_mctx(ts, mctx);
        ts = set_goals(ts, list<expr>(goal));

        if (auto ts2 = intron(n_hyps, ts, false)) {
            ts = *ts2;
        } else {
            throw exception(sstream() << "wrong number of hypotheses in goal state parser:\n" << ts.pp());
        }

        goals.push_back(head(ts.goals()));
    }
    return set_goals(ts, to_list(goals));
}

// Returns all expression metavariables that are transitively reachable from
// mvar, including mvar itself.
// In addition, also returns all level metavariables (wrapped in mk_sort)
// reachable from mvar.
expr_set referenced_mvars(metavar_context ctx, expr const & mvar) {
    expr_set result, visited;
    std::function<void(level const&)> visit_level = [&] (level const & lvl) {
        for_each(ctx.instantiate_mvars(lvl), [&] (level const & lvl) {
            if (!has_meta(lvl)) return false;
            if (is_meta(lvl)) result.insert(mk_sort(lvl));
            return true;
        });
    };
    std::function<void(expr const&)> visit_mvar = [&] (expr const & mvar) {
        result.insert(mvar);
        if (auto decl = ctx.find_metavar_decl(mvar)) {
            std::function<void(expr const&)> visit = [&] (expr const & e) {
                for_each(ctx.instantiate_mvars(e), [&] (expr const & e, unsigned) -> bool {
                    if (!has_metavar(e)) return false;
                    if (is_metavar(e) && visited.insert(e).second) {
                        visit_mvar(e);
                    }
                    if (is_constant(e)) {
                        for (auto & l : const_levels(e)) visit_level(l);
                    }
                    if (is_sort(e)) {
                        visit_level(sort_level(e));
                    }
                    return true;
                });
            };
            visit(decl->get_type());
            decl->get_context().for_each([&] (auto & ldecl) {
                visit(ldecl.get_type());
                if (auto v = ldecl.get_value()) visit(*v);
            });
        }
    };
    visit_mvar(mvar);
    return result;
}

template <class K, class H, class KE, class A>
bool intersects(std::unordered_set<K, H, KE, A> const & as, std::unordered_set<K, H, KE, A> const & bs) {
    for (auto & a : as) if (bs.count(a)) return true;
    return false;
}

list<tactic_state> split_tactic_state(tactic_state const & ts, std::vector<expr_set> & subgoal_metavars) {
    buffer<expr> goals; to_buffer(ts.goals(), goals);

    buffer<expr_set> refd_mvars;
    for (auto & g : goals)
        refd_mvars.push_back(referenced_mvars(ts.mctx(), g));

    buffer<unsigned> highest_connected_goal;
    highest_connected_goal.resize(goals.size());
    for (unsigned i = goals.size(); i--;) {
        unsigned h = i;
        bool intersected = false;
        // find highest goal k we intersect
        // then hcg[i] = max_{j in [i, k]} hcg[j]
        for (unsigned j = goals.size() - 1; j > i; j--) {
            intersected |= intersects(refd_mvars[i], refd_mvars[j]);
            if (intersected) {
                h = std::max(h, highest_connected_goal[j]);
            }
        }
        highest_connected_goal[i] = h;
    }

    buffer<tactic_state> split;
    for (unsigned i = 0; i < goals.size();) {
        unsigned j = highest_connected_goal[i] + 1;
        expr_set subgoal_mvars;
        for (unsigned k = i; k < j; k++) {
            for (auto & v : refd_mvars[k]) {
                subgoal_mvars.insert(v);
            }
        }
        subgoal_metavars.push_back(subgoal_mvars);

        auto splitted_ts = set_goals(ts, to_list(goals.begin() + i, goals.begin() + j));
        split.push_back(splitted_ts);

        i = j;
    }

    return to_list(split);
}

void ground_mvars(type_context_old & tc, buffer<expr> & locals, expr const & e) {
  for_each(e, [&] (expr const & e, unsigned) {
    if (!e.has_expr_metavar()) return false;
    if (is_metavar(e)) {
      auto e_ = tc.instantiate_mvars(e);
      if (e_ != e) {
        ground_mvars(tc, locals, e_);
      } else if (auto mvar_decl = tc.find_metavar_decl(e)) {
        auto lctx = mvar_decl->get_context();
        buffer<expr> args;
        lctx.for_each([&] (auto & ldecl) { args.push_back(ldecl.mk_ref()); });
        auto ty = tc.mk_pi(lctx, args, mvar_decl->get_type());
        ground_mvars(tc, locals, ty);
        ty = tc.instantiate_mvars(ty);
        auto repl = mk_local(mvar_decl->get_name(), mvar_decl->get_name(), ty, binder_info());
        locals.push_back(repl);
        lctx.for_each([&] (auto & ldecl) { if (!ldecl.get_value()) repl = mk_app(repl, ldecl.mk_ref()); });
        tc.assign(e, repl);
      } else {
        throw exception(sstream() << "undeclared metavariable " << e);
      }
    }
    return true; // recurse
  });
}

void ground_uvars(type_context_old & tc, level const & l) {
  for_each(l, [&] (level const & l) {
    if (is_meta(l)) {
      if (!tc.is_assigned(l))
        tc.assign(l, mk_param_univ(meta_id(l)));
    }
    return true; // recurse
  });
}

void ground_uvars(type_context_old & tc, expr const & e) {
  for_each(tc.instantiate_mvars(e), [&] (expr const & e, unsigned) {
    if (!e.has_univ_metavar()) return false;
    if (is_sort(e)) {
      ground_uvars(tc, sort_level(e));
    } else if (is_constant(e)) {
      for (auto & l : const_levels(e))
        ground_uvars(tc, l);
    }
    return true; // recurse
  });
}

// The set of referenced metavariables are the metavariables reachable from
// the tactic state's goals.
expr_set get_refd_mvars_from_goals(tactic_state const & s) {
  expr_set refd_mvars;
  for (auto & goal : s.goals()){
    for (auto & mvar : referenced_mvars(s.mctx(), goal))
      refd_mvars.insert(mvar);
  }
  return refd_mvars;
}


void typecheck(expr const & to_check, environment const & fresh, environment const & old){
    for_each(to_check, [&] (expr const & e, unsigned) {
        // no metavar and no locals in an auxiliary def
        if (is_sorry(e)) throw exception("proof contains sorry");
        if (is_rec_fn_macro(e)) throw exception("proof contains rec_fn_macro");
        if (is_constant(e)) {
            if (auto decl = fresh.find(const_name(e))) {
                if (!decl->is_trusted()){
                    throw exception(sstream() <<
                        "proof contains meta constant: " << const_name(e));
                } else{
                    if (!old.find(const_name(e))) {
                        auto maybe_task = decl->get_value_task();
                        if (!maybe_task || maybe_task->peek()){
                            typecheck(decl->get_value(), fresh, old);
                        } else {
                            throw exception(sstream() <<
                                "Async constant: " << const_name(e));
                        }
                    }

                }
            } else {
                throw exception(sstream() <<
                    "proof contains unknown constant: " << const_name(e) << "\n");
            }
        }
        return true; // recurse
    });
}

void check_tactic_state2(tactic_state const & old, tactic_state const & s) {
  auto mctx = s.mctx();

  // The set of referenced metavariables are the metavariables reachable from
  // the new tactic state's goals.  No other expression metavariables should occur in the
  // proof.
  expr_set refd_mvars = get_refd_mvars_from_goals(s);
  for (auto & old_goal : get_refd_mvars_from_goals(old)) {
    if (!is_metavar(old_goal)) continue; // No need to check "proofs" of universe metavariables.
    auto result = mctx.instantiate_mvars(old_goal);
    for_each(result, [&] (expr const & e, unsigned) {
        if (is_metavar(e)) {
            if (refd_mvars.count(e) == 0) {
                throw exception(sstream() << "proof contains metavariable " << mlocal_name(e) << "\n"
                    << set_goals(s, {e}).pp());
            }
            return false;
        }
        if (is_local(e)) {
          // ignore type of indexed local
          return false;
        }
        if (is_sorry(e)) throw exception("proof contains sorry");
        if (is_rec_fn_macro(e)) throw exception("proof contains rec_fn_macro");
        if (is_constant(e)) {
            if (auto decl = s.env().find(const_name(e))) {
                if (!decl->is_trusted()){
                    throw exception(sstream() <<
                        "proof contains meta constant: " << const_name(e));
                } else{
                    if (!old.env().find(const_name(e))) {
                        auto maybe_task = decl->get_value_task();
                        if (!maybe_task || maybe_task->peek()){
                            typecheck(decl->get_value(), s.env(), old.env());
                        } else {
                            throw exception(sstream() <<
                                "Async constant: " << const_name(e));
                        }
                    }
                }
            } else {
                throw exception(sstream() <<
                    "proof contains unknown constant: " << const_name(e) << "\n"
                    << result);
            }
        }
        return true; // recurse
    });
  }

  auto tctx = mk_type_context_for(s);
  type_checker type_chk(old.env());
  buffer<expr> mvar_locals;
  for (auto & new_goal : get_refd_mvars_from_goals(s)) {
    ground_mvars(tctx, mvar_locals, new_goal);
  }
  for (auto & old_goal : get_refd_mvars_from_goals(old)) {
    if (!is_metavar(old_goal)) continue; // No need to check "proofs" of universe metavariables.
    auto mdecl = tctx.find_metavar_decl(old_goal);
    if (!mdecl) throw exception(sstream() << "undeclared metavar " << old_goal);
    auto lctx = mdecl->get_context();
    buffer<expr> locals;
    lctx.for_each([&] (auto & ldecl) { locals.push_back(ldecl.mk_ref()); });

    auto proof = mk_app(mk_lambda(name(), mdecl->get_type(), mk_var(0)), old_goal);
    proof = tctx.mk_lambda(lctx, locals, proof);
    for (unsigned i = mvar_locals.size(); i--;) {
      auto & mvar_local = mvar_locals[i];
      proof = mk_lambda(mlocal_name(mvar_local), mlocal_type(mvar_local),
            abstract_local(proof, mlocal_name(mvar_local)));
    }
    proof = tctx.instantiate_mvars(proof);
    ground_uvars(tctx, proof);
    proof = tctx.instantiate_mvars(proof);
    try {
      if (proof.has_local()) throw ext_exception("has local constant");
      if (has_metavar(proof)) throw ext_exception("has metavariable");
      type_chk.check(proof);
    } catch (ext_exception & e) {
      auto text = std::make_shared<string_output_channel>();
      io_state_stream out(tctx.env(), mk_pretty_formatter_factory()(tctx.env(), s.get_options(), tctx), text);
      out << "failed to type-check proof\n";
      out << proof << "\n";
      out << e;
      throw exception(text->str());
    }
  }
}

//legacy
void check_tactic_state(tactic_state const & s, expr_set const & split_goals, bool with_type_check) {
  auto mctx = s.mctx();
  auto result = mctx.instantiate_mvars(s.main());

  // for each goals: 
  //   - count if there are repeated hyps
  //   - mark referenced_mvars
  expr_set refd_mvars;
  for (auto & goal : s.goals()){
    for (auto & mvar : referenced_mvars(s.mctx(), goal)){
      refd_mvars.insert(mvar);
    }
  }
  for_each(result, [&] (expr const & e, unsigned) {
      if (is_metavar(e)) {
          if (refd_mvars.count(e) == 0 && split_goals.count(e) == 0) {
              throw exception(sstream() << "proof contains metavariable " << mlocal_name(e) << "\n"
                  << set_goals(s, {e}).pp());
          }
          return false;
      }
      if (is_local(e)) {
          // Shouldn't happen except for the ml_server_splitted ones,
          // but I still see some false positives.
          return false;
      }
      if (is_sorry(e)) throw exception("proof contains sorry");
      if (is_rec_fn_macro(e)) throw exception("proof contains rec_fn_macro");
      if (is_constant(e)) {
          if (auto decl = s.env().find(const_name(e))) {
              if (!decl->is_trusted())
                  throw exception(sstream() <<
                      "proof contains meta constant: " << const_name(e));
          } else {
              throw exception(sstream() <<
                  "proof contains unknown constant: " << const_name(e));
          }
      }
      return true; // recurse
  });

  //attempt to type check
  if (with_type_check){
    auto type_context = mk_type_context_for(s);
  // set_option trace.check true for more details ? 
  // options opts     = s.get_options();
  // opts = opts.update(string_to_name("trace.check"), true);
  // scope_trace_env _(s.env(), opts, type_context);
    check(type_context, result, false);   // true or false here  ?
  }
}

// Turns a tactic state into a single expression
// such that the expression is alpha-equivalent iff the goals are.
//
// Concretely, the expression is as follows, where the final Prop is applied to
// the mvarᵢ for the goals only.  (The nᵢ are the number of hypotheses of the
// corresponding goal---this is needed to disambiguate `⊢ true → true` and `h : true ⊢ true`.)
//
//   λ mvar₁ : ∀ hyp₁ ..., tgt,
//   ...
//   λ mvarₙ : ∀ hyp₁ ..., tgt,
//   Prop (mvarᵢ nᵢ) ...
expr goals_to_expr(tactic_state const & ts) {
  auto mctx = ts.mctx();
  buffer<expr> mvar_locals;
  auto res = mk_Prop();
  for (auto & goal : ts.goals()) {
    auto & decl = mctx.get_metavar_decl(goal);
    auto tc = mk_type_context_for(ts, decl.get_context());
    tc.set_mctx(mctx);
    ground_mvars(tc, mvar_locals, goal);
    unsigned num_hyps = 0;
    decl.get_context().for_each([&] (auto &) { num_hyps++; });
    res = mk_app(res, mk_app(get_app_fn(tc.instantiate_mvars(goal)), to_nat_expr(mpz(num_hyps))));
    mctx = tc.mctx();
  }
  for (unsigned i = mvar_locals.size(); i--;) {
    auto & mvar_local = mvar_locals[i];
    res = mk_lambda(mlocal_name(mvar_local), mlocal_type(mvar_local),
          abstract_local(res, mlocal_name(mvar_local)));
  }
  return mctx.instantiate_mvars(res);
}

// Super hacky way to expose private fields.
class metavar_context_exposed {
public:
    name_map<metavar_decl>    m_decls;
    name_map<level>           m_uassignment;
    name_map<expr>            m_eassignment;
};

uint64_t rough_mctx_size(metavar_context const & mctx) {
  metavar_context_exposed const * mctx2 = reinterpret_cast<metavar_context_exposed const *>(&mctx);
  uint64_t size = 0;
  mctx2->m_eassignment.for_each([&size] (name const &, expr const & e) {
      size += get_weight(e);
  });
  return size;
}
uint64_t rough_tactic_state_size(tactic_state const & s) {
  return rough_mctx_size(s.mctx());
}

json pp_ts(
    std::shared_ptr<mt_accounting> acct,
    std::shared_ptr<session> this_session,
    tactic_state& ts,
    unsigned int n_metavars,
    unsigned int max_size,
    unsigned int max_subgoals,
    unsigned int max_metavars
    ){
    auto ts_size = rough_tactic_state_size(ts);
    if (ts_size > max_size) {
        throw std::runtime_error("Big subgoal");
    }
    if (length(ts.goals()) > max_subgoals) {
        throw std::runtime_error("Too many subgoals");
    }
    if (n_metavars > max_metavars) {
        throw std::runtime_error("Too many metavars");
    }
    json this_goal = json::object();
    this_goal["size"] = ts_size;
    this_goal["n_subgoals"] = length(ts.goals());
    this_goal["n_metavars"] = n_metavars;

    auto goal_to_match = goals_to_expr(ts);

    auto lock_start = chrono::steady_clock::now();
    std::lock_guard<std::mutex> sessions_lock(this_session->session_mutex);
    acct->waiting_for_lock += chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - lock_start).count();

    bool has_match = false;
    unsigned final_node_id = this_session->nodes.size();
    if (this_session->merge_alpha_equiv){
        auto found = this_session->expr_to_id.find(goal_to_match);
        if (found != this_session->expr_to_id.end()){
            final_node_id = found->second;
            has_match = true;
        }
    }
    if (!has_match){
        this_session->expr_to_id[goal_to_match] = final_node_id;
        this_session->nodes.emplace_back(std::make_shared<session_node>(ts, this_session->print_ts(ts)));
    }
    this_goal["full_pp"] = this_session->nodes[final_node_id]->pp_ts;
    this_goal["node_id"] = final_node_id;
    return this_goal;
}

std::shared_ptr<session> get_session(std::shared_ptr<mt_accounting> acct, std::string sess_name) {
    auto lock_start = chrono::steady_clock::now();
    std::unique_lock<std::mutex> sess_lock(acct->sessions_mutex);
    acct->waiting_for_lock += (chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - lock_start)).count();
    auto maybe_session = acct->sessions.find(sess_name);
    auto session_found = maybe_session != acct->sessions.end();
    sess_lock.unlock();
    if (!session_found){
        throw std::runtime_error("Unknown session");
    }
    return maybe_session->second;
}

json get_and_emplace_nodes(
    std::shared_ptr<mt_accounting> acct,
    std::shared_ptr<session> this_session,
    std::vector<expr_set>& subgoal_metavars,
    list<tactic_state>& split_res,
    unsigned int max_size,
    unsigned int max_subgoals,
    unsigned int max_metavars,
    bool nosplit
){
    unsigned i = 0;
    json nodes = json::array();
    if (nosplit && length(head(split_res).goals()) == 0) return nodes;
    for (auto res_ts : split_res) {
        json this_goal = pp_ts(acct, this_session, res_ts, subgoal_metavars[i].size(), max_size, max_subgoals, max_metavars);
        nodes.push_back(this_goal);
        i++;
    }
    return nodes;
}


unsigned int count_repeated_hyps(tactic_state& ts){
    auto mctx = ts.mctx();
    unsigned int n_repeated = 0;
    for (auto & goal : ts.goals()){
        metavar_decl decl = mctx.get_metavar_decl(goal);
        local_context lctx = decl.get_context();
        name_set hyp_names;
        lctx.for_each([&](local_decl const & d) {
            name n = d.get_pp_name();
            if (hyp_names.contains(n)) n_repeated++;
            hyp_names.insert(n);
        });
    }
    return n_repeated;
}

double timestamp(){
    return std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
}

json apply_tactic(
    std::shared_ptr<mt_accounting> acct,
    std::string sess_name,
    size_t state_id,
    const std::string& tactic_str,
    unsigned int max_milliseconds,
    unsigned int max_size,
    unsigned int max_subgoals,        // maximum number of subgoals in a tactic child
    unsigned int max_metavars,        // maximum number of meta variables in a tactic child
    unsigned int max_repeated_hyps,   // maximum number of repeated hyps in the resulting tactic state
    bool nosplit
) {
    auto this_session = get_session(acct, sess_name);
    std::unique_lock<std::mutex> this_sess_lock(this_session->session_mutex);
    if (state_id >= this_session->nodes.size()) {
        throw std::runtime_error("Unknown State");
    }
    auto cur_node = this_session->nodes[state_id];
    this_sess_lock.unlock();

    std::vector<expr_set> subgoal_metavars;
    list<tactic_state> split_res;
    cancellation_token ctok = mk_cancellation_token(global_cancellation_token());
    auto eval_start = chrono::steady_clock::now();
    g_cancel_sched->cancel_after(chrono::milliseconds(max_milliseconds), ctok);
    unsigned int n_repeated_hyps = 0;
    try {
        scope_cancellation_token scope1(&ctok);
        auto ts = apply(cur_node->ts, *this_session->m_snapshot, tactic_str);

        n_repeated_hyps = count_repeated_hyps(ts);
        if (n_repeated_hyps > max_repeated_hyps) {
            throw std::runtime_error("Too many repeated hypothesis names.");
        }

        check_tactic_state2(cur_node->ts, ts);
        if (!nosplit) {
            split_res = split_tactic_state(ts, subgoal_metavars);
            assert (subgoal_metavars.size() == length(split_res));
        } else {
            expr_set refd_mvars;
            for (auto & goal : ts.goals()){
                for (auto & mvar : referenced_mvars(ts.mctx(), goal)){
                refd_mvars.insert(mvar);
                }
            }
            split_res.emplace_front(ts);
            subgoal_metavars.push_back(refd_mvars);
        }

    } catch (lean::interrupted &) {
        throw std::runtime_error("tactic timeout");
    }
    json nodes = get_and_emplace_nodes(acct, this_session, subgoal_metavars, split_res, max_size, max_subgoals, max_metavars, nosplit);
    unsigned int eval_time_ms = (chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - eval_start)).count();
    auto result = json{{"nodes", nodes}, {"eval_time", eval_time_ms}, {"repeated_hyps", n_repeated_hyps}};
    return result;
}

json parse_goal(std::shared_ptr<mt_accounting> const & acct, std::string const & sess_name, json parsed_goals, unsigned int max_milliseconds, unsigned int max_repeated_hyps, bool as_goal) {
    cancellation_token ctok = mk_cancellation_token(global_cancellation_token());
    g_cancel_sched->cancel_after(chrono::milliseconds(max_milliseconds), ctok);
    auto eval_start = chrono::steady_clock::now();
    try{
        scope_cancellation_token scope1(&ctok);
        json result = json::object();

        auto this_session = get_session(acct, sess_name);

        expr_set empty_expr_set;
        auto res_ts = parse_goal(*this_session->m_snapshot, parsed_goals, as_goal);

        if (count_repeated_hyps(res_ts) > max_repeated_hyps) {
            throw std::runtime_error("Too many repeated hypothesis names.");
        }

        std::vector<expr_set> subgoal_metavars;
        auto split_res = split_tactic_state(res_ts, subgoal_metavars);
        json nodes = get_and_emplace_nodes(acct, this_session, subgoal_metavars, split_res, 1e9, 1e9, 1e9, false);
        result["nodes"] = nodes;
        unsigned int eval_time_ms = (chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - eval_start)).count();
        

        expr_set refd_mvars;
        for (auto & goal : res_ts.goals()){
            for (auto & mvar : referenced_mvars(res_ts.mctx(), goal)){
                refd_mvars.insert(mvar);
            }
        }

        json full_node = pp_ts(acct, this_session, res_ts, refd_mvars.size(), 1e9, 1e9, 1e9);
        result["eval_time"] = eval_time_ms;
        result["n_subgoals"] = length(res_ts.goals());
        result["node_id"] = full_node["node_id"];
        result["full_pp"] = full_node["full_pp"];

        return result;
    } catch (lean::interrupted &) {
        throw std::runtime_error(as_goal ? "parse goal timeout" : "parse command timeout");
    }
}

std::tuple<json, tactic_state> parse_goal_and_check_not_multigoal(std::shared_ptr<mt_accounting> acct, std::shared_ptr<session> this_session, json preparsed_goals, unsigned int max_metavars){
    try {
        auto eval_start = chrono::steady_clock::now();
        
        if (preparsed_goals.size() != 1) {
            throw std::runtime_error("multigoal");
        }

        auto parsed_ts = parse_goal(*this_session->m_snapshot, preparsed_goals, true);

        std::vector<expr_set> subgoal_metavars;
        auto splitted = split_tactic_state(parsed_ts, subgoal_metavars);
        
        if (length(splitted) != 1) {
            throw std::runtime_error("multigoal");
        }
        auto first_parsed_ts = head(splitted);
        auto first_metavars = subgoal_metavars[0];

        json parsed_goal = pp_ts(acct, this_session, first_parsed_ts, first_metavars.size(), 1e9, 1e9, max_metavars);
        unsigned int eval_time_ms = (chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - eval_start)).count();
        parsed_goal["parse_time"] = eval_time_ms;

        return {parsed_goal, first_parsed_ts};
    } catch (std::exception & ex) {
        // Add "parse_goal_error:" to error msg to differentiate parse_goal and apply_tactic errors
        throw std::runtime_error(std::string("parse_goal_error: ") + ex.what());
    }
}

json parse_goal_and_apply_tactic(
    std::shared_ptr<mt_accounting> const & acct,
    std::string const & sess_name,
    json parsed_goals,
    const std::string& tactic_str,
    unsigned int max_milliseconds,
    unsigned int max_size,
    unsigned int max_subgoals,  // maximum number of subgoals in a tactic child
    unsigned int max_metavars,   // maximum number of meta variables in a tactic child
    unsigned int max_repeated_hyps,
    bool strip_tags
) {
    cancellation_token ctok = mk_cancellation_token(global_cancellation_token());
    g_cancel_sched->cancel_after(chrono::milliseconds(max_milliseconds), ctok);
    auto eval_start = chrono::steady_clock::now();
    try{
        scope_cancellation_token scope1(&ctok);
        json result = json::object();

        auto this_session = get_session(acct, sess_name);
        auto [parsed_goal, first_parsed_ts] = parse_goal_and_check_not_multigoal(acct, this_session, parsed_goals, max_metavars);
        if (count_repeated_hyps(first_parsed_ts) > max_repeated_hyps) {
            throw std::runtime_error("Too many repeated hypothesis names.");
        }
        result["parsed_goal"] = parsed_goal;

        auto res_ts = optional<tactic_state>(apply(first_parsed_ts, *this_session->m_snapshot, tactic_str));
        if (strip_tags){
            res_ts->raw()->m_tag_info.m_tags.clear();
        }
        expr_set empty_expr_set;
        check_tactic_state2(first_parsed_ts, *res_ts);
        
        if (count_repeated_hyps(*res_ts) > max_repeated_hyps) {
            throw std::runtime_error("Too many repeated hypothesis names.");
        }

        json subgoals = json::array();
        std::vector<expr_set> subgoal_metavars;
        unsigned i = 0;
        for (auto this_ts : split_tactic_state(*res_ts, subgoal_metavars)) {
            json this_goal = pp_ts(acct, this_session, this_ts, subgoal_metavars[i].size(), max_size, max_subgoals, max_metavars);
            subgoals.push_back(this_goal);
            i++;
        }
        result["subgoals"] = subgoals;
        unsigned int eval_time_ms = (chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - eval_start)).count();
        result["eval_time"] = eval_time_ms;
        return result;
    } catch (lean::interrupted &) {
        throw std::runtime_error("parse goal timeout");
    }
}

json parse_children(
    std::shared_ptr<mt_accounting> const & acct,
    std::string const & sess_name,
    json preparsed_children,
    unsigned int max_milliseconds,
    unsigned int max_metavars,   // maximum number of meta variables in a tactic child) 
    unsigned int max_repeated_hyps
) {
    cancellation_token ctok = mk_cancellation_token(global_cancellation_token());
    g_cancel_sched->cancel_after(chrono::milliseconds(max_milliseconds), ctok);
    auto eval_start = chrono::steady_clock::now();
    try{
        scope_cancellation_token scope1(&ctok);
        json result = json::object();
        auto this_session = get_session(acct, sess_name);
        // early exit if already multi goals
        for (auto & c : preparsed_children) {
            if (c.size() != 1) {
                throw std::runtime_error("parse_goal_error: multigoal");
            }
        }

        json parsed_children = json::array();
        for (auto & c : preparsed_children) {
            auto [parsed_child, ts] = parse_goal_and_check_not_multigoal(acct, this_session, c, max_metavars);
            if (count_repeated_hyps(ts) > max_repeated_hyps) {
                throw std::runtime_error("Too many repeated hypothesis names.");
            }
            parsed_children.push_back(parsed_child);
        }

        unsigned int eval_time_ms = (chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - eval_start)).count();
        result["parsed_children"] = parsed_children;
        result["eval_time"] = eval_time_ms;
        return result;
    } catch (lean::interrupted &) {
        throw std::runtime_error("parse goal timeout");
    }
}

tactic_state apply(tactic_state& some_ts, snapshot const & snapshot, const std::string& tactic_str) {
    // for easy access in gdb
    std::string pp_ts = (sstream() << some_ts.pp()).str();

    // necessary for match statements to work
    // otherwise fails with "failed to register private name '_match_1', prefix has not been registered"
    declaration_info_scope dis(some_ts.env(), {}, {});

    // parse
    std::istringstream input_stream("`[" + tactic_str + "]");
    auto some_parser = parser(snapshot.m_env, get_global_ios(), mk_dummy_loader(), input_stream, "dummy file");
    some_parser.from_snapshot(snapshot);
    some_parser.scan();

    auto pre_expr = (some_parser.no_error_recovery_scope(), some_parser.parse_expr());

    //elaborate (from tactic_to_expr)
    optional<metavar_decl> g = some_ts.get_main_goal_decl();
    elaborator elab(mk_type_context_for(some_ts), some_ts.get_options(), some_ts.decl_name(), false /* recover_from_errors */);
    expr parsed_expr = elab.elaborate(resolve_names(some_ts.env(), g->get_context(), pre_expr));
    auto elab_mctx = elab.mctx();
    parsed_expr = elab_mctx.instantiate_mvars(parsed_expr);

    //evaluate `[...] tactic into a tactic unit
    vm_state S(some_ts.env(), some_ts.get_options());
    metavar_context s_mctx = some_ts.mctx();
    parsed_expr = s_mctx.instantiate_mvars(parsed_expr);
    environment aux_env = S.env();
    name eval_aux_name = mk_unused_name(aux_env, "_eval_expr");
    expr expected_type = mk_tactic_unit();
    auto cd = check(aux_env, mk_definition(aux_env, eval_aux_name, {}, expected_type, parsed_expr, true, false));
    auto declaration = cd.get_declaration();
    expr evaled_expr = declaration.get_value();

    // run the tactic unit on the current tactic_state
    auto type_context = mk_type_context_for(some_ts);
    auto evaluator = tactic::evaluator(type_context, some_ts.get_options(), false);
    vm_obj r = evaluator(evaled_expr, some_ts);
    auto maybe_ts = tactic::is_success(r);
    if (maybe_ts) {
        return maybe_ts.value();
    }
    throw std::runtime_error("apply tactic failed");
}

output_queue::output_queue(std::ostream & out, std::shared_ptr<std::atomic<int>> r) : m_out(out), req_in_queue(r) {
    m_out_thread.reset(new lthread([&] {
        std::unique_lock<std::mutex> lock(m_out_mutex);
        while (!m_out_finished || !m_to_print.empty()) {
            while (!m_to_print.empty()) {
                auto str = m_to_print.front();
                m_to_print.pop();
                lock.unlock();
                m_out << str << std::endl;
                if (req_in_queue){
                    (*req_in_queue)--;
                }
                lock.lock();
            }
            m_out_cv.wait_for(lock, 100ms);
        }
    }));
}

void output_queue::send(std::string const & s) {
    send(s, std::shared_ptr<std::atomic<int>>(nullptr));
}

void output_queue::send(std::string const & s, std::shared_ptr<std::atomic<int>> r) {
    if (r) (*r)++;
    (std::unique_lock<std::mutex>(m_out_mutex), m_to_print.push(s));
    m_out_cv.notify_one();
}

output_queue::~output_queue() {
    (std::unique_lock<std::mutex>(m_out_mutex), m_out_finished = true);
    m_out_thread->join();
}


void ml_server::setup_handlers(unsigned num_threads){
    m_ios.set_regular_channel(std::make_shared<stderr_channel>());
    m_ios.set_diagnostic_channel(std::make_shared<stderr_channel>());

    scope_global_ios scoped_ios(m_ios);

    if (num_threads == 0) {
        m_tq.reset(new st_task_queue);
    } else {
        m_tq.reset(new mt_task_queue(num_threads));
    }
    set_task_queue(m_tq.get());

    m_lt.add_listener([&] (std::vector<log_tree::event> const & evs) {
        for (auto & ev : evs)
            if (auto prod = ev.m_node.get_producer())
                taskq().submit(prod);
    });
}


void ml_server::clear_handlers(){
    m_tq.reset(nullptr);
    m_lt.clear_listeners();
}

ml_server::ml_server(unsigned num_threads, search_path const & path, environment const & initial_env, io_state const & ios, bool fast_start) :
    m_path(path), m_initial_env(initial_env), m_ios(ios),
    handling(std::make_shared<std::atomic<int>>(0)), req_in_queue(std::make_shared<std::atomic<int>>(0)),
    acct(new mt_accounting) {

    setup_handlers(num_threads);

    m_mod_mgr.reset(new module_mgr(this, m_lt.get_root(), m_path, m_initial_env, m_ios));
    m_mod_mgr->set_use_old_oleans(true);
    m_mod_mgr->set_report_widgets(false);
    m_mod_mgr->set_server_mode(true);
    m_mod_mgr->set_save_info(false);
    m_mod_mgr->set_save_olean(false);
    set_global_module_mgr(*m_mod_mgr);

    if (!fast_start) pre_load(path);
}

ml_server::~ml_server() {}

std::shared_ptr<module_info> ml_server::load_module(module_id const & id, bool can_use_olean) {
    bool never_use_olean = true;
    return m_fs_vfs.load_module(id, can_use_olean && !never_use_olean);
}

void ml_server::run(search_path const & paths) {
    // after checkpoint is done, preload others and proof_cleaning
    get_global_module_mgr()->add_paths(paths);
    std::vector<std::string> all_new_paths;
    for (auto & root : paths) {
        std::cerr << "[ML Server] Preloading " << root << std::endl;
        const std::string ext = ".lean";
        for (const auto& dir_entry : std::filesystem::recursive_directory_iterator(root)){
            std::string module_path = dir_entry.path().c_str();
            if (module_path.length() >= ext.length()){
                if (0 == module_path.compare(module_path.length() - ext.length(), ext.length(), ext)){
                    all_new_paths.push_back(module_path);
                }
            }
        }
    }

    // Start parsing.
    for (auto & mod_path : all_new_paths)
        m_mod_mgr->get_module(mod_path);

    // Wait for everything to be done by getting all snapshots.
    for (auto & mod_path : all_new_paths) {
        auto mod = m_mod_mgr->get_module(mod_path);
        auto res = mod->m_snapshots;
        while (res && res->m_next) {
            res = get(res->m_next);
        }
    }

    acct->waiting_for_lock.store(0);
    acct->total_processing.store(0);

    std::string req_string;
    auto cancel_sched = std::make_unique<cancellation_scheduler>();
    g_cancel_sched = cancel_sched.get();
    
    m_stdout = std::make_shared<output_queue>(std::cout, req_in_queue);
    acct->m_stderr = std::make_shared<output_queue>(std::cerr, std::shared_ptr<std::atomic<int>>(nullptr)); 

    lean::tp next_log = chrono::steady_clock::now() + chrono::seconds(30);
    while (true) {
        lean::tp now = chrono::steady_clock::now();
        try {
            if (now >= next_log) {
                double ratio_lock = (long double)(acct->waiting_for_lock.load()) /  (long double)(acct->total_processing.load());
                json cpp_stats{{"handling", handling->load()}, {"req_in_queue", req_in_queue->load()}, {"ratio_lock", ratio_lock}};
                acct->m_stderr->send(sstream() << "[ML Server stats] " << cpp_stats);
                next_log += chrono::seconds(30);
            }
            std::getline(std::cin, req_string);
            if (std::cin.eof()) return;
            json req = json::parse(req_string);
            std::string req_id = req["req_id"];
            handle_async_request(req);

        } catch (std::exception & ex) {
            json exception;
            exception["error"] = ex.what();
            acct->m_stderr->send(sstream() << "[ML Server - main loop] " << exception << "\n" << "input: " << req_string);
            m_stdout->send(exception, req_in_queue);
        }
    }
}

void ml_server::handle_async_request(json const & req) {
    std::string req_id = req["req_id"];
    auto fn = [a=acct, s=m_snapshot_cmd, h=handling, iq=req_in_queue, stdout=m_stdout, req_id, req] {
        auto process_start = chrono::steady_clock::now();
        log_tree lt;
        scope_log_tree _(lt.get_root());
        (*h)++;
        std::string ret_type;
        json res;
        bool success = false;
        try {
            if (a->profile_tactic_process){
                a->m_stderr->send(sstream() << json{{"req_id", req_id}, {"ts", timestamp()}, {"type", "begin"}});
            }
            res = handle_ml_request(a, req);
            ret_type =std::string("finish_ok");
            success = true;
        } catch (std::exception & ex) {
            res = json{{"error", ex.what()}};
            ret_type = std::string("finish_err");
        } catch (...) {
            res = json{{"error", "Unknown exception"}};
            ret_type = std::string("finish_err");
        }
        if (!success) {
            a->m_stderr->send(sstream() << "[ML Server] " << res);
        }
        res["req_id"] = req_id;
        res["thread_id"] = (sstream() << std::this_thread::get_id()).str();
        if (a->profile_tactic_process){
            a->m_stderr->send(sstream() << json{{"req_id", req_id}, {"ts", timestamp()}, {"type", ret_type}});
        }
        stdout->send(res, iq);
        (*h)--;
        a->total_processing += (chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - process_start)).count();
        return unit{};
    };

    auto task = task_builder<unit>(std::move(fn))
        .wrap(library_scopes({}))
        .build();
    taskq().submit(task);
}


json new_session(std::shared_ptr<mt_accounting> acct, json const & req){
    std::string module_path = req.at("module_path");
    std::string decl_name = req.at("decl_name");
    bool merge_alpha_equiv = req.at("merge_alpha_equiv");
    json opts = req.at("opts");


    auto mod = get_global_module_mgr()->get_module(module_path);
    auto res = mod->m_snapshots;
    optional<module_parser_result> next_res;
    auto some_name = string_to_name(decl_name);
    bool found = false;
    while (res && res->m_next) {
        next_res = get(res->m_next);
        if (next_res->m_snapshot_at_end->m_env.find(some_name).has_value()) {
            found = true;
            break; // next_res is first snapshot after decl, res is last snapshot before decl
        }
        res = next_res;
    }
    if (!found) {
        throw std::runtime_error((sstream() << "decl not found " << some_name << " " << module_path).str());
    }
    auto snapshot_after = std::make_shared<lean::snapshot>(*next_res.value().m_snapshot_at_end);
    auto snapshot_before = std::make_shared<lean::snapshot>(*res.value().m_snapshot_at_end);

    // TODO: just ignore on the lean side
    auto add_level = [&] (name const & n) { snapshot_before->m_lds.insert(n, mk_param_univ(n)); };
    add_level("u");
    add_level("v");
    add_level("w");
    for (int i = 0; i < 16; i++) add_level((sstream() << "u_" << i).str());

    auto decl = snapshot_after->m_env.get(some_name);
    auto some_local_context = local_context();
    auto ts = mk_tactic_state_for(
        snapshot_before->m_env, options(), some_name, some_local_context, decl.get_type()
    );

    while (true) {
        auto maybe_ts = intron(1, ts, false);
        if (!maybe_ts) break;
        ts = maybe_ts.value();
    }
    json response;

    
    std::mt19937 generator{std::random_device{}()};
    std::uniform_int_distribution<int> distribution{'a', 'z'};
    std::string rand_str(6, '\0');
    bool found_new_name = false;
    while (!found_new_name){
        // generate random string
        for(auto& dis: rand_str) dis = distribution(generator);
        //if unused, create the session
        {
            auto lock_start = chrono::steady_clock::now();
            std::lock_guard<std::mutex> sessions_lock(acct->sessions_mutex);
            acct->waiting_for_lock += (chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - lock_start)).count();
            found_new_name = acct->sessions.find(rand_str) == acct->sessions.end();
            if (found_new_name){
                acct->m_stderr->send(sstream() << "[ML Server] " << "Loading -- " << req);
                acct->sessions.emplace(
                    rand_str,
                    std::make_shared<session>(
                        decl_name,
                        snapshot_before,
                        merge_alpha_equiv,
                        opts
                    )
                );
            }
        }
    }
    response["name"] = rand_str;
    response["initial_goal"] = pp_ts(acct, acct->sessions.find(rand_str)->second, ts, 0, 1e9, 1e9, 1e9);
    return response;
}

json del_session(std::shared_ptr<mt_accounting> acct, json const & req){
    std::string session_name = req.at("name");
    std::lock_guard<std::mutex> sess_lock(acct->sessions_mutex);
    if (acct->sessions.erase(session_name) == 1){
        return json{{"erased", 1}};
    } else {
        return json{{"error", "No such session"}};
    }
}

json eval_cmd(std::string to_run, std::string module_path, unsigned int max_milliseconds) {
    auto mod = get_global_module_mgr()->get_module(module_path);
    // Grab last snapshot in file
    auto res = mod->m_snapshots;
    optional<module_parser_result> next_res;
    while (res && res->m_next) {
        next_res = get(res->m_next);
        res = next_res;
    }
    auto s = std::make_shared<lean::snapshot>(*res.value().m_snapshot_at_end);
    
    cancellation_token ctok = mk_cancellation_token(global_cancellation_token());
    g_cancel_sched->cancel_after(chrono::milliseconds(max_milliseconds), ctok);
    try {
        scope_cancellation_token scope1(&ctok);
        std::istringstream input_stream(to_run);

        parser p(s->m_env, get_global_ios(), mk_dummy_loader(), input_stream, "dummy file");
        p.from_snapshot(*s);
        p.scan();
        {
            lean::lazy_type_context tc(s->m_env, p.get_options());
            // scope_global_ios scope1(m_ios);
            scope_trace_env  scope2(s->m_env, get_global_ios().get_options(), tc);
            scope_traces_as_string traces_as_string;
            ast_id cmd_id = 0;
            auto e = lean::eval_cmd(p, cmd_id);

            json to_return{{"output", traces_as_string.get_string()}};
            return to_return;
        }
    } catch(lean::interrupted& e){
        return json{{"error", "eval_cmd timeout"}};
    }
}


json handle_ml_request(std::shared_ptr<mt_accounting> acct, json const & jreq) {
    std::string req_type = jreq.at("req_type");
    if (req_type == "new_session") {
        return new_session(acct, jreq);
    } else if (req_type == "tactic") {
        return apply_tactic(
            acct,
            jreq.at("name"),
            jreq.at("state_id"),
            jreq.at("tactic_str"),
            jreq.at("timeout"),
            jreq.at("max_size"),
            jreq.at("max_subgoals"),
            jreq.at("max_metavars"),
            jreq.at("max_repeated_hyps"),
            jreq.at("nosplit")
        );
    } else if (req_type == "parse_goal") {
        return parse_goal(acct, jreq.at("name"), jreq.at("parsed_goals"), jreq.at("timeout"), jreq.at("max_repeated_hyps"), true);
    } else if (req_type == "parse_command") {
        return parse_goal(acct, jreq.at("name"), jreq.at("parsed_goals"), jreq.at("timeout"), jreq.at("max_repeated_hyps"), false);
    } else if (req_type == "parse_goal_and_apply_tactic") {
        return parse_goal_and_apply_tactic(
            acct,
            jreq.at("name"), jreq.at("parsed_goals"), jreq.at("tactic_str"),
            jreq.at("timeout"), jreq.at("max_size"), jreq.at("max_subgoals"), jreq.at("max_metavars"),
            jreq.at("max_repeated_hyps"), jreq.at("strip_tags")
        );
    } else if (req_type == "parse_children") {
        return parse_children(acct, jreq.at("name"), jreq.at("preparsed_children"), jreq.at("timeout"), jreq.at("max_metavars"), jreq.at("max_repeated_hyps"));
    } else if (req_type == "del_session") {
        return del_session(acct, jreq);
    } else if (req_type == "eval_cmd") {
        return eval_cmd(jreq.at("to_run"), jreq.at("module_path"), jreq.at("timeout"));
    } else {
        json response = json::object();
        response["error"] = "Unknown req_type " + req_type;
        return response;
    }
}


}
