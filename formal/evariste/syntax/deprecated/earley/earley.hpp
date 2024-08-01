// Copyright (c) 2019-present, Facebook, Inc.
// All rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef EARLEY
#define EARLEY

#include <vector>
#include <utility>
#include <algorithm>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>

using namespace std;

int display(std::vector<int>::iterator it, const string& s, int depth=0) {
    auto tab = string(depth, '\t');
    cout << tab << s.substr(*it, *(it + 1)) << endl;
    int next = 3;
    for (int i = 0; i < *(it+ 2); i++){
        next += display(it + next, s, depth + 1);
    }
    return next;
}

class SimpleParseTree {
public:
    SimpleParseTree* parent;
    int nt_id, rule_id, pos_in_input;
    vector<SimpleParseTree*> children;
    SimpleParseTree(int nt_id, int rule_id, int pos_in_input, vector<SimpleParseTree*>& refs):
        nt_id(nt_id), rule_id(rule_id), pos_in_input(pos_in_input) {
        refs.push_back(this);
    };
    void add_child(SimpleParseTree* child) {
        child->parent = this;
        children.push_back(child);
    };
    void display(int depth = 0) {
        for(int i = 0 ; i < depth; i ++ ){
            cout << "\t";
        }
        cout << nt_id << " " << rule_id << " " << pos_in_input << endl;
        for (auto c: children) {
            c->display(depth + 1);
        }
    };

    int serialize(int pos, vector<int>& result, vector<pair<int, int>>& rules, vector<int>& lengths) {
        int id_begin = result.size();
        if (rule_id == -1) {
            // terminal
            return lengths[pos_in_input];
        } else {
                rules.push_back(pair<int,int>(nt_id, rule_id));
            result.push_back(pos);
            result.push_back(-1); // place holder
            result.push_back(children.size());
            int this_length = 0;
            int n_children = 0;
            int prev_res_size = result.size();
            for (auto c: children) {
                this_length += c->serialize(pos + this_length, result, rules, lengths) + 1;
                if (result.size() > size_t(prev_res_size)) {
                    prev_res_size = result.size();
                    n_children += 1;
                }
            }
            result[id_begin + 1] = this_length -  1;
            result[id_begin + 2] = n_children;
        }
        return result[id_begin + 1];
    }
};

class EarleyItem {
public:
    int nt_id, rule_id; // rule identifier
    int pos_in_input, pos_in_rule;
    SimpleParseTree* tree;

    EarleyItem(int nt_id, int rule_id, int pos_in_input, int pos_in_rule, vector<SimpleParseTree*>& refs):
        nt_id(nt_id), rule_id(rule_id),
        pos_in_input(pos_in_input), pos_in_rule(pos_in_rule){
        tree = new SimpleParseTree(nt_id, rule_id, pos_in_input,refs);
    };

    bool operator==(const EarleyItem& o) const
    {
        return nt_id == o.nt_id && rule_id == o.rule_id &&
            pos_in_input == o.pos_in_input && pos_in_rule == o.pos_in_rule;
    }
    EarleyItem advance(vector<SimpleParseTree*>& refs) {
        auto to_ret = EarleyItem(nt_id, rule_id, pos_in_input, pos_in_rule + 1, refs);
        to_ret.tree = new SimpleParseTree(nt_id, rule_id, pos_in_input, refs);
        to_ret.tree->parent = tree->parent;
        for (auto child: tree->children) {
            to_ret.tree->add_child(child);
        }
        return to_ret;
    }

};

class EarleyItemHash {
    public:
    size_t operator()(const EarleyItem& e) const
    {
        size_t res = 0;
        res ^= e.nt_id + 0x9e3779b9 + (res<<6) + (res>>2);
        res ^= e.rule_id + 0x9e3779b9 + (res<<6) + (res>>2);
        res ^= e.pos_in_input + 0x9e3779b9 + (res<<6) + (res>>2);
        res ^= e.pos_in_rule + 0x9e3779b9 + (res<<6) + (res>>2);
        return res;
    }
};

ostream& operator <<(ostream& os, const EarleyItem& item)
{
    os << item.nt_id << " " << item.rule_id << " " << item.pos_in_rule << " " << item.pos_in_input << endl;
    return os;
}

class Grammar {
public:
    unordered_map<string, int> word2tok;
    vector<string> tok2word;
    unordered_map<string, string> vars;

    unordered_set<int> terminals;
    unordered_set<int> non_terminals;

    unordered_map<int, vector<vector<int>>> rules;  // non_terminal -> list of rules
    unordered_map<int, vector<string>> rule_names;  // rule names if any

    int get(const string& s) {
        if (word2tok.find(s) == word2tok.end()){
            int this_id = tok2word.size();
            word2tok[s] = this_id;
            tok2word.push_back(s);
            if (s[0] == '$') {
                rules[this_id] = vector<vector<int>>();
                non_terminals.insert(this_id);
            } else {
                terminals.insert(this_id);
            }
            return this_id;
        } else {
            return word2tok[s];
        }
    }

    Grammar(const string& input_file){
        ifstream infile(input_file);
        if (infile.is_open()) {
            string line;
            while (getline(infile, line))
            {
                istringstream ss(line);
                string first_word;
                ss >> first_word;
                if (first_word[0] == '$') {
                    // add non-terminal to rule names if needed
                    auto rule_id = get(first_word);

                    // parse rule
                    vector<int> this_rule;
                    bool has_name = false;
                    while (ss) {
                        string word;
                        ss >> word;
                        if (!word.empty()) {
                            if (word[0] == '$' && word[1] == '*') {
                                rule_names[rule_id].push_back(word.substr(2));
                                has_name = true;
                            } else {
                                this_rule.push_back(get(word));
                            }
                        }
                    }
                    if (!has_name){
                        rule_names[rule_id].push_back("*");
                    }

                    rules[rule_id].push_back(this_rule);
                }
                else if(first_word[0] == '#') {
                    while (ss) {
                        string word;
                        ss >> word;
                        if (!word.empty()) {
                            vars[word] = first_word;
                        }
                    }
                } else {
                    throw runtime_error("Unexpected non terminal in beginning of line.");
                }
            }
//            cout << "Read " << terminals.size() << " terminals in input." << endl;
//            cout << "Read " << non_terminals.size() << " non_terminals in input." << endl;
        } else {
            cout << "Couldn't open input file " << input_file << endl;
        }
    };

    vector<int> tokenize(const string& s, vector<int>& lengths) {
        vector<int> to_return;
        istringstream ss(s);
        while (ss) {
            string word;
            ss >> word;
            if (!word.empty()) {
                    lengths.push_back(word.size());
                if (vars.find(word) != vars.end()) {
                    to_return.push_back(get(vars[word]));
                } else {
                    to_return.push_back(get(word));
                }
            }
        }
        return to_return;
    }

    const vector<vector<int>>& get_rules(int r_id) {
        return rules[r_id];
    };

    int get_next_token(EarleyItem& item) {
        auto& rule = rules[item.nt_id][item.rule_id];
        if (size_t(item.pos_in_rule) < rule.size()) {
            return rule[item.pos_in_rule];
        }
        return -1;
    };

    bool is_done(EarleyItem& item) {
        return get_next_token(item) == -1;
    };

    pair<vector<int>, vector<string>> parse(const string& s) {
        vector<SimpleParseTree*> refs;
        vector<int> lengths;
        auto tokenized = tokenize(s, lengths);

        unordered_set<EarleyItem, EarleyItemHash> seen_items;
        vector<vector<EarleyItem>> queues;
        for(size_t i = 0; i <= tokenized.size(); i ++) {
            queues.push_back(vector<EarleyItem>());
        }

        // initialize the queues with _start expansions
        auto start_id = get("$start");
        for(size_t rule_id = 0; rule_id < get_rules(start_id).size(); rule_id++) {
            auto item = EarleyItem(start_id, rule_id, 0, 0, refs);
            queues[0].push_back(item);
            seen_items.insert(item);
        }

        for (size_t pos_in_input = 0; pos_in_input <= tokenized.size(); pos_in_input++) {
            if (pos_in_input > 0) {
                seen_items.clear();
            }
            // Scanner
            for (size_t item_id = 0; item_id < queues[pos_in_input].size(); item_id++){
                auto& item = queues[pos_in_input][item_id];

                int next_token = get_next_token(item);
                if (next_token > 0) {
                    // item is not finished
                    if (non_terminals.count(next_token) > 0) {
                        // item's next element is a non_terminal
                        for (size_t rule_id = 0; rule_id < rules[next_token].size(); rule_id++){
                            auto new_item = EarleyItem(next_token, rule_id, pos_in_input, 0, refs);
                            if (seen_items.find(new_item) == seen_items.end()) {
                                queues[pos_in_input].push_back(new_item);
                                seen_items.insert(new_item);
                            }
                        }
                    } else {
                        // item's next element is a terminal
                        if (pos_in_input < tokenized.size() && next_token == tokenized[pos_in_input]) {
                            auto new_item = item.advance(refs);
                            if (seen_items.find(new_item) == seen_items.end()) {
                                new_item.tree->add_child(new SimpleParseTree(next_token, -1, pos_in_input, refs));
                                queues[pos_in_input + 1].push_back(new_item);
                                seen_items.insert(new_item);
                            }
                        }
                    }
                } else {
                    int item_pos_input = item.pos_in_input;
                    for (auto& prev_item: queues[item_pos_input]) {
                        auto prev_next_token = get_next_token(prev_item);
                        if (prev_next_token == queues[pos_in_input][item_id].nt_id) {
                            auto new_item = prev_item.advance(refs);
                            new_item.tree->add_child(queues[pos_in_input][item_id].tree);
                            if (seen_items.find(new_item) == seen_items.end()) {
                                queues[pos_in_input].push_back(new_item);
                                seen_items.insert(new_item);
                            }
                        }
                    }
                }
            }
        }

        // If we have a done EarleyItem (_start, ..., 0, end_of_rule) in queues[n] then we are done
        for (auto& item: queues[tokenized.size()]) {
            if (item.nt_id == start_id && item.pos_in_input == 0 && is_done(item)){
                vector<int> serialized;
                vector<pair<int, int>> rules;
                item.tree->serialize(0, serialized, rules, lengths);

                vector<string> to_ret_rule_names;
                for(auto x : rules){
                    to_ret_rule_names.push_back(rule_names[x.first][x.second]);
                }
                for(auto x: refs) {
                    delete x;
                }
                return make_pair(serialized, to_ret_rule_names);
            }
        }
        for(auto x: refs) {
            delete x;
        }
        return make_pair(vector<int>(), vector<string>());
    };

};

#endif