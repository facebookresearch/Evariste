<template>
    <li :class="item_class" v-if="is_visible">
        <node v-if="node_type==='node'"
            :settings="settings"
            :is_folder="is_folder"
            :goal_data="node.goal_data"
            :n_children="n_children"
            :solved="solved || isSolved"
            :footer="footer"
            @is-clicked="clicked"
            class="hoverable">
        </node>
        <tactic v-else
            :settings="settings"
            :is_folder="is_folder"
            :tactic_data="node.tactic_data"
            :n_children="n_children"
            :n_solved="n_solved"
            :solving="n_solved === n_children || isSolving"
            :footer="footer"
            class="hoverable">
        </tactic>

        <span v-if="is_folder" @click="expand()" class="expand">[{{is_open ? '-' : '+' }}]</span>&nbsp;
        <!-- query the model ⚡⚙️🤖▶️🔻🔎↯ -->
        <span v-if="node_type === 'node'">
            <span class="queryButton" @click="queryModel()">⚡</span>
        </span>
<!--        <ul >-->
<!--        <p v-if="node_type === 'node'">-->
<!--            <component :is="auto_comp"></component>-->
<!--&lt;!&ndash;            <input id="human_tac" placeholder="2p2e4">&ndash;&gt;-->
<!--        </p>-->
<!--        </ul>-->
        <ul :class="item_class" v-show="is_open" v-if="is_folder">
            <!--              @add-tactic="$emit('add-tactic', ...arguments)"-->
            <!--              @tactic-solves="tacticSolves"-->
            <tree
                v-for="(child, index) in node.children"
                :key="index"
                :index="index"
                :node="child"
                :node_type="next_type"
                :settings="settings"
                :session_name="session_name"
                :footer="footer"
                @add-tactic="reemit_tac"
                @goal-is-solved="goalIsSolved(index)"
                @tactic-solves="tacticSolves"
                @node-clicked="reemit_clicked"
            ></tree>
        </ul>
    </li>
</template>

<script>

    import node from './node/Node.vue'
    import tactic from './tactic/Tactic.vue'
    import axios from "axios";
    import utils from "@/static/utils";
    // import mmautocomplete from './Autocomplete/MMAutocomplete.vue'

    export default {
        name: 'tree',
        data: function () {
            return {
                isOpen: false,
                tactic_hash: new Set(),
                solvedChildren: new Set(),
                n_solved: 0,
                solved: false,
            }
        },
        props: {
            footer: Object,
            node: Object,
            settings: Object,
            session_name: String,
            node_type: String,
            index: Number,
        },
        components: {
            node, tactic,
            // mmautocomplete
        },
        computed: {
            auto_comp: function() {
                return this.settings.language_str + 'autocomplete';
            },
            is_folder: function () {
                return this.node.children.length > 0;
            },
            is_open: function () {
                return this.isOpen;
            },
            item_class: function () {
                return this.node_type === 'node' ? 'item-tactic' : 'item-subgoal';
            },
            next_type: function () {
                return this.node_type === 'node' ? 'tactic' : 'node';
            },
            n_children: function () {
                return this.node.children.length;
            },
            is_visible: function () {
                if (this.node_type === 'node') {
                    if (this.node.goal_data === undefined)  // not MCTS or proof
                        return true;
                    let creation_time = this.node.goal_data.creation_time;
                    let depth = this.node.goal_data.depth;
                    let res = true;
                    if (creation_time !== null && creation_time !== undefined)
                        res = res && creation_time <= this.settings.current_t;
                    if (depth !== null && depth !== undefined)
                        res = res && depth <= this.settings.max_display_depth;
                    return res;
                } else {
                    let res = this.node.is_valid || this.settings.show_invalid;
                    let visits = this.node.tactic_data.visits;
                    if (visits !== undefined)
                        res = res && (visits > 0 || this.settings.show_zero_visits);
                    return res;
                }
            },
            isSolved: function() {
                return ('is_solved' in this.node && this.node.is_solved);
            },
            isSolving: function() {
                return ('is_solving' in this.node && this.node.is_solving);
            }
        },
        methods: {
            clicked: function(){
                this.$emit('node-clicked', this.node.goal_id);
            },
            expand: function () {
                this.isOpen = !this.isOpen;
            },
            reemit_tac: function (node, tac) {
                this.$emit("add-tactic", node, tac);
            },
            reemit_clicked: function (node_id) {
                this.$emit("node-clicked", node_id);
            },
            goalIsSolved: function (index) {
                this.solvedChildren.add(index);
                this.n_solved = this.solvedChildren.size;
                if (this.solvedChildren.size === this.n_children) {
                    this.$emit("tactic-solves");
                }
            },
            tacticSolves: function () {
                if (!this.solved) {
                    this.solved = true;
                    this.$emit("goal-is-solved", this.index);
                }
            },
            expandCollapse: function (expand) {
                if (this.is_folder) {
                    this.isOpen = expand;
                    this.$children.forEach(function (child,) {
                        if ('expandCollapse' in child) {
                            child.expandCollapse(expand);
                        }
                    });
                }
            },
            queryModel: function () {
                let node = this.node;
                axios.post('/' + this.session_name, {
                    action: "query_model",
                    goal_id: node.goal_id,
                    settings: this.settings,
                }).then((response) => {
                    let data = response["data"];
                    console.log(data);
                    let prevSolved = this.solved;
                    // read received data
                    let log_critic = data["log_critic"];
                    let tactics = data["tactics"];
                    // update node
                    this.node.goal_data.log_critic = log_critic;
                    if ("server_error" in data) {
                      utils.handleServerError(data["server_error"]);
                    }
                    else if (tactics.length === 0) {
                        window.alert("Found no tactic!");
                    } else {
                        for (let i = 0; i < tactics.length; i++) {
                            // skip this tactic if it already exists
                            if (this.tactic_hash.has(tactics[i].hash)) {
                                console.log("tactic already exists.");
                                continue;
                            } else {
                                this.tactic_hash.add(tactics[i].hash);
                            }
                            // check whether this tactic solves the goal
                            let t_tactics = tactics[i].children;
                            if (tactics[i].is_valid && t_tactics.length === 0) {
                                this.solved = true;
                            }
                            this.$emit('add-tactic', this.node, tactics[i]);
                        }
                        // emit signal if this goal wasn't already solved
                        if (this.solved && prevSolved) {
                            console.warn("Goal " + this.node.data.statement + " was already solved");
                        } else if (this.solved && !prevSolved) {
                            this.$emit("goal-is-solved");
                        }
                    }
                    if (!this.solved) {
                        this.isOpen = true;
                    }
                });
            }
        }
    }
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
    .expand {
        cursor: pointer;
    }
    .queryButton {
        font-weight: bold;
        color: #f92672;
        cursor: pointer;
    }
    .queryButton:hover {
        color: #F9EE3E;
    }
    .hoverable:hover {
        background: #75715e;
        cursor: pointer;
    }
</style>
