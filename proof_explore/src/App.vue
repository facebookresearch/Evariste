<template>
    <div id="demo">
        <div id="header_bar">
            <div v-if="tree === null">
                <p>
                  <label for="theorem_statement">Theorem statement: </label>
                  <input id="theorem_statement" v-model="statement_str" placeholder="|- ( 2 + 2 ) = 4">
                </p>
                <p v-if='settings.language_str === "mm" || settings.language_str === "eq"'>
                    <label for="theorem_hyps">Theorem hypotheses:</label><br><br>
                    <textarea id="theorem_hyps" v-model="hyps_str" placeholder="Theorem hypotheses"></textarea>
                </p>
                Or<br>
                <p>
                  <label for="theorem_label">Theorem label: </label>
                  <input id="theorem_label" v-model="label_str" placeholder="2p2e4">
                </p>
                Or<br>
                <p>
                    <label for="proof_path">Proof dump file: </label>
                    <input id="proof_path" v-model="proof_path" placeholder="/path/to/proof_dump.pkl">
                </p>
                Or<br>
                <p>
                    <label for="mcts_path">MCTS dump file: </label>
                    <input id="mcts_path" v-model="mcts_path" placeholder="/path/to/mcts_dump.pkl">
                </p>
                <button v-on:click="saveGoal">Save</button>
            </div>
            <div v-else>
                <span>Theorem statement: {{goal.statement}}</span>
                <div v-if="goal.hyps.length > 0">
                    Theorem hypotheses:
                    <br>
                    <span v-for="(hyp, index) in goal.hyps" :key="index">{{ hyp }}<br></span>
                </div>
                <settings
                    :settings="settings"
                    @expand-collapse="expandCollapse"
                    @update-policy="updatePolicy">
                </settings>
                <!--@changed-t="replot"-->
            </div>
        </div>
        <ul id="main_frame" class="item-subgoal">
            <tree v-if="tree !== null" ref="tree"
                :node="tree"
                :settings="settings"
                node_type="node"
                :footer="footer"
                @add-tactic="addTactic" :session_name="session_name">
            </tree>
                <!--@node-clicked="nodeClicked"-->
        </ul>
        <div v-if="footer.footer_type !== undefined" id="footer">
            <footer_holder :data="footer"/>
        </div>
        <!--<policy_plot ref="policy_plot" v-if="show_policy"></policy_plot>-->
    </div>
</template>

<script>
    import tree from './components/Tree.vue';
    import footer_holder from './components/footer/Footer.vue';
    import settings from './components/settings/Settings.vue';
    // import policy_plot from './components/PolicyPlot/PolicyPlot.vue';
    import utils from './static/utils';
    import axios from "axios";

    let settings_data = {
        language_str: undefined,
        show_predictable: false,
        show_solved: true,
        show_tactics: true,
        show_invalid: false,
        use_beam: true,
        use_sampling: true,
        n_samples: 8,
        sample_temperature: 1,
        sample_topk: 50,
        prefix: "",
        // for mcts
        mcts: false,
        max_t: undefined,
        current_t: 0,
        max_display_depth: 4,
        max_reload_depth: 6,
        show_zero_visits: true,
        selected_node_id: undefined,
        policy_type: "other",
        exploration: 5.0,
    };
    let footer_data = {
        footer_type: undefined,
        footer_data: {},
    };
    export default {
        name: 'App',
        mounted() {
            axios.post('/' + this.session_name, {
                action: "grab_state",
            }).then((response) => {
                this.receiveTree(response["data"]);
            });
        },
        components: {
            tree, footer_holder, settings  //, policy_plot
        },
        props: {
            session_name: String,
        },
        data: function () {
            return {
                /* these will be maybe set by the mounted ajax call */
                language_str: undefined,
                tree: null,
                goal: undefined,
                statement_str: "",
                hyps_str: "",
                label_str: "",
                proof_path: "",
                mcts_path: "",
                settings: settings_data,
                footer: footer_data  // contains settings
            }
        },
        methods: {
            saveGoal: function () {
                axios.post('/' + this.session_name, {
                    action: "initialize",
                    conclusion: this.statement_str,
                    hyps: this.hyps_str,
                    label: this.label_str,
                    proof_dump: this.proof_path,
                    mcts_dump: this.mcts_path,
                    language: this.language_str,
                }).then((response) => {
                    let data = response["data"];
                    // console.log(data);
                    this.receiveTree(data);
                });
            },
            receiveTree( data ) {
                if ("server_error" in data) {
                    utils.handleServerError(data["server_error"]);
                } else if ("error" in data) {
                    window.alert(data["error"]);
                } else {
                    this.goal = data["goal"];
                    this.tree = data["tree"];
                    this.settings.language_str = data["session_type"];
                    if (data["mcts"] !== undefined) {
                        let max_t = data["mcts"]["max_t"]
                        // default visualization to last time if first loading
                        if (this.settings.max_t === undefined) {
                            this.settings.current_t = max_t;
                        }
                        this.settings.mcts = true;
                        this.settings.max_t = max_t;
                    }
                    console.log("MCTS: " + JSON.stringify(data["mcts"]));
                }
            },
            addTactic: function (item, tactic) {
                item.children.push(tactic);
            },
            expandCollapse: function (expand) {
                this.$refs.tree.expandCollapse(expand);
            },
            updatePolicy: function ( ) {
                console.log("Updating policy...", this.settings.policy_type, this.settings.exploration, this.settings.max_reload_depth);
                axios.post('/' + this.session_name, {
                    action: "update_policy",
                    policy_type: this.settings.policy_type,
                    exploration: parseFloat(this.settings.exploration),
                    max_reload_depth: parseInt(this.settings.max_reload_depth),
                }).then((response) => {
                    this.receiveTree(response["data"]);
                });
            }
            // nodeClicked: function (node_id) {
            //     if (this.settings.max_t !== undefined) {
            //         this.selected_node_id = node_id;
            //         this.replot();
            //     }
            // },
            // replot: function() {
            //     if (this.selected_node_id !== undefined) {
            //         axios.post('/' + this.session_name, {
            //             action: "get_policy",
            //             node_id: this.selected_node_id,
            //             timestamp: this.settings.current_t,
            //         }).then((response) => {
            //             let data = response["data"];
            //             if ("server_error" in data) {
            //                 console.log(data["server_error"]);
            //                 window.alert("server error");
            //             } else if ("error" in data) {
            //                 console.log(data["error"]);
            //                 window.alert("error");
            //             } else {
            //                 this.$nextTick(function () {
            //                     this.$refs.policy_plot.plot(
            //                         data["policy"],
            //                         data["counts"],
            //                         data["virtual_counts"],
            //                         data["Q"],
            //                         data["priors"]
            //                     );
            //                 });
            //             }
            //         });
            //     }
            // }
        }
        // computed: {
        //     show_policy () {
        //         return this.settings.max_t !== undefined &&  // MCTS mode
        //                this.footer.footer_type !== undefined &&  // node selected
        //                this.footer.footer_type.startsWith('node');
        //     }
        // }
    }
</script>

<style>
    #demo {
      height: 100%;
      width: max-content;
      min-height: 100%;
      min-width: 100%;
      display: flex;
      flex-direction: column;
      flex-wrap: nowrap;
    }
    #header_bar {
      flex-shrink: 0;
      width: 100%;
    }
    #main_frame {
      flex-grow: 1;
      overflow: auto;
      min-height: 2em;
      border-top: 2px solid white;
      border-bottom: 2px solid white;
    }
    #footer {
      padding-bottom: 12px;
      flex-shrink: 0;
      width: max-content;
    }
    li {
      margin-left: -20px;
    }
    .item-tactic {
        list-style-type: disc;
    }
    .item-subgoal {
        list-style-type: square;
    }
    .progressBar {
        vertical-align: middle;
    }
</style>
