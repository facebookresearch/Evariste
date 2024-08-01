<template>
    <span v-if="settings.show_tactics" @click="$refs.taccomp.setFooterData();setFooterData()">
        <span v-if="isValid" :class="{solvingTactic: solving}">
          <span v-if="settings.mcts && hasMCTSData">
            <!--   ON THE FLY COMPUTATION   -->
<!--            <svg class="progressBar" height="15" :width="maxBarWidth" stroke="white" style="border:1px solid white">-->
<!--                <rect :height="15" :width="maxTacBarWidth" fill="#480607"/>-->
<!--                <rect :height="15" :width="getVisitsWidthAt.width" fill="red"/>-->
<!--                <text font-family="Arial" font-size="11" x="50%" y="50%" dominant-baseline="middle" text-anchor="middle">{{ computeValues.visits }}/{{ maxVisits }}</text>-->
<!--            </svg>-->
<!--            &nbsp;-->
<!--            <svg class="progressBar" height="15" :width="maxNBarWidth" stroke="white" style="border:1px solid white">-->
<!--                <rect :height="15" :width="maxNBarWidth" fill="#480607"/>-->
<!--                <rect :height="15" :width="computeValues.QWidth" fill="red"/>-->
<!--                <text font-family="Arial" font-size="11" x="50%" y="50%" dominant-baseline="middle" text-anchor="middle">Q={{ computeValues.Q }}</text>-->
<!--            </svg>-->
            <!--   PRECOMPUTED   -->
            <svg class="progressBar" height="15" :width="maxBarWidth" stroke="white" style="border:1px solid white">
                <rect :height="15" :width="maxTacBarWidth" fill="#480607"/>
                <rect :height="15" :width="getVisitsWidthAt.width" fill="red"/>
                <text font-family="Arial" font-size="11" x="50%" y="50%" dominant-baseline="middle" text-anchor="middle">{{ computeValuesV2.visits }}/{{ maxVisits }}</text>
            </svg>
            &nbsp;
            <svg class="progressBar" height="15" :width="maxNBarWidth" stroke="white" style="border:1px solid white">
                <rect :height="15" :width="maxNBarWidth" fill="#480607"/>
                <rect :height="15" :width="getNormWidth(computeValuesV2.Q)" fill="red"/>
                <text font-family="Arial" font-size="11" x="50%" y="50%" dominant-baseline="middle" text-anchor="middle">Q={{ computeValuesV2.Q }}</text>
            </svg>
            &nbsp;
            <svg class="progressBar" height="15" :width="maxNBarWidth" stroke="white" style="border:1px solid white">
                <rect :height="15" :width="maxNBarWidth" fill="#480607"/>
                <rect :height="15" :width="getNormWidth(computeValuesV2.policy)" fill="red"/>
                <text font-family="Arial" font-size="11" x="50%" y="50%" dominant-baseline="middle" text-anchor="middle">p={{ computeValuesV2.policy }}</text>
            </svg>
            <span :style="{visibility: killedAt ? 'visible' : 'hidden'}"> 💀</span>
          </span>
          <span>[MP={{modelPrior}}]</span>
          <!--[{{n_solved}}/{{n_children}}]-->
          <component
              ref="taccomp"
              :is="comp_name"
              :tactic_data="tactic_data"
              :settings="settings"
              :footer="footer">
          </component>
        </span>
        <!-- invalid tactic -->
        <span v-else>
          <span class="invalidTactic">[{{modelPrior}}] {{ tactic_data.tac.error_msg }}</span>
        </span>
    </span>
</template>

<script>

    import tacmm from './TacMM.vue'
    import tachl from './TacHL.vue'
    import taceq from './TacEQ.vue'
    import taclean from './TacLean.vue'

    export default {
        name: "tactic",
        // data: function () {
        //     return {}
        // },
        props: {
            settings: Object,
            tactic_data: Object,
            n_children: Number,
            n_solved: Number,
            show_invalid: Boolean,
            solving: Boolean,
            footer: Object,
        },
        computed: {
            comp_name: function () {
                return "tac" + this.settings.language_str;
            },
            isValid: function () {
                return !("error_msg" in this.tactic_data.tac);
            },
            // killedTime: function () {
            //     return this.tactic_data.killed_time;
            // },
            modelPrior: function () {
                return this.tactic_data.prior.toFixed(3);
            },
            Q: function () {
                if (this.tactic_data.log_Q !== undefined) {
                  return Math.exp(this.tactic_data.log_Q).toFixed(3);
                } else {
                  return "~";
                }
            },
            hasMCTSData: function () {
                return this.tactic_data.tac_history !== undefined;
            },
            killedAt: function () {
                return (
                    this.tactic_data.killed_time !== null &&
                    this.tactic_data.killed_time <= this.settings.current_t
                );
            },
            maxVisits: function () {
                return this.tactic_data.visits;
            },
            tacVisitRatio: function () {
                return this.tactic_data.visits / Math.max(1, this.tactic_data.max_visits);
            },
            maxBarWidth: function () {
                return 80.0;
            },
            maxNBarWidth: function () {
                return 60.0;  // when value is normalized between 0 and 1
            },
            maxTacBarWidth: function () {
                return Math.ceil(this.tacVisitRatio * this.maxBarWidth);
            },
            getVisitsWidthAt: function () {
                // ratio between this tactic and the most visited tactic of this node
                let fracVisits = this.computeValues.visits / Math.max(1, this.tactic_data.visits);
                return {'width': Math.ceil(this.maxTacBarWidth * fracVisits)};
            },
            computeValues: function () {
                if (this.tactic_data.tac_history.length === 0)
                    return {'lastUpdate': -1, 'visits': 0, 'W': -1, 'Q': -1, 'lastUpdateWidth': 0, 'QWidth': 0};
                let update = -1e9;
                let W = 0;
                let visits = 0;
                for (const x of this.tactic_data.tac_history) {
                    if (x[0] > this.settings.current_t)
                        break;
                    visits += 1;
                    update = Math.exp(x[1]); // (timestep, log_value)
                    W += update;
                }
                if (visits === 0)
                    return {'lastUpdate': -1, 'visits': 0, 'W': -1, 'Q': -1, 'lastUpdateWidth': 0, 'QWidth': 0};
                let Q = W / Math.max(visits, 1);
                return {
                    // 'lastUpdate': update.toFixed(3),
                    // "W": W.toFixed(3),
                    "visits": visits,
                    "Q": Q.toFixed(3),
                    "QWidth": this.getNormWidth(Q),
                    // "lastUpdateWidth": this.getNormWidth(update),
                }
            },
            computeValuesV2: function () {
                let last = {"counts": 0, "Q": 0, policy: 0};
                for (const x of this.tactic_data.tac_policy_data) {
                    if (x.timestep > this.settings.current_t)
                        break;
                    last = x;
                }
                return {
                    // "W": Math.exp(last.logW).toFixed(3),
                    "visits": last.counts,
                    "Q": last.Q.toFixed(3),
                    "policy": last.policy.toFixed(3),
                };
            }
        },
        components: {tacmm, tachl, taceq, taclean},
        methods: {
            setFooterData: function () {
                if (this.isValid) {
                    this.footer.footer_type = this.comp_name;
                } else {
                    this.footer.footer_type = 'invalidtac';
                }
                this.footer.footer_data = this.tactic_data;
                // // use set to tell vue something was updated even though these properties didn't exist at first.
                // this.$set(this.footer.footer_data, 'prior', this.prior);
                // this.$set(this.footer.footer_data, 'n_solved', this.n_solved);
                // this.$set(this.footer.footer_data, 'n_children', this.n_children);
            },
            getNormWidth: function (v) {
                return Math.ceil(this.maxNBarWidth * v);
            },
        }
    }
</script>

<style scoped>
    .invalidTactic {
        color: #f92672;
        background-color: #790022;
    }
    .solvingTactic {
        color: #A6E22E;
        background-color: #304238;
    }
</style>