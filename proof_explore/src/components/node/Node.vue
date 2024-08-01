<template>
    <span @click="$refs.nodecomp.setFooterData();setFooterData()">
        <span v-if="settings.mcts && hasBeenVisited">
          <svg class="progressBar" height="15" :width="maxBarWidth" stroke="white" style="border:1px solid white">
              <rect :height="15" :width="maxBarWidth" fill="darkslategrey"/>
              <rect :height="15" :width="getVisitsWidthAt.width" fill="blue"/>
              <text font-family="Arial" font-size="11" x="50%" y="50%" dominant-baseline="middle" text-anchor="middle">{{ visitsAt }}/{{ maxVisits }}</text>
          </svg>
          &nbsp;
          <svg class="progressBar" height="15" :width="maxNBarWidth" stroke="white" style="border:1px solid white">
              <rect :height="15" :width="maxNBarWidth" fill="darkslategrey"/>
              <rect :height="15" :width="getNormWidth(valueAt)" fill="blue"/>
              <text font-family="Arial" font-size="11" x="50%" y="50%" dominant-baseline="middle" text-anchor="middle">V={{ valueAt }}</text>
          </svg>
          <span :style="{visibility: killedAt ? 'visible' : 'hidden'}"> 💀</span>
          [MC={{modelCritic}}]
          <span v-if="goal_data.dead_node">DEAD&nbsp;</span>
          <span v-if="goal_data.is_cycle">CYCLE&nbsp;</span>
          <span class="earlyTerminal" v-if="goal_data.terminal_cause !== ''">STOP ({{ goal_data.terminal_cause }}) </span>
        </span>
        <span :class="{solvedGoal: solved}">&nbsp;</span>
        <component
            ref="nodecomp"
            :is="comp_name"
            :goal="goal_data"
            :class="{solvedGoal: solved}"
            :footer="footer">
        </component>
    </span>
</template>

<script>

    import nodemm from './NodeMM.vue'
    import nodehl from './NodeHL.vue'
    import nodeeq from './NodeEQ.vue'
    import nodelean from './NodeLean.vue'

    export default {
        name: "node",
        data: function () {
            return {}
        },
        props: {
            settings: Object,
            is_folder: Boolean,
            solved: Boolean,
            goal_data: Object,
            footer: Object,
        },
        computed: {
            comp_name: function () {
                return "node" + this.settings.language_str;
            },
            hasBeenVisited: function () {
                return this.goal_data.history !== null && this.goal_data.history !== undefined;
            },
            killedAt: function () {
                return (
                    this.goal_data.killed_time !== null &&
                    this.goal_data.killed_time <= this.settings.current_t
                );
            },
            visitsAt: function () {
                let c = 0;
                if (this.goal_data.history === null) {
                    return 0;
                }
                for (const x of this.goal_data.history) {
                    if (x[0] > this.settings.current_t)
                        break;
                    c += 1;
                }
                return c;
            },
            maxVisits: function () {
                return this.goal_data.visits;
            },
            goal_visit_ratio: function () {
                return this.goal_data.visits / Math.max(1, this.goal_data.max_visits);
            },
            maxBarWidth: function () {
                return 80.0;
            },
            maxNBarWidth: function () {
                return 60.0;  // when value is normalized between 0 and 1
            },
            getVisitsWidthAt: function () {
                // ratio between this tactic and the most visited tactic of this node
                let fracFisits = this.visitsAt / Math.max(1, this.goal_data.visits);
                return {'width': Math.ceil(this.maxBarWidth * fracFisits)};
            },
            modelCritic: function () {
                let value = this.goal_data.log_critic;
                if (value < -1e8)  // node was killed, model critic has been stored in old_log_critic
                    value = this.goal_data.old_log_critic;
                return Math.exp(value).toFixed(3);
            },
            valueAt: function () {
                if (this.goal_data.policy_data === undefined)
                  return this.modelCritic;
                let last = null;
                for (const x of this.goal_data.policy_data) {
                    if (x.timestep > this.settings.current_t)
                        break;
                    last = x;
                }
                if (last === null)
                    return this.modelCritic;
                let tid = last.best_tid;  // best ID according to the policy
                return last.Q[tid].toFixed(3);
            }
        },
        components: {
            nodemm, nodehl, nodeeq, nodelean
        },
        methods: {
            setFooterData: function () {
                this.footer.footer_type = this.comp_name;
                // use set to tell vue something was updated even though these properties didn't exist at first.
                this.$set(this.footer.footer_data, 'log_critic', this.goal_data.log_critic);
                this.$set(this.footer.footer_data, 'solved', this.solved);
                this.$emit('is-clicked');
            },
            getNormWidth: function (v) {
                return Math.ceil(this.maxNBarWidth * v);
            }
        }
    }
</script>

<style scoped>
    .solvedGoal {
        color: #A6E22E;
        background-color: #304238;
    }
    .earlyTerminal {
        color: red;
    }
</style>