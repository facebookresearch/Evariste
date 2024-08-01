<template>
    <span>
        <!-- <vue-code-highlight language="javascript" style="display:inline"> -->
        <span v-if="tactic_data.tac.rule_type==='transformation'">
          {{tactic_data.tac.left}}
          {{direction}}
          {{tactic_data.tac.right}}
        </span>
        <span v-else-if="tactic_data.tac.rule_type==='assertion'">
          {{tactic_data.tac.node}} -- {{ hypotheses }}
        </span>
        <span v-else>
          Unkown tactic type: {{ tactic_data.tac.rule_type }}
        </span>
        <!-- </vue-code-highlight> -->
        <!-- <vue-code-highlight language="javascript" style="display:inline"> -->
        <!-- </vue-code-highlight> -->
    </span>
</template>

<script>

    export default {
        name: "taceq",
        data: function () {
            return {}
        },
        props: {
            tactic_data: Object,
            settings: Object,
            footer: Object,
        },
        computed: {
            direction: function () {
                return this.tactic_data.tac.fwd ? '→' : '←';
            },
            hypotheses: function () {
                return this.tactic_data.tac.hyps.join(" , ");
            }
        },
        methods: {
            setFooterData: function () {}
        }
    }
</script>