<template>
    <span>
        <span v-if="goal.hyps.length > 0">
          Hypotheses:
          <span v-for="(hyp, hypid) in tokenizedHyps" :key="hypid">
            <span v-if="hyp[0]">{{ hyp[0] }}: </span>
            <span v-for="(token, index) in hyp[1]" :key="index" :class="token.tokenClass">
                {{ token.token }}&nbsp;
            </span>
            <br>
          </span>
        </span>
        <!-- Goal: {{ item.statement }} -->
        Goal:
        <span v-for="(token, index) in tokenizedStatement" :class="token.tokenClass" :key="index">
            {{ token.token }}
        </span>
    </span>
</template>

<script>
    import hl_utils from '../../static/hl_utils.js';

    export default {
        name: "nodehl",
        data: function () {
            return {}
        },
        props: {
            goal: Object,
            footer: Object,
        },
        computed: {
            tokenizedStatement: function () {
                return hl_utils.splitTokens(this.goal.statement);
            },
            tokenizedHyps: function () {
                return this.goal.hyps.map((hyp, id) => [id, hl_utils.splitTokens(hyp)]);
            },
        },
        components: {},
        methods: {
            setFooterData: function () {
                this.footer.footer_data = {
                    statement: this.goal.statement
                };
            }
        }
    }
</script>

<style scoped>
    @import '../../static/hl_style.css';
</style>