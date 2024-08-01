<template>
    <span>
        <span>
            <span v-for="(token, index) in tokenizedStatement" :class="token.tokenClass" :key="index">{{token.token}}</span>
        </span>
    </span>
</template>

<script>
    import eq_utils from '../../static/eq_utils.js';

    export default {
        name: "nodeeq",
        data: function () {
            return {}
        },
        props: {
            goal: Object,
            footer: Object,
        },
        computed: {
            tokenizedStatement: function () {
                return eq_utils.splitTokens(this.goal.statement);
            },
            tokenizedHyps: function () {
                return this.goal.hyps.map((hyp, id) => [id, eq_utils.splitTokens(hyp)]);
            },
        },
        // components: {VueCodeHighlight,},
        methods:{
            setFooterData: function(){
                this.footer.footer_data = {
                    statement: this.goal.statement
                };
            }
        }
    }
</script>

<style scoped>
    @import '../../static/eq_style.css';
</style>

<style>
    pre {
        display:inline;
    }
</style>