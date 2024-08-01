<template>
    <span>
       <span v-for="(token, id) in tokenizedStr" :key="id" :class="token.tokenClass">
           {{token.token }}&nbsp;
       </span>
    </span>
</template>

<script>
    import hl_utils from '../../static/hl_utils.js';

    export default {
        name: "tachl",
        props: {
            tactic_data: Object,
            settings: Object,
            footer: Object,
        },
        computed: {
            tokenizedStr: function () {
                return hl_utils.splitTokens(this.tactic_data.tac.str);
            },
        },
        components: {},
        methods: {
            setFooterData: function () {}
        }
    }
</script>

<style scoped>
    @import '../../static/hl_style.css';
</style>