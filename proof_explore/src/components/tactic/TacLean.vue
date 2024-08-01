<template>
    <span>
        <vue-code-highlight language="javascript" style="display:inline">
             {{tactic_data.tac.str}}
        </vue-code-highlight>
    </span>
</template>

<script>
    import { component as VueCodeHighlight } from 'vue-code-highlight';

    export default {
        name: "taclean",
        props: {
            tactic_data: Object,
            settings: Object,
            footer: Object,
        },
        computed: {},
        components: {VueCodeHighlight,},
        methods: {
            setFooterData: function () {}
        }
    }
</script>

<style scoped>
    @import '../../static/syntax.css';
</style>

<style>
    pre {
        display:inline;
    }
</style>