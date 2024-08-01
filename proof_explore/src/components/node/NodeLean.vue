<template>
    <span>
        <vue-code-highlight language="javascript" style="display:inline">
           {{goal.statement}}
        </vue-code-highlight>
    </span>
</template>

<script>
    import { component as VueCodeHighlight } from 'vue-code-highlight';
    export default {
        name: "nodelean",
        data: function () {
            return {}
        },
        props: {
            goal: Object,
            footer: Object,
        },
        computed: {},
        components: {VueCodeHighlight,},
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
    @import '../../static/syntax.css';
</style>

<style>
    pre {
        display:inline;
    }
</style>