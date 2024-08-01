<template>
    <span>
        {{goal.statement}}
    </span>
</template>

<script>


    export default {
        name: "nodemm",
        data: function () {
            return {}
        },
        props: {
            goal: Object,
            footer: Object,
        },
        computed: {},
        components: {},
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

</style>