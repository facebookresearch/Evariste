<template>
    <div>
        Goal: {{ data.statement }}
        <br><br>
        log-critic: {{ data.log_critic }}
        <br><br>
        Solved: {{ data.solved }}
    </div>
</template>

<script>
    import hl_utils from '../../static/hl_utils.js';

    export default {
        name: "footernodehl",
        props: {
            data: Object
        },
        computed: {
            tokenizedStr: function () {
                return hl_utils.splitTokens(this.data.str);
            },
        }
    }
</script>

<style scoped>

</style>