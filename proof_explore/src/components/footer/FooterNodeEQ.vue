<template>
    <div>
        Goal: {{ data.statement }}
        <br>
        Critic: {{ data.log_critic }}
        <br>
        Solved: {{ data.solved }}
    </div>
</template>

<script>
    export default {
        name: "footernodeeq",
        props: {
            data: Object
        },
        computed: {}
    }
</script>

<style scoped>

</style>