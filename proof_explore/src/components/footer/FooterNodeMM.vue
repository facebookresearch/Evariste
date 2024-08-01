<template>
    <div>
        Goal: {{ data.statement }}
        <br><br>
        Critic: {{ data.log_critic }}
        <br><br>
        Solved: {{ data.solved }}
    </div>
</template>

<script>
    export default {
        name: "footernodemm",
        props: {
            data: Object
        }
    }
</script>

<style scoped>

</style>