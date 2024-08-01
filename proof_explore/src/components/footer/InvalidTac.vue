<template>
    <div>
        Tactic [{{data.prior}}]
        <br>
        <span class="invalidTactic">Invalid tactic: </span><br>
        {{ data.error_msg }}

    </div>
</template>

<script>
    export default {
        name: "footerinvalidtac",
        props: {
            data: Object
        }
    }
</script>

<style scoped>
    .invalidTactic {
        color: #f92672;
    }
</style>