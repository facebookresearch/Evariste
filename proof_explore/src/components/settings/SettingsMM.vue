<template>
    <div>
        <input id="checkbox_show_predictable" type="checkbox"
               v-model="settings.show_predictable"
               checked
        >
        <label for="checkbox_show_predictable">Show predictable substitutions</label>
        <br>
        <br>
    </div>
</template>

<script>
    export default {
        name: "settingsmm",
        props: {
            settings: Object
        }
    }
</script>

<style scoped>

</style>