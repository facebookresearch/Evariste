<template>
    <div>
       <span v-for="(token, id) in tokenizedStr" :key="id" :class="token.tokenClass">
           {{token.token }}&nbsp;
       </span>
    </div>
</template>

<script>
    import hl_utils from '../../static/hl_utils.js';

    export default {
        name: "footertachl",
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