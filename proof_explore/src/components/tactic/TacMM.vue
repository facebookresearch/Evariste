<template>
    <span>
          <span class="tacticLabel">{{tactic_data.tac.label }}</span>&nbsp;
          <span v-for="(value, name) in visible_subs" :key="name">
            <span class="substName">{{name }}</span> =
            <span class="substValue">{{value}}&nbsp;</span>
          </span>
    </span>
</template>

<script>
    export default {
        name: "tacmm",
        props: {
            tactic_data: Object,
            settings: Object,
            footer: Object,
        },
        computed: {
            visible_subs: function () {
                if (this.settings.show_predictable) {
                    return this.tactic_data.tac.subs;
                } else {
                    return Object.fromEntries(Object.entries(this.tactic_data.tac.subs).filter(
                        ([name,]) => this.tactic_data.tac.pred_substs.indexOf(name) === -1)
                    );
                }
            },
        },
        components: {},
        methods: {
            setFooterData: function () {}
        }
    }
</script>

<style scoped>
    .substName {
        font-weight: bold;
        color: #ae81ff;
    }

    .substValue {
        color: #f8f8f2;
    }

    .tacticLabel {
        font-weight: bold;
        color: #66d9ef;
    }
</style>