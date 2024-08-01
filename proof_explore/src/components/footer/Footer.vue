<template>
    <div>
        <component :is="footer_type" :data="data.footer_data">
        </component>
    </div>
</template>

<script>
    import footertacmm from './FooterTacticMM.vue';
    import footernodemm from './FooterNodeMM.vue';
    import footertachl from './FooterTacticHL.vue';
    import footernodehl from './FooterNodeHL.vue';
    import footertaceq from './FooterTacticEQ.vue';
    import footernodeeq from './FooterNodeEQ.vue';
    import footerinvalidtac from './InvalidTac.vue';

    export default {
        name: "footer_holder",
        props: {
            data: Object
        },
        components: {
            footertacmm, footernodemm, footertachl, footernodehl,  footertaceq, footernodeeq, footerinvalidtac
        },
        computed: {
            footer_type: function() {
                return 'footer' + this.data.footer_type;
            }
        },
    }
</script>

<style scoped>

</style>