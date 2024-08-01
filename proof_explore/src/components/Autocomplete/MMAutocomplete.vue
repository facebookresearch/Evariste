<template>
    <span>
        Label:&nbsp;
        <vue-simple-suggest
                v-model="chosen"
                :destyled="false"
                :list="suggestionList"
                :filter-by-query="true" style="display: inline">
    <!-- Filter by input text to only show the matching results -->
      </vue-simple-suggest>
    </span>
</template>
<script>
    // import Axios from 'axios';
    import VueSimpleSuggest from 'vue-simple-suggest'
    import 'vue-simple-suggest/dist/styles.css' // Optional CSS

    export default {
        name: "mmautocomplete",
        props: {},
        components: {VueSimpleSuggest},
        data() {
            return {
                chosen: ''
            };
        },
        methods: {
            suggestionList() {
                // return fetch('https://swapi.co/api/people', { method: 'GET' })
                //   .then(response => response.json())
                //   .then(json => json.results);
                return ['ax-mp', 'ax-mp1', 'ax-mp2'];
            }
        }
    };
</script>

<style>
/*  vue-simple-suggest.designed.focus // .focus is applied whenever the component is focused. */
/*.input-wrapper*/
/*  .default-input // Replaced with your custom input if default slot is provided. */
/*.suggestions // Also has transition classes, see below. */
/*  .suggest-item*/

.vue-simple-suggest.designed .suggestions {

    max-width: 100px;
    padding-left: 5px;
    margin-left: 20px;
    margin-top: 10px;
}
.vue-simple-suggest.designed ul {
    list-style-type: None;
}
.vue-simple-suggest.designed .suggest-item:hover {
    background: #75715e;
    cursor: pointer;
}
</style>