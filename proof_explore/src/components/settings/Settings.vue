<template>
    <div>
        <component :is="settings_type" :settings="settings"></component>
        <!--show solved-->
        <input
            id="checkbox_show_solved"
            type="checkbox"
            v-model="settings.show_solved"
            checked
        >
        <label for="checkbox_show_solved">Show solved subgoals </label>
        <!--show tactics-->
        <input
            id="checkbox_show_tactics"
            type="checkbox"
            v-model="settings.show_tactics"
            checked
        >
        <label for="checkbox_show_tactics">Show tactics </label>
        <!--show invalid tactics-->
        <input
            id="checkbox_show_invalid"
            type="checkbox"
            v-model="settings.show_invalid"
            v-if="settings.show_tactics"
            checked
        >
        <label v-if="settings.show_tactics" for="checkbox_show_invalid">Show invalid tactics</label>
        <br>
        <!--use beam-->
        <input
            id="checkbox_use_beam"
            type="checkbox"
            v-model="settings.use_beam"
            checked
        >
        <label for="checkbox_use_beam">Use beam </label>
        <!--use sampling-->
        <input
            id="checkbox_sampling"
            type="checkbox"
            v-model="settings.use_sampling"
            checked
        >
        <label for="checkbox_sampling">Use sampling </label>
        <br>
        <span>
          <label for="input_n_samples">Generation samples: </label>
          <input id="input_n_samples" v-model.number="settings.n_samples" placeholder="8" size="4" :disabled="!settings.use_beam && !settings.use_sampling">
          &nbsp;
          <label for="input_sample_temperature">Temperature: </label>
          <input id="input_sample_temperature" v-model.number="settings.sample_temperature" placeholder="1.0" size="4" :disabled="!settings.use_beam && !settings.use_sampling">
          &nbsp;
          <label for="input_topk">Top-k: </label>
          <input id="input_topk" v-model.number="settings.sample_topk" placeholder="50" size="4" :disabled="!settings.use_beam && !settings.use_sampling">
        </span>
        <span v-if="!settings.use_beam && !settings.use_sampling" class="textWarning">
          This configuration can only generate one sample!
        </span>
        <label for="input_prefix"> Prefix: </label>
        <input id="input_prefix" v-model="settings.prefix" placeholder="" size="60">
        <br>
        <button @click="$emit('expand-collapse', true)">Expand all</button>
        &nbsp;
        <button @click="$emit('expand-collapse', false)">Collapse all</button>

        <div v-if="settings.max_t !== undefined">
            <hr>
            <label for="input_current_t">MCTS Timestep </label>
            <input id="input_current_t" v-model.number="settings.current_t" @change="changed_t" min="0" :max="settings.max_t" type="range"> {{ settings.current_t }} / {{ settings.max_t }}
            <span v-if="playing == null" class="playstop" @click="play()">▶</span>
            <span v-else class="playstop" @click="stop()">■</span>
            <br>
            <label for="input_max_display_depth">Max display depth </label>
            <input id="input_max_display_depth" v-model.number="settings.max_display_depth" size="4">
            &nbsp;
            <label for="input_max_reload_depth">Max reload depth </label>
            <input id="input_max_reload_depth" v-model.number="settings.max_reload_depth" size="4">
            <br>
            <input id="checkbox_show_zero_visits" type="checkbox" v-model="settings.show_zero_visits" checked>
            <label for="checkbox_show_zero_visits">Show tactics without visits</label>&nbsp;
            <br>

            <label for="policy_type_a0">
                <input type="radio" v-model="settings.policy_type" value="alpha_zero" id="policy_type_a0">
                Alpha-Zero
            </label>
            <label for="policy_type_other">
                <input type="radio" v-model="settings.policy_type" value="other" id="policy_type_other">
                Other
            </label>
            &nbsp; -- &nbsp;
            <label for="input_exploration">Exploration: </label>
            <input id="input_exploration" v-model.number="settings.exploration" placeholder="1.0" size="7">
            &nbsp;
            <button @click="$emit('update-policy')">Update policy</button>
        </div>
    </div>

</template>

<script>
    import settingshl from './SettingsHL.vue';
    import settingsmm from './SettingsMM.vue';
    import settingseq from './SettingsEQ.vue';

    export default {
        name: "Settings.vue",
        props: {
            settings: Object
        },
        data: function () {
            return {
                // for mcts only
                playing: null,
                // new_max_depth: 5,
            }
        },
        components: {
            settingshl, settingsmm, settingseq
        },
        methods: {
            changed_t: function() {
                this.$emit('changed-t');
                if (this.settings.current_t >= this.settings.max_t) {
                    this.stop();
                }
            },
            // for mcts
            play: function(){
                this.playing = setInterval(
                    (function(self) {
                        return function() {
                            self.settings.current_t += 1;
                            self.changed_t();
                        };
                    })(this), 500
                );
            },
            stop: function() {
                if (this.playing != null) {
                    clearInterval(this.playing);
                    this.playing = null;
                }
            },
        },
        computed: {
            settings_type: function () {
                return 'settings' + this.settings.language_str;
            }
        }
    }
</script>

<style scoped>
.playstop:hover {
    color: #F92672;
    cursor: pointer;
}
.textWarning {
  color: #ff0066;
}
</style>