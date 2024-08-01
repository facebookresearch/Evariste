<template>
    <div>
        Tactic [{{data.prior}}]
        <span class="tacticLabel">{{ data.label }}</span>
        <br>
        <span v-for="(value, name) in data.subs" :key="name">
            <span class="substName">{{ name }}</span> = <span class="substValue">{{ value }}&nbsp;</span>
        </span>
        <br><br>
        <div v-if="data.valid">
            TODO
        </div>
        <!-- invalid tactic -->
        <div v-else>
            Invalid tactic: {{ data.error_msg }}
        </div>
        <br>

        <!-- label info -->
        <div v-if="data.label_type !== undefined">
            {{ data.label_type }} label:
            <a v-bind:href="'http://us.metamath.org/mpeuni/' + data.label + '.html'" target="_blank">
                {{data.label }}
            </a>
            -- statement: <span id="bottom-rule-statement">{{ data.label_statement }}</span>
            <br><br>
            <span id="bottom-tactic-ehyps">Rule $e hypotheses:</span>
            <span v-if="data.label_e_hyps.length === 0">None</span>
            <ul>
                <li v-for="value in data.label_e_hyps" class="bottom-rule-ehyp" :key="value">{{ value }}</li>
            </ul>
        </div>

        <!-- substitutions -->
        <span id="bottom-tactic-subs">Substitutions:</span>
        <ul>
            <li v-for="(value, name) in data.subs" :key="name">
                <span class="substName">{{ name }}</span> = <span
                    class="substValue bottom-subst-value">{{ value }}</span>
            </li>
        </ul>

    </div>
</template>

<script>
    export default {
        name: "footertacmm",
        props: {
            data: Object
        }
    }
</script>

<style scoped>
    .tacticLabel {
        font-weight: bold;
        color: #66d9ef;
    }

    .substName {
        font-weight: bold;
        color: #ae81ff;
    }

    .substValue {
        color: #f8f8f2;
    }

    .solvedAllChildren {
        color: #a6e22e;
    }

    .missingChildren {
        color: #f92672;
    }

    #bottom-tactic-ehyps {
        font-weight: bold;
    }

    #bottom-tactic-subs {
        font-weight: bold;
    }

    #bottom-rule-statement {
        font-family: Consolas, monospace;
    }

    .bottom-rule-ehyp {
        font-family: Consolas, monospace;
    }

    .bottom-subst-value {
        font-family: Consolas, monospace;
    }

    a {
        color: #66d9ef;
        font-weight: bold;
    }
</style>