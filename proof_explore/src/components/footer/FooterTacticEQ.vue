<template>
    <div>
        <div v-if='data.tac.rule_type === "transformation"'>
            T-rule: {{ data.tac.label }} -- Position: {{ data.tac.prefix_pos }}
            <br>
            {{ data.tac.left }} {{ direction }} {{ data.tac.right }}
        </div>
        <div v-else-if='data.tac.rule_type === "assertion"'>
            A-rule: {{ data.tac.label }}
            <br>
            Node: {{ data.tac.node }}
        </div>
        <div v-else>
            TODO: UNKNOWN RULE TYPE
        </div>
        Prior: {{ prior }} --
        Visits: {{ data.visits }} --
        Killed: {{ data.is_killed }}

        <div v-if='data.tac.hyps.length > 0'>
          Hypotheses:
          <br>
          <span v-for="hyp in data.tac.hyps" :key="hyp">
              {{ hyp }}
              <br>
          </span>
        </div>
        <div v-else>
          No hypotheses.
          <br>
        </div>

        <div v-if='Object.keys(data.tac.to_fill_infix).length > 0'>
            Substitutions:
            <br>
            <span v-for="(value, name) in data.tac.to_fill_infix" :key="name">
                <span class="substName">{{ name }}</span> = <span
                    class="substValue bottom-subst-value">{{ value }}</span>
                <br>
            </span>
        </div>
        <div v-else>
          No substitutions.
        </div>

    </div>
</template>

<script>
    export default {
        name: "footertaceq",
        props: {
            data: Object
        },
        computed: {
            direction: function () {
                return this.fwd ? '→' : '←';
            },
            hypotheses: function () {
                return this.data.tac.hyps.join(" , ");
            },
            prior: function () {
                return this.data.prior.toFixed(3);
            }
        }
    }
</script>

