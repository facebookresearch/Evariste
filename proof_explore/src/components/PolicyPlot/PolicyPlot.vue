<template>
  <div class="small">
    <bar-chart :chart-data="datacollection" :options="options"></bar-chart>
  </div>
</template>

<script>
  import BarChart from './PolicyPlot.js'

  export default {
    name: "policy_plot",
    components: {
      BarChart
    },
    data () {
      return {
        datacollection: {
          labels: [],
          datasets: []
        },
        options:{
            scales: { yAxes: [{ ticks: { beginAtZero: true,}}]}
        },
      }
    },
    methods: {
      plot(policy, counts, virtual_counts, Q, priors){
        this.datacollection = {
            labels: [...Array(policy.length).keys()].map(String),
            datasets: [
                {
                    label: 'policy',
                    data: policy,
                    backgroundColor: '#E69F66',
                    borderColor: '#E69F66',
                    borderWidth: 1
                },
                {
                    label: 'Q',
                    data: Q,
                    backgroundColor: '#ae81ff',
                    borderColor: '#ae81ff',
                    borderWidth: 1
                },
                {
                    label: 'priors',
                    data: priors,
                    backgroundColor: '#66d9ef',
                    borderColor: '#66d9ef',
                    borderWidth: 1
                },
                {
                    label: 'counts',
                    data: counts,
                    backgroundColor: '#E6DB74',
                    borderColor: '#E6DB74',
                    borderWidth: 1
                },
                {
                    label: 'virtual_counts',
                    data: virtual_counts,
                    backgroundColor: '#F8F8F2',
                    borderColor: '#F8F8F2',
                    borderWidth: 1
                }
            ]
        }
      },
    }
  }
</script>

<style>
  .small {
    max-width: 600px;
    /*margin:  150px auto;*/
  }
</style>