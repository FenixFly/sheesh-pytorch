<template>
  <canvas ref="chart" width="100%" height="500px" style="max-height: 500px;"></canvas>
</template>

<script>
import Chart from "chart.js";

export default {
  name: 'Bar',
  props: ['barData'],
  data() {
    return {
      chart: null,
    }
  },
  mounted() {
    let ctx = this.$refs.chart.getContext('2d');
    this.chart = new Chart(ctx, {
      type: 'bar',
      data: this.barData,
      options: {
        responsive: true,
        maintainAspectRatio: true,
        legend: {
          display: false,
        },
        scales: {
          xAxes: [{
            gridLines: {
              display:false
            }
          }],
          yAxes: [{
            type: 'linear',
            position: 'bottom',
            gridLines: {
              display:false
            },
            ticks: {
              beginAtZero: true,
              min: 0
            }
          }]
        },
      }
    });
  },
}
</script>
