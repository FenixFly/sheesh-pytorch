<template>
  <canvas ref="chart" class="chart"></canvas>
</template>

<script>
import Chart from "chart.js";

export default {
  name: "MiniHealthChart",
  data: () => ({
    chartData: null,
  }),
  mounted() {
    let ctx = this.$refs.chart.getContext('2d');
    this.chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: ["","","","","","","",],
        datasets: [{
          borderColor: '#609DD3',
          fill: false,
          borderWidth: 5,
          data: [0, 10, 5, 2, 20, 30, 45],
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        legend:{
          display: false,
        },
        elements: {
          point:{
            radius: 0
          }
        },
        scales: {
          xAxes: [{
            gridLines: {
              display:false
            },
            ticks: {
              display: false
            }
          }],
          yAxes: [{
            gridLines: {
              display:false
            },
            ticks: {
              display: false
            }
          }]
        }

      }
    });
  },
}
</script>

<style scoped>
.chart{
  width: 100%;
  height: 200px;
  max-height: 150px;
  margin-left: -10px;
  margin-right: -50px;
  transform: scaleX(1.1);
}
</style>
