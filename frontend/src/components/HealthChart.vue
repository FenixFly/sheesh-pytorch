<template>
  <canvas ref="chart" width="100%" height="250px" style="max-height: 250px;"></canvas>
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
        labels: ["12.06.2021","13.06.2021","14.06.2021","15.06.2021","16.06.2021","17.06.2021","18.06.2021"],
        datasets: [{
          borderColor: '#609DD3',
          backgroundColor: "#FF0000",
          fill: true,
          data: [0, 10, 5, 2, 20, 30, 45],
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        scales: {
          xAxes: [{
            type: 'linear',
            position: 'bottom',
            ticks: {
            }
          }]
        },
        legend: {
          display: false,
        },
        elements: {
          point:{
            radius: 0
          }
        },
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
}
</style>
