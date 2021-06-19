<template>
  <canvas ref="chart" width="100%" height="500px" style="max-height: 500px;"></canvas>
</template>

<script>
import Chart from "chart.js";

export default {
  name: 'Doughnut',
  props: ['chartData'],
  data() {
    return {
      text: 'Здесь будет кардиограмма',
      chart: null,
    }
  },
  methods:{
  },
  mounted() {
    let ctx = this.$refs.chart.getContext('2d');
    this.chart = new Chart(ctx, {
      type: 'doughnut',
      data: this.chartData,
      options: {
        responsive: true,
        maintainAspectRatio: true,
        legend: {
          position: 'top',
        },
      }
    });
  },
}
</script>
