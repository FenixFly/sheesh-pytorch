<template>
  <canvas ref="chart" width="100%" height="500px" style="max-height: 500px;"></canvas>
</template>

<script>
import Chart from "chart.js";

export default {
  name: 'Rhythmogram',
  props: ['rgData', 'static'],
  data() {
    return {
      text: 'Здесь будет ритмограмма',
      chart: null,
      animationDuration: 1000,
    }
  },
  methods:{
    updateChart(){
      this.chart.config.options.animation.duration = 0;
      this.chart.update();
    },
  },
  mounted() {
    if(this.static) this.animationDuration = 0;
    let ctx = this.$refs.chart.getContext('2d');
    this.chart = new Chart(ctx, {
      type: 'line',
      data: this.rgData,
      options: {
        responsive: true,
        maintainAspectRatio: true,
        scales: {
          xAxes: [{
            type: 'linear',
            position: 'bottom',
            ticks: {
              suggestedMin: 0,    // minimum will be 0, unless there is a lower value.
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
        animation: {
          duration: this.animationDuration,
        },
      }
    });

  },
}
</script>
