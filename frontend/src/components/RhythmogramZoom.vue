<template>
  <canvas ref="chart" :width="width+'%'" height="500px"></canvas>
  <a @click="zoom()">увеличить</a>
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
      width: 500,
    }
  },
  methods:{
    zoom(){
      this.chart.config.options.animation.duration = 0;
      this.chart.update();
    },
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
              min: 10000,    // minimum will be 0, unless there is a lower value.
              max: 20000
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
