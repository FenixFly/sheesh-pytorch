<template>
  <q-page>
    <div class="q-pa-md row">
      <div class="col-12 q-pa-md">
            <q-list bordered class="rounded-borders" >
              <q-item-label header>Оповещения</q-item-label>

              <q-item clickable v-ripple v-for="notification in notifications" :key="notification.id" :to="notification.link">
                <q-item-section avatar>
                  <q-avatar>
                    <img :src="notification.icon">
                  </q-avatar>
                </q-item-section>

                <q-item-section>
                  <q-item-label lines="1">{{notification.text}}</q-item-label>
                  <q-item-label caption lines="2">
                    {{notification.subText}}
                  </q-item-label>
                </q-item-section>

                <q-item-section side top>
                  {{notification.timeString}}
                </q-item-section>
              </q-item>
              <q-separator inset="item" />

            </q-list>
      </div>
      <div class="col-12 col-md-6 q-pa-md">
        <q-card style="height: 500px;">
          <Doughnut :chartData="healthDoughnutData" v-if="healthDoughnutData !== null"/>
        </q-card>
      </div>
      <div class="col-12 col-md-6 q-pa-md">
        <q-card style="height: 500px;">
            <BarChart :barData="problemsBar" v-if="problemsBar !== null"/>
        </q-card>
      </div>
    </div>
  </q-page>
</template>

<script>
/*
TODO: перенести оповещения в компонент

 */
import Doughnut from '../components/Doughnut.vue';
import BarChart from '../components/BarChart.vue';

export default {
  name: 'PageIndex',
  components: {
    Doughnut,
    BarChart,
  },
  data:() => ({
    notifications: [],
    healthDoughnutData: null,
    problemsBar: {
      labels: ["0%", "10%", "20%" , "30%", "40%"],
      datasets: [
        {
          label: "Ice Cream Sales ",
          fill: true,
          backgroundColor:'red',
          hoverBackgroundColor: 'blue',
          data: [50, 30, 12, 5, 3]
        }
      ]
    },
  }),
  methods: {

  },
  async created(){
    let response = await fetch(this.$root.api.notificationsUrl);
    if (response.ok) {
      this.notifications = await response.json();
    } else {
      alert("Ошибка HTTP: " + response.status);
    }

    response = await fetch(this.$root.api.healthDoughnutDataUrl);
    if (response.ok) {
      this.healthDoughnutData = await response.json();
    } else {
      alert("Ошибка HTTP: " + response.status);
    }

    response = await fetch(this.$root.api.problemsBarUrl);
    if (response.ok) {
      this.problemsBar = await response.json();
    } else {
      alert("Ошибка HTTP: " + response.status);
    }



  },
}
</script>
