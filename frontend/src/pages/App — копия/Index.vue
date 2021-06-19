<template>
  <q-page >
    <div class="q-pa-sm row">
      <div class="col-12 q-pa-sm">
        <q-card class="big-card">
          <q-card-section>
            <div class="text-h6">Процесс восстановления</div>
          </q-card-section>
          <mini-health-chart />
        </q-card>
      </div>
      <div class="col-12 col-md-4 q-pa-sm">
        <q-card class="big-card phone-sensor-card" @click="$router.push('test')">
        </q-card>
      </div>
      <div class="col-6 col-md-4 q-pa-sm">
        <q-btn color="primary">
          <q-icon left size="3em" name="cloud_upload" />
          <div>Загрузить ритмограмму</div>
        </q-btn>
      </div>
      <div class="col-6 col-md-4 q-pa-sm">
        <q-btn color="grey">
          <q-icon left size="3em" name="watch" />
          <div>Прикрепить устройство <br> <span style="font-size: 80%;text-transform: none;">в разработке</span></div>
        </q-btn>
      </div>

    </div>

  </q-page>
</template>


<script>
import MiniHealthChart from '../../components/MiniHealthChart';

export default {
  name: 'PageIndex',
  components: {
    MiniHealthChart,
  },
  data:() => ({

  }),
  methods: {

  },
}
</script>

<style scoped>
.mini-card{
  min-height: 70px;
  border-radius: 20px;
}
.big-card{
  min-height: 150px;
  border-radius: 20px;
  overflow: hidden;
  padding: 0px;
}
.phone-sensor-card{
  background-image: url(https://cardiospike.ip3.ru/img/camera2.jpg);
  background-size: contain;
  background-position-y: top;
  background-position-x: center;
  background-repeat: no-repeat;
  background-color: #1976D2;
}
</style>
