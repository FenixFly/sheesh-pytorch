<template>
  <q-page>
    <div class="q-pa-md row">
      <div class="col-12 q-pa-md">
        <q-card v-if="rgData !== null">
          <q-card-section>
            {{rgData.timeString}}
            <q-space />
            Риск Covid19: {{rgData.risk}}%
          </q-card-section>
        </q-card>
      </div>
      <div class="col-12 q-pa-md">
        <q-card v-if="rgData !== null">
          <q-card-section style="height: 500px;">
            <RhythmogramZoom :rgData="rgData"/>
          </q-card-section>
        </q-card>
      </div>
      <div class="col-12 q-pa-md" v-if="ecgData !== null">
        <h6 style="margin-bottom: 5px; margin-top: 15px;">Кардиограмма</h6>
        <q-card>
          <q-card-section style="height: 500px;">
            <Cardiogram :ecgData="ecgData"/>
          </q-card-section>
        </q-card>
      </div>
    </div>
  </q-page>
</template>

<script>
import Cardiogram from '../components/Cardiogram.vue';
import RhythmogramZoom from '../components/RhythmogramZoom.vue';

export default {
  name: 'PageRhythmogramZoom',
  components: {
    Cardiogram,
    RhythmogramZoom,
  },
  data:() => ({
    rgData: null,
    ecgData: null,
  }),
  methods: {
  },
  async created(){
    //let response = await fetch(this.$root.api.rhythmogramUrl + '/' + id);
    let response = await fetch(this.$root.api.rhythmogramUrl);
    if (response.ok) {
      this.rgData = await response.json();
    } else {
      alert("Ошибка HTTP: " + response.status);
    }
    if(!('cardiogramId' in this.rgData) || this.rgData.cardiogramId === null) return 0;
    //return 0;
    //response = await fetch(this.$root.api.cardiogramUrl + '/' + this.rgData.cardiogramId);
    response = await fetch(this.$root.api.cardiogramUrl);
    if (response.ok) {
      this.ecgData = await response.json();
      console.log(this.ecgData);
    } else {
      alert("Ошибка HTTP: " + response.status);
    }

  },
}
</script>
