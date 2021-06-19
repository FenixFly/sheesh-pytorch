<template>
  <q-page>
    <div class="q-pa-md row">
      <div class="col-12 col-md-3 q-pa-md">
        <q-card >
          <q-card-section>
            <div class="q-row">
              <div class="col-12 " style="width: 90%; margin: auto; text-align: center;">
                <q-avatar color="primary" text-color="white" style="width: 100px; height: 100px; font-size: 500%;">
                <!--<q-avatar style="width: 100%; height: auto;">
                  <img src="https://cdn.quasar.dev/img/avatar.png">-->
                  <span v-if="name && name.length > 0">{{name[0]}}</span>
                </q-avatar>
              </div>

            </div>
            <div class="q-row" style="text-align: center;">
              <h6 style="margin: 15px;">{{name}}</h6>


            </div>

          </q-card-section>
          <q-list bordered separator>
            <q-item v-ripple >
              <q-item-section>Риск осложнений</q-item-section>
              <q-item-section side :style="'color: '+color"><b>{{risk}}%</b></q-item-section>
            </q-item>
            <q-item v-ripple >
              <q-item-section>Возраст</q-item-section>
              <q-item-section side><b>{{age}}</b></q-item-section>
            </q-item>
            <q-item v-ripple >
              <q-item-section>Последнее измерение</q-item-section>
              <q-item-section side><b>{{lastRgTimeString}}</b></q-item-section>
            </q-item>


          </q-list>
        </q-card>
      </div>
      <div class="col-12 col-md-9 q-pa-md" v-if="rhythmogramsToShow.length > 0">
        <div class="row">
          <h6 style="margin-top: 0px; margin-bottom: 15px;">Последние ритмограммы</h6>
          <div class="col-12">
            <div v-for="rhythmogram in rhythmogramsToShow"
                 :key="rhythmogram.id"
                 style="cursor: pointer;"
                 @click="$router.push('/rhythmogram/'+rhythmogram.id)">
              <div>{{rhythmogram.timeString}}</div>
              <q-card style="height: 210px; margin-top: 5px; margin-bottom: 15px; padding: 15px;">
                <rhythmogram :rgData="rhythmogram" static="true"/>
              </q-card>
            </div>

          </div>

        </div>
      </div>
      <div class="col-12 col-md-8 q-pa-md">
        <q-file label="Загрузить ритмограмму" outlined v-model="fileToLoad">
          <template v-slot:prepend>
            <q-icon name="attach_file" />
          </template>
        </q-file>
      </div>
      <div class="col-12 col-md-4 q-pa-md">
        <q-btn color="primary" class="full-width" label="Проверить" icon="cloud_upload" size="lg" :disabled="fileToLoad === null" @click="sendRhythmogram()" v-if="!isLoading"/>
        <q-circular-progress
          indeterminate
          size="lg"
          v-else
          style="padding-top: 10px;"
        />
      </div>
      <div class="col-12 q-pa-md">
        <q-btn color="grey" class="full-width" label="скачать пример файла ритмограммы" icon="attach_file" size="md" @click="downloadTest()" />
      </div>

      <div class="col-12 q-pa-md">
        <q-card >
          <q-list bordered class="rounded-borders" >
            <q-item-label header>Все ритмограммы</q-item-label>

            <q-item clickable v-ripple v-for="info in rhythmogramsInfo" :key="info.id" :to="'/rhythmogram/'+info.id">
              <q-item-section>
                <q-item-label lines="1">ритмограмма {{info.id}}</q-item-label>
              </q-item-section>

              <q-item-section side top>
                {{info.timeString}}
              </q-item-section>
            </q-item>
            <q-separator inset="item" />

          </q-list>
        </q-card>
      </div>

    </div>
  </q-page>

  <q-dialog v-model="showNewRhythmogramChart" full-width>
    <q-card style="height: 210px; margin-top: 5px; margin-bottom: 15px; padding: 15px;">
      <rhythmogram :rgData="newRhythmogramData" static="true"/>
    </q-card>
  </q-dialog>
</template>

<script>
import Rhythmogram from '../components/Rhythmogram.vue';
import {Notify} from "quasar";

export default {
  name: 'PagePatient',
  components: {
    Rhythmogram
  },
  data:() => ({
    id: null,
    name: null,
    age: null,
    lastRgTimeString: null,
    risk: null,
    photo: null,
    rhythmogramIds: [],
    rhythmogramsInfo: [],
    rhythmogramsToShow: [],

    fileToLoad: null,
    color: 'green',
    isLoading: false,
    showNewRhythmogramChart: false,
    newRhythmogramData: null,
  }),
  methods: {
    downloadTest(){
      window.open('https://cardiospike.ip3.ru/test_rhythmogram.csv', '_blank').focus();
    },

    async sendRhythmogram(){
      console.log('отправляем ритмограмму');
      console.log('fileToLoad', this.fileToLoad);

      const formData = new FormData()

      formData.append('rgfile ', this.fileToLoad)
      formData.append('id', this.$root.api.uid);
      formData.append('version', this.$root.api.version);
      this.isLoading = true;

      let response = await fetch(this.$root.api.loadRhythmogramUrl, {
        method: 'POST',
        cache: 'no-cache',
        mode: 'cors',
        body: formData,
      });
      if (response.ok) {
        let userInfo = await response.json();
        //window.userInfo = userInfo;
        this.newRhythmogramData = userInfo;
        //window.newRhythmogramData = this.newRhythmogramData;
        this.showNewRhythmogramChart = true;
        console.log('ответ', userInfo);

        this.rhythmogramsToShow.unshift(userInfo);
        this.rhythmogramsToShow.pop();

        this.rhythmogramsInfo.push( {
          "id": userInfo.id,
          "timeString": userInfo.timeString,
          "risk": userInfo.risk
        });
      } else {
        Notify.create('Что-то пошло не так, Попробуйте перезагрузить страницу');
      }
      this.isLoading = false;
    },

    async loadRhythmogramsToShow(){
      this.rhythmogramsInfo.forEach( rg=>this.rhythmogramIds.push(rg.id) );
      let ids = this.rhythmogramIds.reverse();
      ids = ids.splice(0, 2);
      console.log('ids', ids);

      ids.forEach((id)=>{
        this.loadRhythmogram(id);
      });
    },

    async loadRhythmogram(id){
      //let response = await fetch(this.$root.api.rhythmogramUrl + '/' + id);
      //let response = await fetch(this.$root.api.rhythmogramUrl);
      let response = await fetch(this.$root.api.rhythmogramUrl+`?id=${id}`);
      if (response.ok) {
        let rhythmogram = await response.json();
        console.log('rhythmogram', rhythmogram);
        this.rhythmogramsToShow.push(rhythmogram);
      } else {
        alert("Ошибка HTTP: " + response.status);
      }
    },
  },
  async created(){
    //let response = await fetch(this.$root.api.patientUrl+'/'+this.$route.params.id);
    //let response = await fetch(this.$root.api.patientUrl);
    let response = await fetch(this.$root.api.patientUrl+`?id=${this.$route.params.id}`);
    if (response.ok) {
      let patientInfo = await response.json();
      this.id = patientInfo.id;
      this.name = patientInfo.name;
      this.age = patientInfo.age;
      this.lastRgTimeString = patientInfo.lastRgTimeString;
      this.risk = patientInfo.risk;
      this.photo = patientInfo.photo;
      this.rhythmogramsInfo = patientInfo.rhythmograms;
      //this.name = patientInfo.name;

    } else {
      alert("Ошибка HTTP: " + response.status);
    }

    await this.loadRhythmogramsToShow();

  },
}
</script>
