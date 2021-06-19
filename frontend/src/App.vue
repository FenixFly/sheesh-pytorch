<template>
  <router-view />
</template>
<script>

export default {
  name: 'App',
  data: ()=>({
    test: 't',
    api: {
      login: '',
      uid: null,
      role: '',
      version: 'cardiospike01',

      notificationsUrl: 'https://cardiospike.ip3.ru/notifications.json',
      healthDoughnutDataUrl: 'https://cardiospike.ip3.ru/health-doughnut-data.json',
      rhythmogramUrl: 'https://cardiospike.ip3.ru/rg/',
      cardiogramUrl: 'https://cardiospike.ip3.ru/ecg.json',
      patientsUrl: 'https://cardiospike.ip3.ru/patients/',
      patientUrl: 'https://cardiospike.ip3.ru/patient/',
      problemsBarUrl: 'https://cardiospike.ip3.ru/problemsBar.json',
      logInUrl: 'https://cardiospike.ip3.ru/login/',
      loadRhythmogramUrl: 'https://cardiospike.ip3.ru/uploadrg/',
    },
  }),
}
</script>
