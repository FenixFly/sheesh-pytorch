<template>
  <q-layout view="lHh Lpr lFf">
    <q-header elevated>
      <q-toolbar>

        <q-tabs v-if="this.$root.api.role === 'doctor'">
          <q-route-tab
            icon="home"
            to="/"
            label="Личный кабинет"
            exact
          />
          <q-route-tab
            icon="people"
            to="/patients"
            label="Пациенты"
            exact
          />
        </q-tabs>
        <q-space />

        <q-toolbar-title class="text-right">
          <q-img src="https://cardiospike.ip3.ru/img/logo2.png" style="max-height: 64px;" fit="scale-down " position="right" class="mobile-hide"/>
        </q-toolbar-title>




      </q-toolbar>
    </q-header>

    <q-page-container>
      <router-view />
    </q-page-container>
  </q-layout>
</template>

<script>
export default {
  name: 'MainLayout',
  components: {
  },
  data:()=>({}),
  created() {
    let initDataJson = localStorage.getItem('init');
    if(initDataJson !== null){
      let initData = JSON.parse(initDataJson);
      this.$root.api.login = initData.login;
      this.$root.api.uid = initData.uid;
      this.$root.api.role = initData.role;
    }
    if(this.$root.api.uid === null) this.$router.push('/login');
  }
}
</script>
