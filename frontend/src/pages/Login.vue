<template>
  <q-layout view="lHh Lpr lFf">
    <q-page-container class="bg-grey-11">
      <q-page class="bg-grey-12 window-height window-width row justify-center items-center">
        <div class="column" style="max-width: 98%">
          <div class="row">
            <div style="width: 100%; max-width: 370px;"><img src="https://cardiospike.ip3.ru/img/logo2-blue.png"  style="max-width: 100%; width: 370px;;"/></div>
          </div>
          <div class="row" style="max-width: 100%">
            <q-card square bordered class="q-pa-md-lg q-pa-md-lg shadow-1">
              <q-card-section>
                <q-form class="q-gutter-md">
                  <q-input square filled clearable v-model="login" type="text" label="логин" />
                  <q-input square filled clearable v-model="password" type="password" label="пароль" />
                </q-form>
              </q-card-section>
              <q-card-actions class="q-px-md">
                <q-btn unelevated color="blue-9" size="lg" class="full-width" label="Войти" @click="logIn()"/>
              </q-card-actions>
              <q-card-section class="text-center q-pa-none">
                <br>
                <!--<p class="text-grey-6"><a @click="anonLogin()" style="cursor: pointer;">Войти анонимно</a></p>-->
                <p class="text-grey-6">Используя сервис Вы принимаете условия пользовательского соглашения и политики обработки персональных данных</p>
              </q-card-section>
            </q-card>
          </div>
        </div>
      </q-page>
    </q-page-container>
  </q-layout>
</template>

<script>
import { Notify } from 'quasar'

export default {
  name: 'Login',
  components: {
  },
  data:() => ({
    login: '',
    password: ''
  }),
  methods:{
    async logIn(){
      try {
        let formData = new FormData();
        formData.append('login', this.login);
        formData.append('password', this.password);
        formData.append('version', this.$root.api.version);


        let response = await fetch(this.$root.api.logInUrl, {
          method: 'POST',
          cache: 'no-cache',
          mode: 'cors',
          body: formData,
        });
        console.log('response', response);
        window.response = response;
        if (response.ok) {
          let userInfo = await response.json();
          console.log('ответ', userInfo);
          this.$root.api.login = this.login;
          this.$root.api.uid = userInfo.id;
          this.$root.api.role = 'patient';
          if( userInfo.doctor === true ) this.$root.api.role = 'doctor';
          localStorage.clear();
          localStorage.setItem('init', JSON.stringify({
            login: this.$root.api.login,
            uid: this.$root.api.uid,
            role: this.$root.api.role,
          }));
          if( userInfo.doctor === true ){
            this.$router.push('/patient/testuser');
            return 0;
          }
          this.$router.push('/patient/'+this.$root.api.login);
        } else {
          console.log('-', t);
          Notify.create('Логин или пароль указаны не верно');
        }
      } catch (e) {
        Notify.create('Что-то пошло не так, Попробуйте перезагрузить страницу');
      }
    },
    anonLogin(){
      console.log('анонимный вход');
    },
  },
}
</script>

<style scoped>
.q-card {
  max-width: 360px;
  width: 100%;
}
</style>
