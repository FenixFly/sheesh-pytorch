(self["webpackChunksheesh"]=self["webpackChunksheesh"]||[]).push([[882],{1882:(e,o,a)=>{"use strict";a.r(o),a.d(o,{default:()=>x});var l=a(3673);const s=(0,l.HX)("data-v-75ac88c2");(0,l.dD)("data-v-75ac88c2");const i={class:"column",style:{"max-width":"98%"}},r=(0,l.Wm)("div",{class:"row"},[(0,l.Wm)("div",{style:{width:"100%","max-width":"370px"}},[(0,l.Wm)("img",{src:"https://cardiospike.ip3.ru/spa/img/logo2-blue.png",style:{"max-width":"100%",width:"370px"}})])],-1),n={class:"row",style:{"max-width":"100%"}},d=(0,l.Wm)("br",null,null,-1),c=(0,l.Wm)("p",{class:"text-grey-6"},"Используя сервис Вы принимаете условия пользовательского соглашения и политики обработки персональных данных",-1);(0,l.Cn)();const p=s(((e,o,t,a,p,u)=>{const m=(0,l.up)("q-input"),h=(0,l.up)("q-form"),g=(0,l.up)("q-card-section"),w=(0,l.up)("q-btn"),f=(0,l.up)("q-card-actions"),q=(0,l.up)("q-card"),W=(0,l.up)("q-page"),y=(0,l.up)("q-page-container"),b=(0,l.up)("q-layout");return(0,l.wg)(),(0,l.j4)(b,{view:"lHh Lpr lFf"},{default:s((()=>[(0,l.Wm)(y,{class:"bg-grey-11"},{default:s((()=>[(0,l.Wm)(W,{class:"bg-grey-12 window-height window-width row justify-center items-center"},{default:s((()=>[(0,l.Wm)("div",i,[r,(0,l.Wm)("div",n,[(0,l.Wm)(q,{square:"",bordered:"",class:"q-pa-md-lg q-pa-md-lg shadow-1"},{default:s((()=>[(0,l.Wm)(g,null,{default:s((()=>[(0,l.Wm)(h,{class:"q-gutter-md"},{default:s((()=>[(0,l.Wm)(m,{square:"",filled:"",clearable:"",modelValue:e.login,"onUpdate:modelValue":o[1]||(o[1]=o=>e.login=o),type:"text",label:"логин"},null,8,["modelValue"]),(0,l.Wm)(m,{square:"",filled:"",clearable:"",modelValue:e.password,"onUpdate:modelValue":o[2]||(o[2]=o=>e.password=o),type:"password",label:"пароль"},null,8,["modelValue"])])),_:1})])),_:1}),(0,l.Wm)(f,{class:"q-px-md"},{default:s((()=>[(0,l.Wm)(w,{unelevated:"",color:"blue-9",size:"lg",class:"full-width",label:"Войти",onClick:o[3]||(o[3]=e=>u.logIn())})])),_:1}),(0,l.Wm)(g,{class:"text-center q-pa-none"},{default:s((()=>[d,c])),_:1})])),_:1})])])])),_:1})])),_:1})])),_:1})}));var u=a(4434);const m={name:"Login",components:{},data:()=>({login:"",password:""}),methods:{async logIn(){try{let e=new FormData;e.append("login",this.login),e.append("password",this.password),e.append("version",this.$root.api.version);let o=await fetch(this.$root.api.logInUrl,{method:"POST",cache:"no-cache",mode:"cors",body:e});if(console.log("response",o),window.response=o,o.ok){let e=await o.json();if(console.log("ответ",e),this.$root.api.login=this.login,this.$root.api.uid=e.id,this.$root.api.role="patient",!0===e.doctor&&(this.$root.api.role="doctor"),localStorage.clear(),localStorage.setItem("init",JSON.stringify({login:this.$root.api.login,uid:this.$root.api.uid,role:this.$root.api.role})),!0===e.doctor)return this.$router.push("/patient/testuser"),0;this.$router.push("/app/"+this.$root.api.uid)}else console.log("-",t),u.Z.create("Логин или пароль указаны не верно")}catch(e){u.Z.create("Что-то пошло не так, Попробуйте перезагрузить страницу")}},anonLogin(){console.log("анонимный вход")}}};var h=a(3066),g=a(2652),w=a(4379),f=a(151),q=a(5589),W=a(5269),y=a(4689),b=a(9367),v=a(8240),$=a(7518),Z=a.n($);m.render=p,m.__scopeId="data-v-75ac88c2";const x=m;Z()(m,"components",{QLayout:h.Z,QPageContainer:g.Z,QPage:w.Z,QCard:f.Z,QCardSection:q.Z,QForm:W.Z,QInput:y.Z,QCardActions:b.Z,QBtn:v.Z})}}]);