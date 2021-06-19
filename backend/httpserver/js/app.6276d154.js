(()=>{"use strict";var e={2323:(e,t,r)=>{r(5363),r(71);var n=r(8880),o=r(9592),a=r(3673);function i(e,t,r,n,o,i){const l=(0,a.up)("router-view");return(0,a.wg)(),(0,a.j4)(l)}const l={name:"App",data:()=>({test:"t",api:{login:"",uid:null,role:"",version:"cardiospike01",notificationsUrl:"https://cardiospike.ip3.ru/spa/notifications.json",healthDoughnutDataUrl:"https://cardiospike.ip3.ru/spa/health-doughnut-data.json",rhythmogramUrl:"https://cardiospike.ip3.ru/rg/",cardiogramUrl:"https://cardiospike.ip3.ru/spa/ecg.json",patientsUrl:"https://cardiospike.ip3.ru/patients/",patientUrl:"https://cardiospike.ip3.ru/patient/",problemsBarUrl:"https://cardiospike.ip3.ru/spa/problemsBar.json",logInUrl:"https://cardiospike.ip3.ru/login/",loadRhythmogramUrl:"https://cardiospike.ip3.ru/uploadrg/"}})};l.render=i;const s=l;var p=r(7083),c=r(9582);const d=[{path:"/",component:()=>Promise.all([r.e(736),r.e(579)]).then(r.bind(r,1579)),children:[{path:"",component:()=>Promise.all([r.e(736),r.e(53)]).then(r.bind(r,5027))},{path:"patient/:id",component:()=>Promise.all([r.e(736),r.e(978)]).then(r.bind(r,9363))},{path:"patients",component:()=>Promise.all([r.e(736),r.e(921)]).then(r.bind(r,7921))},{path:"rhythmogram/:id",component:()=>Promise.all([r.e(736),r.e(334)]).then(r.bind(r,2059))}]},{path:"/app",component:()=>Promise.all([r.e(736),r.e(645)]).then(r.bind(r,2645)),children:[{path:":id",component:()=>Promise.all([r.e(736),r.e(581)]).then(r.bind(r,7313))},{path:"devices",component:()=>Promise.all([r.e(736),r.e(444)]).then(r.bind(r,6796))},{path:"cardiogram/:id",component:()=>Promise.all([r.e(736),r.e(110)]).then(r.bind(r,9110))}]},{path:"/login",component:()=>Promise.all([r.e(736),r.e(811)]).then(r.bind(r,5811))},{path:"/test",component:()=>Promise.all([r.e(736),r.e(757)]).then(r.bind(r,1757))},{path:"/:catchAll(.*)*",component:()=>Promise.all([r.e(736),r.e(243)]).then(r.bind(r,1243))}],u=d,h=(0,p.BC)((function(){const e=c.r5,t=(0,c.p7)({scrollBehavior:()=>({left:0,top:0}),routes:u,history:e("")});return t}));async function f(e,t){const r="function"===typeof h?await h({}):h,n=e(s);return n.use(o.Z,t),{app:n,router:r}}var m=r(9664),b=r(4434);const g={config:{},lang:m.Z,plugins:{Notify:b.Z}},v="";async function y({app:e,router:t},r){let n=!1;const o=e=>{n=!0;const r=Object(e)===e?t.resolve(e).fullPath:e;window.location.href=r},a=window.location.href.replace(window.location.origin,"");for(let l=0;!1===n&&l<r.length;l++)try{await r[l]({app:e,router:t,ssrContext:null,redirect:o,urlPath:a,publicPath:v})}catch(i){return i&&i.url?void(window.location.href=i.url):void console.error("[Quasar] boot error:",i)}!0!==n&&(e.use(t),e.mount("#q-app"))}f(n.ri,g).then((e=>Promise.all([Promise.resolve().then(r.bind(r,4112))]).then((t=>{const r=t.map((e=>e.default)).filter((e=>"function"===typeof e));y(e,r)}))))},4112:(e,t,r)=>{r.r(t),r.d(t,{default:()=>s,i18n:()=>l});var n=r(7083),o=r(5948);const a={failed:"Action failed",success:"Action was successful"},i={"en-US":a},l=(0,o.o)({locale:"en-US",messages:i}),s=(0,n.xr)((({app:e})=>{e.use(l)}))}},t={};function r(n){var o=t[n];if(void 0!==o)return o.exports;var a=t[n]={id:n,loaded:!1,exports:{}};return e[n].call(a.exports,a,a.exports,r),a.loaded=!0,a.exports}r.m=e,(()=>{var e=[];r.O=(t,n,o,a)=>{if(!n){var i=1/0;for(p=0;p<e.length;p++){for(var[n,o,a]=e[p],l=!0,s=0;s<n.length;s++)(!1&a||i>=a)&&Object.keys(r.O).every((e=>r.O[e](n[s])))?n.splice(s--,1):(l=!1,a<i&&(i=a));l&&(e.splice(p--,1),t=o())}return t}a=a||0;for(var p=e.length;p>0&&e[p-1][2]>a;p--)e[p]=e[p-1];e[p]=[n,o,a]}})(),(()=>{r.n=e=>{var t=e&&e.__esModule?()=>e["default"]:()=>e;return r.d(t,{a:t}),t}})(),(()=>{r.d=(e,t)=>{for(var n in t)r.o(t,n)&&!r.o(e,n)&&Object.defineProperty(e,n,{enumerable:!0,get:t[n]})}})(),(()=>{r.f={},r.e=e=>Promise.all(Object.keys(r.f).reduce(((t,n)=>(r.f[n](e,t),t)),[]))})(),(()=>{r.u=e=>"js/"+e+"."+{53:"37d1e4e8",110:"1c5f957b",243:"475154a7",334:"aeb2ca84",444:"e39a65e2",579:"d91ca0dc",581:"0f04efe8",645:"26abf9ce",757:"897ac054",811:"133e83b8",921:"e11e753e",978:"d51671cd"}[e]+".js"})(),(()=>{r.miniCssF=e=>"css/"+(736===e?"vendor":e)+"."+{581:"8ae5a0c9",736:"0e6876d9",811:"dfe78c9b"}[e]+".css"})(),(()=>{r.g=function(){if("object"===typeof globalThis)return globalThis;try{return this||new Function("return this")()}catch(e){if("object"===typeof window)return window}}()})(),(()=>{r.o=(e,t)=>Object.prototype.hasOwnProperty.call(e,t)})(),(()=>{var e={},t="sheesh:";r.l=(n,o,a,i)=>{if(e[n])e[n].push(o);else{var l,s;if(void 0!==a)for(var p=document.getElementsByTagName("script"),c=0;c<p.length;c++){var d=p[c];if(d.getAttribute("src")==n||d.getAttribute("data-webpack")==t+a){l=d;break}}l||(s=!0,l=document.createElement("script"),l.charset="utf-8",l.timeout=120,r.nc&&l.setAttribute("nonce",r.nc),l.setAttribute("data-webpack",t+a),l.src=n),e[n]=[o];var u=(t,r)=>{l.onerror=l.onload=null,clearTimeout(h);var o=e[n];if(delete e[n],l.parentNode&&l.parentNode.removeChild(l),o&&o.forEach((e=>e(r))),t)return t(r)},h=setTimeout(u.bind(null,void 0,{type:"timeout",target:l}),12e4);l.onerror=u.bind(null,l.onerror),l.onload=u.bind(null,l.onload),s&&document.head.appendChild(l)}}})(),(()=>{r.r=e=>{"undefined"!==typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(e,"__esModule",{value:!0})}})(),(()=>{r.nmd=e=>(e.paths=[],e.children||(e.children=[]),e)})(),(()=>{r.p=""})(),(()=>{var e=(e,t,r,n)=>{var o=document.createElement("link");o.rel="stylesheet",o.type="text/css";var a=a=>{if(o.onerror=o.onload=null,"load"===a.type)r();else{var i=a&&("load"===a.type?"missing":a.type),l=a&&a.target&&a.target.href||t,s=new Error("Loading CSS chunk "+e+" failed.\n("+l+")");s.code="CSS_CHUNK_LOAD_FAILED",s.type=i,s.request=l,o.parentNode.removeChild(o),n(s)}};return o.onerror=o.onload=a,o.href=t,document.head.appendChild(o),o},t=(e,t)=>{for(var r=document.getElementsByTagName("link"),n=0;n<r.length;n++){var o=r[n],a=o.getAttribute("data-href")||o.getAttribute("href");if("stylesheet"===o.rel&&(a===e||a===t))return o}var i=document.getElementsByTagName("style");for(n=0;n<i.length;n++){o=i[n],a=o.getAttribute("data-href");if(a===e||a===t)return o}},n=n=>new Promise(((o,a)=>{var i=r.miniCssF(n),l=r.p+i;if(t(i,l))return o();e(n,l,o,a)})),o={143:0};r.f.miniCss=(e,t)=>{var r={581:1,811:1};o[e]?t.push(o[e]):0!==o[e]&&r[e]&&t.push(o[e]=n(e).then((()=>{o[e]=0}),(t=>{throw delete o[e],t})))}})(),(()=>{var e={143:0};r.f.j=(t,n)=>{var o=r.o(e,t)?e[t]:void 0;if(0!==o)if(o)n.push(o[2]);else{var a=new Promise(((r,n)=>o=e[t]=[r,n]));n.push(o[2]=a);var i=r.p+r.u(t),l=new Error,s=n=>{if(r.o(e,t)&&(o=e[t],0!==o&&(e[t]=void 0),o)){var a=n&&("load"===n.type?"missing":n.type),i=n&&n.target&&n.target.src;l.message="Loading chunk "+t+" failed.\n("+a+": "+i+")",l.name="ChunkLoadError",l.type=a,l.request=i,o[1](l)}};r.l(i,s,"chunk-"+t,t)}},r.O.j=t=>0===e[t];var t=(t,n)=>{var o,a,[i,l,s]=n,p=0;for(o in l)r.o(l,o)&&(r.m[o]=l[o]);if(s)var c=s(r);for(t&&t(n);p<i.length;p++)a=i[p],r.o(e,a)&&e[a]&&e[a][0](),e[i[p]]=0;return r.O(c)},n=self["webpackChunksheesh"]=self["webpackChunksheesh"]||[];n.forEach(t.bind(null,0)),n.push=t.bind(null,n.push.bind(n))})();var n=r.O(void 0,[736],(()=>r(2323)));n=r.O(n)})();