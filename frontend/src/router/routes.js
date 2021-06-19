
const routes = [
  {
    path: '/',
    component: () => import('layouts/MainLayout.vue'),
    children: [
      { path: '', component: () => import('pages/Index.vue') },
      { path: 'patient/:id', component: () => import('pages/Patient.vue') },
      { path: 'patients', component: () => import('pages/Patients.vue') },
      { path: 'rhythmogram/:id', component: () => import('pages/Rhythmogram.vue') },
    ]
  },
  {
    path: '/app',
    component: () => import('layouts/AppLayout.vue'),
    children: [
      { path: ':id', component: () => import('pages/App/Index.vue') },
      { path: 'devices', component: () => import('pages/App/Devices.vue') },
      { path: 'cardiogram/:id', component: () => import('pages/App/Cardiogram.vue') },
    ]
  },
  {
    path: '/login',
    component: () => import('pages/Login.vue')
  },

  // Always leave this as last one,
  // but you can also remove it
  {
    path: '/:catchAll(.*)*',
    component: () => import('pages/Error404.vue')
  }
]

export default routes
