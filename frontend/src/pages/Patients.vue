<template>
  <q-page>
    <div class="q-pa-md row">
      <div class="col-12 q-pa-md">
        <q-card >
            <q-table
              title="Пациенты"
              :rows="rows"
              :columns="columns"
              :filter="filter"
              row-key="name"
            >
              <template v-slot:top>
                <q-space />
                <q-input borderless dense debounce="300" color="primary" v-model="filter" placeholder="Поиск">
                  <template v-slot:append>
                    <q-icon name="search" />
                  </template>
                </q-input>
              </template>
              <template v-slot:body-cell="props">
                <q-td :props="props" @click.native="$router.push(`/patient/${props.row.id}`)" style="cursor: pointer;">
                  <div>{{ props.value }}</div>
                </q-td>
              </template>
            </q-table>
        </q-card>
      </div>
    </div>
  </q-page>
</template>

<script>
import { Notify } from 'quasar'

export default {
  name: 'PagePatients',
  components: {

  },
  data:() => ({
    rows: [],
    columns: [
      {
        name: 'name',
        field: 'name',
        label: 'Имя',
        align: 'left',
        sortable: true
      },
      {
        name: 'age',
        field: 'age',
        label: 'Возраст',
        align: 'left',
        sortable: true
      },
      {
        name: 'lastRgTimeString',
        field: 'lastRgTimeString',
        label: 'Дата последней ртмограммы',
        align: 'left',
        sortable: true
      },
      {
        name: 'risk',
        field: 'risk',
        label: 'Риск осложнений',
        align: 'left',
        format: (val, row) => `${val}%`,
        sortable: true
      },

    ],
    filter: '',
  }),
  methods: {
  },
  async created(){
    try {
      let response = await fetch(this.$root.api.patientsUrl);
      if (response.ok) {
        this.rows = await response.json();
      } else {
        Notify.create('Что-то пошло не так, Попробуйте перезагрузить страницу (ошибка' + response.status + ')');
      }
    } catch (e) {
      Notify.create('Что-то пошло не так, Попробуйте перезагрузить страницу');
    }
  },
}
</script>
