<template>
  <q-page >
    <div class="q-pa-sm row">
      <div class="col-12 q-pa-sm">
        <q-card class="big-card">
          <q-card-section>
            <div class="text-h6">Процесс восстановления</div>
          </q-card-section>
          <health-chart />
        </q-card>
      </div>
      <div class="col-6 q-pa-sm">
        <q-btn color="primary" class="full-width">
          <q-icon left size="3em" name="cloud_upload" />
          <div>Загрузить ритмограмму</div>
        </q-btn>
      </div>
      <div class="col-6 q-pa-sm">
        <q-btn color="grey" class="full-width">
          <q-icon left size="3em" name="watch" />
          <div>Прикрепить устройство <br> <span style="font-size: 80%;text-transform: none;">в разработке</span></div>
        </q-btn>
      </div>

      <div class="col-12 q-pa-sm text-center"><h4>последние загруженные ритмограммы</h4></div>

      <div class="col-12 col-md-6 col-lg-4 q-pa-sm">
        <q-card>
          <q-list>
            <q-item clickable>
              <q-item-section avatar>
                <q-icon color="green" name="favorite" />
              </q-item-section>

              <q-item-section>
                <q-item-label>Хороший результат</q-item-label>
                <q-item-label caption>менее 0.1% отклонений</q-item-label>
              </q-item-section>
            </q-item>

            <q-item clickable>
              <q-item-section avatar>
                <q-icon color="gray" name="date_range" />
              </q-item-section>

              <q-item-section>
                <q-item-label>Дата исследования</q-item-label>
                <q-item-label caption>19.06.2021</q-item-label>
              </q-item-section>
            </q-item>
          </q-list>
        </q-card>


      </div>
      <div class="col-12 col-md-6 col-lg-4 q-pa-sm">
        <q-card>
          <q-list>
            <q-item clickable>
              <q-item-section avatar>
                <q-icon color="orange-14" name="favorite" />
              </q-item-section>

              <q-item-section>
                <q-item-label>Есть проьлемы</q-item-label>
                <q-item-label caption>0.4% отклонений</q-item-label>
              </q-item-section>
            </q-item>

            <q-item clickable>
              <q-item-section avatar>
                <q-icon color="gray" name="date_range" />
              </q-item-section>

              <q-item-section>
                <q-item-label>Дата исследования</q-item-label>
                <q-item-label caption>19.06.2021</q-item-label>
              </q-item-section>
            </q-item>
          </q-list>
        </q-card>


      </div>
      <div class="col-12 col-md-6 col-lg-4 q-pa-sm">
        <q-card>
          <q-list>
            <q-item clickable>
              <q-item-section avatar>
                <q-icon color="red" name="favorite" />
              </q-item-section>

              <q-item-section>
                <q-item-label>Опасность осложнений</q-item-label>
                <q-item-label caption>2% отклонений</q-item-label>
              </q-item-section>
            </q-item>

            <q-item clickable>
              <q-item-section avatar>
                <q-icon color="gray" name="date_range" />
              </q-item-section>

              <q-item-section>
                <q-item-label>Дата исследования</q-item-label>
                <q-item-label caption>19.06.2021</q-item-label>
              </q-item-section>
            </q-item>
          </q-list>
        </q-card>


      </div>
      <div class="col-12 col-md-6 col-lg-4 q-pa-sm">
        <q-card>
          <q-list>
            <q-item clickable>
              <q-item-section avatar>
                <q-icon color="red" name="favorite" />
              </q-item-section>

              <q-item-section>
                <q-item-label>Опасность осложнений</q-item-label>
                <q-item-label caption>2.1% отклонений</q-item-label>
              </q-item-section>
            </q-item>

            <q-item clickable>
              <q-item-section avatar>
                <q-icon color="gray" name="date_range" />
              </q-item-section>

              <q-item-section>
                <q-item-label>Дата исследования</q-item-label>
                <q-item-label caption>19.06.2021</q-item-label>
              </q-item-section>
            </q-item>
          </q-list>
        </q-card>


      </div>

    </div>

  </q-page>
</template>


<script>
import HealthChart from '../../components/HealthChart';

export default {
  name: 'PageIndex',
  components: {
    HealthChart,
  },
  data:() => ({

  }),
  methods: {

  },
}
</script>

<style scoped>

</style>
