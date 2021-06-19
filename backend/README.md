# Сборка и установка бэкенда для Cardiospike 

## Structure of the README

- [Folder structure](#Folder-structure)
- [Build](#Build)
- [Deploy](#Deploy)

### Folder structure
Исходники бэкенда такой структуры:

```
├── httpserver
│   ├── bin
│   ├── filestorage
│   ├── ...etc
└── httpserver-bin-folder-source
    ├── CardioSpike
    └── CardioSpikeTest
```

`httpserver/bin` - сюда развертывать dllки собранные из `httpserver-bin-folder-source`.
`httpserver/filestorage` - здесь база ритмограмм
`httpserver-bin-folder-source` - это билдить

### Build
Собрать `httpserver-bin-folder-source/CardioSpike`, в Visual Studio или чем хотите.
В исходниках есть несколько строковых констант. Задайте их как вам нужно

### Deploy 
Залить папку `httpserver` на сервер лучше под IIS но выбирайте сами.
В папку `httpserver/bin` докинуть что вы сбилдили на прошлом шаге.
В папку `httpserver/filestorage` можно добавить пользователей и их ритмограммы, если у вас был архив
Должно работать