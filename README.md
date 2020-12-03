# Weather Radar data converter

### jupyter
* Make container for jupyter or playground  
```
$ docker-compose run -p 8008:8008 --name=data-weather-radar_master_run data-weather-radar bash
```

* Settings on docker
```
# jupyter notebook --generate-config
# jupyter notebook password
```

* launch
```
# jupyter lab --ip=0.0.0.0 --port=8008 --allow-root
```
