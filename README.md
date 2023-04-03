# Room Occupnacy Prediction API 

API endpoint for a room occupancy machine learning model, capable of predicting the number of persons (`P.S limited for use in a small space where maximum occupancy is just 3 persons at a time`) in a room based on climate parameters such as CO2, Temperature, Sound, Motion Sensor readings. 

This project was developed using `Python` v "^3.9" and `Flask` v "2.2.3".

Deployed on a `Digital Oceans` Droplet using `Github Actions` for CI/CD.

You can clone project and customise at your end.

### API Documentation

- 'http://127.0.0.1:7000/prediction' Endpoint

METHOD: 'POST'

BODY: {

  - "Temp": temperature readings in degrees Celsius,
  - "Light": light readings in Lux,
  - "Sound": sound readings in Volts,
  - "PIR": motion sensor readings (0 - no motion, 1 - motion detected),
  - "Day_Period": time of day (0 - morning, 1 - afternoon, 2 - evening),
  - "S5_CO2": CO2 readings in PPM

  }

SUCCESS RESPONSE (200): {'success': true, 'prediction': [0 - 3]}

ERROR RESPONSE (4**, 5**): {'success': false, 'error': '***********'}
