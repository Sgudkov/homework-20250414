Diabetes Prediction API
=========================

# Overview
This API uses a pre-trained ONNX model to predict the likelihood of diabetes based on user input.

## API Endpoint
POST /
Predict diabetes likelihood based on user input

### Request Body
Fields
* Pregnancies: int (>= 0)
* Glucose: int (>= 0)
* BMI: float (>= 0)
* Age: int (>= 0)
### Response
Fields
* description: string (description of the prediction)
* prediction: string (prediction result: 0 = no diabetes, 1 = diabetes)
#### Example Request
```json
{
  "Pregnancies": 1,
  "Glucose": 120,
  "BMI": 25.5,
  "Age": 30
}
```
#### Example Response
```json
{
  "description": "Предсказание (0=нет диабета, 1=есть)",
  "prediction": "1"
}
```
##### Running the API
```bash
uvicorn main:app --host 127.0.0.1 --port 8005 --reload
```
This will start the API server on http://127.0.0.1:8005. You can use a tool like curl to test the API.
 
 