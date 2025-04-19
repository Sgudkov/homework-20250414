import json

from fastapi import FastAPI
from pydantic import BaseModel, Field
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("ml/diabetes_model.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

app = FastAPI()


class UserInfo(BaseModel):
    Pregnancies: int = Field(ge=0)
    Glucose: int = Field(ge=0)
    BMI: float = Field(ge=0)
    Age: int = Field(ge=0)


@app.post("/")
async def root(user_info: UserInfo):
    data = np.array([[user_info.Pregnancies, user_info.Glucose, user_info.BMI, user_info.Age]], dtype=np.float32)
    output = session.run(None, {input_name: data})

    response = {}
    response["description"] = "Предсказание (0=нет диабета, 1=есть)"
    response["prediction"] = str(int(output[0][0] > 0.5))
    print(json.dumps(response, ensure_ascii=False))
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8005, reload=True)
