# app/schemas/result.py
from pydantic import BaseModel

class Result(BaseModel):
    first_name: str
    last_name: str
    date_of_birth: str
    id_number: str
    #metadata: str