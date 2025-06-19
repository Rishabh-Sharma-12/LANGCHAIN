# from pydantic import BaseModel

# class student(BaseModel):
#     name:str
    
# new_s={"name":"Rishabh Sharma"}
# Student=student(**new_s)
# print(Student)

#-------------------------------------------------------------

# from pydantic import BaseModel
# from typing import Optional

# class student(BaseModel):
#     name:str="Rishabh"
#     age:Optional[int]=1112
    
# new_s2={}#will return default values
# new_s2={'name':"RAmesh",'age':33}
# Student=student(**new_s2)
# print(Student)
    
#-------------------------------------------------------------
#issue with the venv change the name rm rf
from pydantic import EmailStr,BaseModel

class Student(BaseModel):
    email:EmailStr

s_2={"email":"rishabh_222@gmail.com"}
student=Student(**s_2)
print(s_2)