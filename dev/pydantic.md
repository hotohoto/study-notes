# Pydantic

## An example

```python
import typing as t
from pydantic import BaseModel, model_validator

class UserModel(BaseModel):
    user_id: str | None = None
    username: str
    password: str
    password_repeat: str

    def model_post_init(self, __context: t.Any) -> None:
        prefix = __context.get("id_prefix", "USER_") if __context else "USER_"
        
        self.user_id = f"{prefix}{self.username.lower()}"
    
    @model_validator(mode='after')
    def validate_passwords_and_conditions(self) -> 'UserModel':
        if self.password != self.password_repeat:
            raise ValueError('Passwords do not match')
        
        if self.username.lower() in self.password.lower():
            raise ValueError('Password contains username')
        
        return self


user = UserModel(
    username="JohnDoe",
    password="secure123",
    password_repeat="secure123",
    context={"id_prefix": "CUST_"}  # __context로 전달됨
)
```

```python
from PIL import Image

class UserModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    profile_image: Image    
```
