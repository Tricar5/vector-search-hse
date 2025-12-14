# Авторизация

## Генерация ключей
private:
```bash
openssl genrsa -out test.pem 1024
```
public:
```bash
openssl rsa -in test.pem -pubout > test.pub
```

Удобное копирование в конфиг
```bash
cat test.pub | awk '{printf "%s\\n", $0}' | pbcopy
```
Установим приватный ключ в переменные окржуения
```bash
export AUTH_PRIVATE_KEY=$(cat test.pem | awk '{printf "%s\\n", $0}')
```

## Генерация токена
```python
import os
from jose import constants, jwt
jwt.encode({'u':'hse-vector', 'i': 'vector-search-service'},key=os.getenv('AUTH_PRIVATE_KEY').replace('\\n', '\n'),algorithm=constants.ALGORITHMS.RS256)
```
