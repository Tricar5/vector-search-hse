## Приложение


##Авторизация



### Генерация ключей
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


## Метрики


Метрики торчат по адресу `/stats`

### Метрики длительности запросов:

vector_search_app_api_request_duration_seconds_count{handler="/api/v1/forward",method="POST",status="200"} 3.0
vector_search_app_api_request_duration_seconds_sum{handler="/api/v1/forward",method="POST",status="200"} 1.6549530839984072

Также есть квартили, среднее получается деление одной на другую

### Кастомные метрики для модели:

### Длина текстов
1. vector_search_app_search_text_query_length_count 3.0
2. vector_search_app_search_text_query_length_sum 23.0

Среднее можно получить делением одной на другую

### Кол-во токенов
1. vector_search_app_search_text_query_tokens_count 3.0
2. vector_search_app_search_text_query_tokens_sum 5.0

Среднее можно получить делением одной на другую
