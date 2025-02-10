
# аутинфикация и получение текста записей на стене

import vk_api

# создаем объект с указанием номера телефона и пароля
vk_ses = vk_api.VkApi('+79165095176', 'Kolobok128mm')

# аутентификация
try:
   vk_ses.auth()
except vk_api.AuthError as error_msg:
   print(error_msg)
   exit(0)

# объект для выполнения некоторых операций в соц сети
vk = vk_ses.get_api()

# получить json-объект с информацией
rs = vk.wall.get(count=0, offset=0)
#  сообщения на стене
for t in rs['items']:
     print(t['text'])