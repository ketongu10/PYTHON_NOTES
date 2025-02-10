import vk






vk_api =  vk.UserAPI(user_login='89165095176',user_password='Kolobok128mm',scope='offline,wall',v='5.131')

user1 = vk_api.users.get(user_id=412336153, fields='online, last_seen', scope='wall')
print(user1)
#vk_api.messages.send(users_id=371140133, random_id=1,messages='hello')
