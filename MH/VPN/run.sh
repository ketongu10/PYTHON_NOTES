# https://habr.com/ru/articles/839418/

# Самый трудоемкий вариант, где будем настраивать VPN на сервере с Ubuntu 24.04 сами.
# Из протоколов выберем проверенный временем OpenVPN,
# поддержка которого заявлена в большинстве современных роутеров.
# Подключаемся к VPS как root, создаем нового пользователя и добавляем его в sudo:
adduser user
usermod -aG sudo user

# Затем входим на сервер как user и выполняем все дальнейшие команды от его имени.
# Обновим списки пакетов и установим OpenVPN и Easy-RSA (для управления сертификатами в инфраструктуре открытых ключей):

sudo apt update -y
sudo apt install openvpn easy-rsa -y

# В папке нашего пользователя создаем директорию с символической ссылкой и нужными правами:
mkdir ~/easy-rsa
ln -s /usr/share/easy-rsa/* ~/easy-rsa/
chmod 700 ~/easy-rsa

# Создаем конфигурационный файл для Easy-RSA и инициализируем инфраструктуру открытых ключей (PKI):
cd ~/easy-rsa
echo -e 'set_var EASYRSA_ALGO ec\nset_var EASYRSA_DIGEST sha512' > vars
./easyrsa init-pki

# Сгенерируем ключи удостоверяющего центра:
./easyrsa build-ca nopass

# Система запросит ввести универсальное имя, здесь можно просто нажать Enter.
# Выпустим и подпишем пару "ключ-сертификат" для сервера:
./easyrsa gen-req server nopass
./easyrsa sign-req server server

# При выполнении первой команды вас попросят указать Common Name, здесь просто нажмите Enter. Для второй команды запрос нужно подтвердить, введя yes.
# Скопируем созданные файлы в каталог OpenVPN:
sudo cp ~/easy-rsa/pki/private/server.key /etc/openvpn/server
sudo cp ~/easy-rsa/pki/issued/server.crt /etc/openvpn/server
sudo cp ~/easy-rsa/pki/ca.crt /etc/openvpn/server

# Для дополнительной защиты, создадим предварительный общий ключ (PSK), который будет использоваться с директивой tls-crypt:
sudo openvpn --genkey secret /etc/openvpn/server/ta.key

# Выпустим и подпишем пару "ключ-сертификат" для клиента client1:
./easyrsa gen-req client1 nopass    # В первой команде на требование указать Common Name нажмите Enter
./easyrsa sign-req client client1   # при выполнении второй команды подтвердите запрос вводом yes.

# Создадим директорию для клиентских конфигов, скопируем туда нужные файлы и установим для них соответствующие права:
mkdir ~/openvpn-clients
chmod -R 700 ~/openvpn-clients
cp ~/easy-rsa/pki/private/client1.key ~/openvpn-clients/
cp ~/easy-rsa/pki/issued/client1.crt ~/openvpn-clients/
sudo cp /etc/openvpn/server/{ca.crt,ta.key} ~/openvpn-clients/
sudo chown user ~/openvpn-clients/*

# Настроим конфиг OpenVPN на основе дефолтного примера. Для этого скопируем шаблонный файл server.conf в рабочую директорию:
sudo cp /usr/share/doc/openvpn/examples/sample-config-files/server.conf /etc/openvpn/server/

# C помощью любого текстового редактора открываем файл server.conf для редактирования:
sudo vim /etc/openvpn/server/server.conf
# В этом файле нужно внести следующие изменения:
#   заменить dh dh2048.pem на dh none
#   раскомментировать строку push "redirect-gateway def1 bypass-dhcp"
#   раскомментировать две строки с DNS серверами:
#   push "dhcp-option DNS 208.67.222.222"
#   push "dhcp-option DNS 208.67.220.220"
# По умолчанию здесь указаны адреса публичных DNS серверов от OpenDNS.
# Рекомендую сразу их заменить на DNS сервера от CloudFlare (1.1.1.1, 1.0.0.1) или Google (8.8.8.8 и 8.8.4.4)
#   заменить tls-auth ta.key 0 на tls-crypt ta.key
#   заменить cipher AES-256-CBC на cipher AES-256-GCM и после этой строки добавить еще одну новую – auth SHA256
#   добавить в конце файла две строки:
#     user nobody
#     group nogroup

# Чтобы включить переадресацию пакетов,
# раскомментируем (вручную или с помощью утилиты sed) строку net.ipv4.ip_forward=1 в файле /etc/sysctl.conf и применим изменения:
sudo sed -i '/net.ipv4.ip_forward=1/s/^#//g' /etc/sysctl.conf
sudo sysctl -p

# Теперь нужно настроить форвардинг и маскарадинг в iptables,
# но для этого сначала посмотрим имя публичного сетевого интерфейса на сервере:
ip route list default

# Пример результата выполнения команды показан ниже, в нем имя нужного нам интерфейса отображается сразу после "dev" :
# --- default via 123.45.67.8 dev ens3 proto static onlink ---
# Здесь интерфейс называется ens3, в вашем случае он может быть другой.
# Разрешаем переадресацию и включаем маскарадинг в iptables.
# При необходимости имя интерфейса (ens3) в трех местах замените на нужное:

sudo apt install iptables-persistent -y
sudo iptables -A INPUT -i tun+ -j ACCEPT
sudo iptables -A FORWARD -i tun+ -j ACCEPT
sudo iptables -A FORWARD -i ens3 -o tun+ -j ACCEPT
sudo iptables -A FORWARD -i tun+ -o ens3 -j ACCEPT
sudo iptables -t nat -A POSTROUTING -s 10.8.0.0/8 -o ens3 -j MASQUERADE
sudo netfilter-persistent save

# Добавляем сервис OpenVPN в автозагрузку и запускаем его:
sudo systemctl enable openvpn-server@server.service
sudo systemctl start openvpn-server@server.service

# останавливать
sudo systemctl stop openvpn-server@server.service

# Проверить, запущен ли VPN можно командой:
sudo systemctl status openvpn-server@server.service

# Нам осталось создать файл конфигурации .ovpn, который клиент будет использовать для подключения к VPN.
# Файл .ovpn должен содержать базовые параметры, сертификаты и ключи.
# В скрипте вместо X.X.X.X впишите IP адрес вашего сервера, поместите файл в любую директорию и установите исполняемые права:
chmod +x create_client_config.sh
./create_client_config.sh client1

# Запускать клиент на своей тачке
openvpn --config client1.ovpn