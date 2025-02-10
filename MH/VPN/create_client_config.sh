#!/bin/bash

# Формат использования: create_client_config.sh <clientname>
# Перед использованием в SERVER_IP вместо X.X.X.X необходимо указать IP адрес вашего сервера

SERVER_IP=X.X.X.X
DIR=~/openvpn-clients

cat <(echo -e \
   "# Client OpenVPN config file"\
   "\nclient" \
   "\ndev tun" \
   "\nproto udp" \
   "\nremote $SERVER_IP 1194" \
   "\nresolv-retry infinite" \
   "\nnobind" \
   "\nuser nobody" \
   "\ngroup nogroup" \
   "\npersist-key" \
   "\npersist-tun" \
   "\nremote-cert-tls server" \
   "\nkey-direction 1" \
   "\ncipher AES-256-GCM" \
   "\nauth SHA256" \
   "\nverb 3" \
   ) \
   <(echo -e "\n<ca>") \
   ${DIR}/ca.crt \
   <(echo -e "</ca>\n\n<cert>") \
   ${DIR}/${1}.crt \
   <(echo -e "</cert>\n\n<key>") \
   ${DIR}/${1}.key \
   <(echo -e "</key>\n\n<tls-crypt>") \
   ${DIR}/ta.key \
   <(echo -e "</tls-crypt>") \
   > ${DIR}/${1}.ovpn