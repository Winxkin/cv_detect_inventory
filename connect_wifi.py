import os
import time

#print('=====SCAN WIFI NETWORK======')
#os.system('nmcli dev wifi')

print('=====SCAN WIFI SSID======')
os.system('sudo iw dev wlan0 scan | grep SSID')


print('======CONNECT WIFI=======')
SSID = input("SSID:")
PASSWORD = input("PASSWORD:")
print('Connect WIFI to  ' + SSID + '...')

os.system('sudo nmcli dev wifi connect ' + '"' + SSID + '"' + ' password ' + '"' + PASSWORD +'"')