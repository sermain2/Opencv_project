import RPi.GPIO as GPIO
from selenium import webdriver

GPIO.setmode(GPIO.BCM)
GPIO.setup(21, GPIO.IN, pull_up_down = GPIO.PUD_DOWN)
sw_pin = 21


driver = webdriver.Chrome('/lib/chromium-browser/chromedriver')
driver.get("http://220.69.207.85:8080/test.html")

try:
    while True:
        if GPIO.input(sw_pin) == GPIO.HIGH:
            driver.close()
            break  
except KeyboardInterrupt :
    exit()