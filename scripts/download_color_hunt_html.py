import os
import time

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

service = Service("//home/shc/chromedriver-linux64/chromedriver")

driver = webdriver.Chrome(service=service)

url = "https://colorhunt.co"
driver.get(url)

time.sleep(2)

last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    driver.find_element(By.TAG_NAME,'body').send_keys(Keys.END)

    time.sleep(2)
    new_height = driver.execute_script("return document.body.scrollHeight")

    if new_height == last_height:
        break
    else:
        last_height = new_height

html_content = driver.page_source

file_path = os.path.join("data", "html", "color_hunt_paletes", url)
with open(file_path, "w+", encoding="utf-8") as file:
    file.write(html_content)

driver.quit()
