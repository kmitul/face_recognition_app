import gspread
import time
from datetime import datetime
import os
import json

fread = open(os.path.join(os.getcwd(),'sheet_api','employee_info.json'), 'r')
employee_data = json.load(fread)
gc = gspread.service_account(filename=os.path.join(os.getcwd(),'sheet_api','credentials.json'))
DEFAULT_KEY = '1MvtfQlhRO86WxCX2ia4SHA_jEgAsXpOO22atTEsriRY'
morning_time = datetime(2020, 5, 13, 9, 0, 0)


def add_info(info, key=DEFAULT_KEY):
    mask_status = "No Mask"
    sh = gc.open_by_key(key)
    worksheet = sh.sheet1

    names = info["employee"]
    emotions, mask = info["emotions"], info["mask"]
    date = datetime.now().strftime("%Y-%m-%d")
    ti = datetime.now() # time
    for i, name in enumerate(names): 
        if name is "UI":
            continue
        if(mask[i] == 1):
            mask_status = "Mask"

        id = employee_data[name]["ID"]
        department = employee_data[name]["Department"]
        position = employee_data[name]["Position"]
        remark = "Late" if ti.time() > morning_time.time() else "On Time"
        user = [id, name, department, position, date,
                ti.strftime("%H:%M:%S"), remark, emotions[i]["dominant_emotion"]]

        worksheet.append_row(user)
        print("SUCCESSS")

# ['Ronaldo', '2021-07-07', '14:58:34', 26.86828502598559, 0, 'neutral']
# add_info(['Ronaldo', '2021-07-07', '14:58:34', 26.86828502598559, 0, 'neutral'])