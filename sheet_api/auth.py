############################################ Imports ###############################################################
import gspread
import time
from datetime import datetime
import os
import json

################################### Global and predefined parameters ###############################################
fread = open(os.path.join(os.getcwd(),'sheet_api','employee_info.json'), 'r')
employee_data = json.load(fread)
gc = gspread.service_account(filename=os.path.join(os.getcwd(),'sheet_api','credentials.json'))
DEFAULT_KEY = '1MvtfQlhRO86WxCX2ia4SHA_jEgAsXpOO22atTEsriRY'
morning_time = datetime(2020, 5, 13, 9, 0, 0)

############################################ Function Definitions ##################################################
def add_info(info, key=DEFAULT_KEY):
    
    # init variables 
    sh = gc.open_by_key(key)
    worksheet = sh.sheet1
    mask_status = "No Mask"
    names = info["employee"]
    emotions, mask = info["emotions"], info["mask"]
    date = datetime.now().strftime("%Y-%m-%d")
    ti = datetime.now() # time
    
    # Iterating through all the identified faces and updating the sheets 
    for i, name in enumerate(names): 
        if name is "UI":
            continue
        if(mask[i] == 1):
            mask_status = "Mask"

        if name not in employee_data.keys():
            id = "Unknown"
            department = "Unknown"
            position = "Unknown"
        else:
            id = employee_data[name]["ID"]
            department = employee_data[name]["Department"]
            position = employee_data[name]["Position"]

        remark = "Late" if ti.time() > morning_time.time() else "On Time"
        user = [id, name, department, position, date,
                ti.strftime("%H:%M:%S"), remark, emotions[i]["dominant_emotion"]]

        worksheet.append_row(user)
        print("SUCCESSS")

###################################################################################################################