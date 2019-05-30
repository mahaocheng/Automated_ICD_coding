import json
import csv
import re
import time
import jieba


# 抽取主诉
def Chief_Complaint(data):
    if 'admissions_records' in data:
        try:
            chief_complaint = data['admissions_records'][0]['CHIEF_COMPLAINT']
        except:
            chief_complaint = ""
    else:
        chief_complaint = ""
    # 有些主诉不规范，把现病史也放进来了，需要剔除，主诉一般在第一句话，不超过30个字符
    #     if len(chief_complaint) > 30:
    #         chief_complaint = chief_complaint[:30]

    return ' '.join(list(jieba.cut(chief_complaint)))


# 抽取检查报告
def examination(data):
    def exam_temp(item):
        text = []
        if item in data:
            for i in range(len(data[item])):
                try:
                    item_name = data[item][i]['EXAMINATION_ITEM']
                except:
                    item_name = ''
                try:
                    opinion = data[item][i]['DIAGNOSIS_OPINION']
                except:
                    opinion = ''
                try:
                    findings = data[item][i]['EXAMINATION_FINDINGS']
                except:
                    findings = ''
                try:
                    diag = data[item][i]['CLINICAL_DIAGNOSIS']
                except:
                    diag = ''
                try:
                    area = data[item][i]['EXAMINATION_AREA']
                except:
                    area = ''
                combine_text = item_name + ' ' + area + ' ' + findings + ' ' + opinion + ' ' + diag
                text.append(combine_text)
        return ' '.join(text)

    # CT，MR，ECT
    ct_report = exam_temp("ct_reports")
    mr_reports = exam_temp("mr_reports")
    ect_report = exam_temp('ect_reports')

    # 心电图
    eleccar = []
    if 'electrocardiogram_reports' in data:
        for i in range(len(data['electrocardiogram_reports'])):
            try:
                eleccar.append(data['electrocardiogram_reports'][i]['ECG_DIAGNOSTIC_OPINION'])
            except:
                pass
    elec_report = ' '.join(eleccar)

    # X线
    xray = []
    if 'xray_image_reports' in data:
        for i in range(len(data['xray_image_reports'])):
            try:
                xray.append(data['xray_image_reports'][i]["EXAMINATION_FINDINGS"])
            except:
                pass
            try:
                xray.append(data['xray_image_reports'][i]["SUGGESTION"])
            except:
                pass
            try:
                xray.append(data['xray_image_reports'][i]["CLINICAL_DIAGNOSIS"])
            except:
                pass
    xray_report = ' '.join(xray)

    # 病理检查
    pathology = []
    if 'pathology_reports' in data:
        for i in range(len(data['pathology_reports'])):
            try:
                pathology.append(data['pathology_reports'][i]['SPECIMENS_NAME'])
            except:
                pass
            try:
                pathology.append(data['pathology_reports'][i]['UNDER_MICROSCOPE'])
            except:
                pass
            try:
                pathology.append(data['pathology_reports'][i]['VISIBLE_PATHOLOGY'])
            except:
                pass
            try:
                pathology.append(data['pathology_reports'][i]['CLINICAL_DIAGNOSIS'])
            except:
                pass
            try:
                pathology.append(data['pathology_reports'][i]['PATHOLOGY_DIAGNOSIS'])
            except:
                pass
    pathology_report = ' '.join(pathology)
    # CT

    # 超声
    ultrasonic = []
    if 'ultrasonic_diagnosis_reports' in data:
        for i in range(len(data['ultrasonic_diagnosis_reports'])):
            try:
                ultrasonic.append(data['ultrasonic_diagnosis_reports'][i]['BODY_PARTS'])
            except:
                pass
            try:
                ultrasonic.append(data['ultrasonic_diagnosis_reports'][i]['"DIAGNOSIS_CONTENTS"'])
            except:
                pass
            try:
                ultrasonic.append(data['ultrasonic_diagnosis_reports'][i]['DIAGNOSIS_RESULT'])
            except:
                pass
    ult_report = ' '.join(ultrasonic)

    total_reports = ct_report + ' ' + mr_reports + ' ' + ect_report + ' ' + elec_report + ' ' + xray_report + ' ' + pathology_report + ' ' + ult_report  # 将所有检查结果合在一起
    total_reports = ' '.join(list(jieba.cut(total_reports)))
    return total_reports


# 抽取现病史
def present_illness_history(data):
    if "admissions_records" in data:
        try:
            text = data["admissions_records"][0]["PRSENT_ILLNESS_HISTORY"]
        except:
            text = ""
    else:
        text = ""
    text = ' '.join(list(jieba.cut(text)))
    return text


# 抽取首次病程记录
def first_course_records(data):
    if "first_course_records" in data:
        try:
            basis = data["first_course_records"][0]["DIAGNOSIS_BASIS"]
        except:
            basis = ''
        try:
            diag = data["first_course_records"][0]["PRELIMINARY_DIAGNOSIS"]
        except:
            diag = ''
        try:
            feature = data["first_course_records"][0]["MEDICAL_FEATURE"]
        except:
            feature = ''
        try:
            treat_plan = data["first_course_records"][0]["TREATMENT_PLAN"]
        except:
            treat_plan = ""
    else:
        basis = ""
        diag = ""
        feature = ''
        treat_plan = ""

    total_text = basis + ' ' + feature + ' ' + ' ' + diag + ' ' + treat_plan
    total_text = ' '.join(list(jieba.cut(total_text)))
    return total_text

# 抽取查房记录
def course_record(data):
    all_records = []
    if "course_records" in data:
        records = data["course_records"]
        for i in range(len(records)):
            try:
                all_records.append(records[i]["WARD_INSPECTION_RECORD"])
            except:
                pass
    all_records = ' '.join(all_records)
    return ' '.join(list(jieba.cut(all_records)))

# 抽取出院记录
def discharge_record(data):
    if "discharge_records" in data:
        discharge_data = data["discharge_records"][0]
        if "HOSPITAL_DISCHARGE_DIAGNOSE" in discharge_data:
            discharge_diag = discharge_data["HOSPITAL_DISCHARGE_DIAGNOSE"]
        else:
            discharge_diag = ''
        if "HOSPITAL_ADMISSION_DIAGNOSE" in discharge_data:
            admission_diag = discharge_data["HOSPITAL_ADMISSION_DIAGNOSE"]
        else:
            admission_diag = ''
        if "HOSPITAL_DISCHARGE_ORDER" in discharge_data:
            discharge_order = discharge_data["HOSPITAL_DISCHARGE_ORDER"]
        else:
            discharge_order = ''
        if "TREATMENT_COURSE" in discharge_data:
            treat_course = discharge_data["TREATMENT_COURSE"]
        else:
            treat_course = ''
    else:
        discharge_diag = ''
        admission_diag = ''
        discharge_order = ''
        treat_course = ''
    total_text = discharge_diag + ' ' + admission_diag + ' ' + treat_course + ' ' + discharge_order
    return ' '.join(list(jieba.cut(total_text)))

# 抽取标签：病案首页主诊断的ICD编码
def main_diagnose(data):
    if 'medical_record_home_page' in data:
        if 'dis_main_diag' in data['medical_record_home_page'][0]:
            try:
                main_diag_code = data['medical_record_home_page'][0]['dis_main_diag'][0]['DIS_DIAG_CODE']
            except:
                main_diag_code = ""
        else:
            main_diag_code = ""
    else:
        main_diag_code = ""
    return main_diag_code


t1 = time.time()
total_text = []
with open('/home/pkudata/medical_home_page_source_data/medical_home_page_8.2M.data', 'r', encoding='utf-8') as f:
    n = 0
    for line in f:
        data = json.loads(line)
        chief_complaint = Chief_Complaint(data)
        exam_report = examination(data)
        present_illness = present_illness_history(data)
        first_records = first_course_records(data)
        cour_record = course_record(data)
        dis_record = discharge_record(data)
        main_diag_code = main_diagnose(data)

        text = [chief_complaint, present_illness, exam_report, first_records, cour_record, dis_record, main_diag_code]
        total_text.append(text)
        n += 1
        if n%100 == 0:
            print('{} data have been extracted'.format(str(n)))

with open('/home/yanrui/ICD/data/ICD_raw_data.csv', 'w+', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    for row in total_text:
        writer.writerow(row)
t2 = time.time()
print(t2 - t1)