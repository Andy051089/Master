cols_to_keep = ['SUBJECT_ID', 'GENDER', 'DOB', 'DOD']
PAT_LIST = PATIENTS[cols_to_keep]

q = '''
SELECT DISTINCT A.*, B.HADM_ID, B.ICD9_CODE
FROM PAT_LIST A
INNER JOIN DIAGNOSE B 
ON A.SUBJECT_ID = B.SUBJECT_ID
WHERE B.ICD9_CODE IN ('39891', '40201', '40211', '40291', '40401', '40403', '40411', '40413', '40491', '40493')
AND B.ICD9_CODE '428%'
AND B.SEQ_NUM = 1
'''
HF_PAT = sqldf(q, globals())

q = '''
SELECT DISTINCT A.*, B.ADMITTIME, B.DISCHTIME, B.INSURANCE
FROM HF_PAT A
INNER JOIN ADMISSION B
ON A.HADM_ID = B.HADM_ID
'''
HF_PAT = sqldf(q, globals())

USE_LAB = LAB.dropna(subset = 'HADM_ID', axis = 0)

HB_USE = USE_LAB[(USE_LAB['ITEMID'] == 51222) | (USE_LAB['ITEMID'] == 50811) | (USE_LAB['ITEMID'] == 50855)]
HB_USE.loc[:, 'VALUENUM'] = HB_USE['VALUENUM'].replace(0, np.nan)
HB_USE = HB_USE.dropna(subset = 'VALUENUM', axis = 0)
HB_USE_1 = HB_USE.sort_values('CHARTTIME').groupby('HADM_ID').first().reset_index()
HB_USE_2 = HB_USE.sort_values('CHARTTIME').groupby('HADM_ID').last().reset_index()
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'HB_1'
FROM HF_PAT A 
LEFT JOIN HB_USE_1 B
ON A.HADM_ID = B.HADM_ID
'''
HB_1 = sqldf(q, globals())
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'HB_2'
FROM HB_1 A 
LEFT JOIN HB_USE_2 B
ON A.HADM_ID = B.HADM_ID
'''
HB_2 = sqldf(q, globals())

BNP_USE = USE_LAB[USE_LAB['ITEMID'] == 50963]
BNP_USE.loc[:, 'VALUENUM'] = BNP_USE['VALUENUM'].replace(0, np.nan)
BNP_USE = BNP_USE.dropna(subset = 'VALUENUM', axis = 0)
BNP_USE_1 = BNP_USE.sort_values('CHARTTIME').groupby('HADM_ID').first().reset_index()
BNP_USE_2 = BNP_USE.sort_values('CHARTTIME').groupby('HADM_ID').last().reset_index()
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'BNP_1'
FROM HB_2 A 
LEFT JOIN BNP_USE_1 B
ON A.HADM_ID = B.HADM_ID
'''
BNP_1 = sqldf(q, globals())
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'BNP_2'
FROM BNP_1 A 
LEFT JOIN BNP_USE_2 B
ON A.HADM_ID = B.HADM_ID
'''
BNP_2 = sqldf(q, globals())

CREA_USE = USE_LAB[USE_LAB['ITEMID'] == 50912]
CREA_USE.loc[:, 'VALUENUM'] = CREA_USE['VALUENUM'].replace(0, np.nan)
CREA_USE = CREA_USE.dropna(subset = 'VALUENUM', axis = 0)
CREA_USE_1 = CREA_USE.sort_values('CHARTTIME').groupby('HADM_ID').first().reset_index()
CREA_USE_2 = CREA_USE.sort_values('CHARTTIME').groupby('HADM_ID').last().reset_index()
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'CREA_1'
FROM BNP_2 A 
LEFT JOIN CREA_USE_1 B
ON A.HADM_ID = B.HADM_ID
'''
CREA_1 = sqldf(q, globals())
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'CREA_2'
FROM CREA_1 A 
LEFT JOIN CREA_USE_2 B
ON A.HADM_ID = B.HADM_ID
'''
CREA_2 = sqldf(q, globals())

BUN_USE = USE_LAB[USE_LAB['ITEMID'] == 51006]
BUN_USE.loc[:, 'VALUENUM'] = BUN_USE['VALUENUM'].replace(0, np.nan)
BUN_USE = BUN_USE.dropna(subset = 'VALUENUM', axis = 0)
BUN_USE_1 = BUN_USE.sort_values('CHARTTIME').groupby('HADM_ID').first().reset_index()
BUN_USE_2 = BUN_USE.sort_values('CHARTTIME').groupby('HADM_ID').last().reset_index()
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'BUN_1'
FROM CREA_2 A 
LEFT JOIN BUN_USE_1 B
ON A.HADM_ID = B.HADM_ID
'''
BUN_1 = sqldf(q, globals())
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'BUN_2'
FROM BUN_1 A 
LEFT JOIN BUN_USE_2 B
ON A.HADM_ID = B.HADM_ID
'''
BUN_2 = sqldf(q, globals())

ALB_USE = USE_LAB[USE_LAB['ITEMID'] == 50862]
ALB_USE.loc[:, 'VALUENUM'] = ALB_USE['VALUENUM'].replace(0, np.nan)
ALB_USE = ALB_USE.dropna(subset = 'VALUENUM', axis = 0)
ALB_USE_1 = ALB_USE.sort_values('CHARTTIME').groupby('HADM_ID').first().reset_index()
ALB_USE_2 = ALB_USE.sort_values('CHARTTIME').groupby('HADM_ID').last().reset_index()
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'ALB_1'
FROM BUN_2 A 
LEFT JOIN ALB_USE_1 B
ON A.HADM_ID = B.HADM_ID
'''
ALB_1 = sqldf(q, globals())
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'ALB_2'
FROM ALB_1 A 
LEFT JOIN ALB_USE_2 B
ON A.HADM_ID = B.HADM_ID
'''
ALB_2 = sqldf(q, globals())

NA_USE = USE_LAB[(USE_LAB['ITEMID'] == 50824) | (USE_LAB['ITEMID'] == 50983)]
NA_USE.loc[:, 'VALUENUM'] = NA_USE['VALUENUM'].replace(0, np.nan)
NA_USE = NA_USE.dropna(subset = 'VALUENUM', axis = 0)
NA_USE_1 = NA_USE.sort_values('CHARTTIME').groupby('HADM_ID').first().reset_index()
NA_USE_2 = NA_USE.sort_values('CHARTTIME').groupby('HADM_ID').last().reset_index()
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'NA_1'
FROM ALB_2 A 
LEFT JOIN NA_USE_1 B
ON A.HADM_ID = B.HADM_ID
'''
NA_1 = sqldf(q, globals())
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'NA_2'
FROM NA_1 A 
LEFT JOIN NA_USE_2 B
ON A.HADM_ID = B.HADM_ID
'''
NA_2 = sqldf(q, globals())

HEIGHT.loc[:, 'VALUENUM'] = HEIGHT['VALUENUM'].replace(0, np.nan)
HEIGHT = HEIGHT.dropna(subset='VALUENUM', axis = 0)
HEIGHT = HEIGHT.sort_values('VALUEUOM').drop_duplicates(subset='HADM_ID', keep='last')
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'HEIGHT', B.VALUEUOM AS 'INCH/CM' 
FROM NA_2 A 
left JOIN HEIGHT B
ON A.HADM_ID = B.HADM_ID
'''
HEI = sqldf(q, globals())

WEIGHT.loc[:, 'VALUENUM'] = WEIGHT['VALUENUM'].replace(0, np.nan)
WEIGHT = WEIGHT.dropna(subset='VALUENUM', axis = 0)
WEIGHT_USE = WEIGHT.sort_values('CHARTTIME').groupby('HADM_ID').last().reset_index()
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'WEIGHT_2', B.ITEMID, B.VALUEUOM AS 'KG/GMS' 
FROM HEI A 
LEFT JOIN WEIGHT_USE B
ON A.HADM_ID = B.HADM_ID
'''
WEI = sqldf(q, globals())
WEIGHT_USE_1 = WEIGHT.sort_values('CHARTTIME').groupby('HADM_ID').first().reset_index()
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'WEIGHT_1', B.ITEMID AS 'ITEMID_1', B.VALUEUOM AS 'KG/GMS_1' 
FROM WEI A 
LEFT JOIN WEIGHT_USE_1 B
ON A.HADM_ID = B.HADM_ID
'''
WEI_1 = sqldf(q, globals())

O_USE = O[O['VALUENUM'] >= 50]
O_USE = O_USE.dropna(subset = 'VALUENUM', axis = 0)
O_USE_1 = O_USE.sort_values('CHARTTIME').groupby('HADM_ID').first().reset_index()
O_USE_2 = O_USE.sort_values('CHARTTIME').groupby('HADM_ID').last().reset_index()
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'O_1' 
FROM WEI_1 A 
LEFT JOIN O_USE_1 B
ON A.HADM_ID = B.HADM_ID
'''
O_1 = sqldf(q, globals())
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'O_2' 
FROM O_1 A 
LEFT JOIN O_USE_2 B
ON A.HADM_ID = B.HADM_ID
'''
O_2 = sqldf(q, globals())

HR.loc[:, 'VALUENUM'] = HR['VALUENUM'].replace(0, np.nan)
HR_USE = HR.dropna(subset='VALUENUM', axis = 0)
HR_USE_1 = HR_USE.sort_values('CHARTTIME').groupby('HADM_ID').first().reset_index()
HR_USE_2 = HR_USE.sort_values('CHARTTIME').groupby('HADM_ID').last().reset_index()
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'HR_1' 
FROM O_2 A 
LEFT JOIN HR_USE_1 B
ON A.HADM_ID = B.HADM_ID
'''
HR_1 = sqldf(q, globals())
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'HR_2' 
FROM HR_1 A 
LEFT JOIN HR_USE_2 B
ON A.HADM_ID = B.HADM_ID
'''
HR_2 = sqldf(q, globals())

RR.loc[:, 'VALUENUM'] = RR['VALUENUM'].replace(0, np.nan)
RR_USE = RR.dropna(subset='VALUENUM', axis = 0)
RR_USE_1 = RR_USE.sort_values('CHARTTIME').groupby('HADM_ID').first().reset_index()
RR_USE_2 = RR_USE.sort_values('CHARTTIME').groupby('HADM_ID').last().reset_index()
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'RR_1' 
FROM HR_2 A 
LEFT JOIN RR_USE_1 B
ON A.HADM_ID = B.HADM_ID
'''
RR_1 = sqldf(q, globals())
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'RR_2' 
FROM RR_1 A 
LEFT JOIN RR_USE_2 B
ON A.HADM_ID = B.HADM_ID
'''
RR_2 = sqldf(q, globals())

SBP.loc[:, 'VALUENUM'] = SBP['VALUENUM'].replace(0, np.nan)
SBP_USE = SBP.dropna(subset='VALUENUM', axis = 0)
SBP_USE_1 = SBP_USE.sort_values('CHARTTIME').groupby('HADM_ID').first().reset_index()
SBP_USE_2 = SBP_USE.sort_values('CHARTTIME').groupby('HADM_ID').last().reset_index()
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'SBP_1' 
FROM RR_2 A 
LEFT JOIN SBP_USE_1 B
ON A.HADM_ID = B.HADM_ID
'''
SBP_1 = sqldf(q, globals())
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'SBP_2' 
FROM SBP_1 A 
LEFT JOIN SBP_USE_2 B
ON A.HADM_ID = B.HADM_ID
'''
SBP_2 = sqldf(q, globals())

CRP_USE = USE_LAB[USE_LAB['ITEMID'] == 50889]
CRP_USE = CRP_USE.dropna(subset='VALUE', axis = 0)
CRP_USE_1 = CRP_USE.sort_values('CHARTTIME').groupby('HADM_ID').first().reset_index()
CRP_USE_2 = CRP_USE.sort_values('CHARTTIME').groupby('HADM_ID').last().reset_index()
q = '''
SELECT DISTINCT A.*, B.VALUE AS 'CRP_W1', B.VALUENUM AS 'CRP_1'
FROM SBP_2 A 
LEFT JOIN CRP_USE_1 B
ON A.HADM_ID = B.HADM_ID
'''
CRP_1 = sqldf(q, globals())
q = '''
SELECT DISTINCT A.*, B.VALUE AS 'CRP_W2', B.VALUENUM AS 'CRP_2'
FROM CRP_1 A 
LEFT JOIN CRP_USE_2 B
ON A.HADM_ID = B.HADM_ID
'''
CRP_2 = sqldf(q, globals())

CRP_2['CRP_1'] = CRP_2.apply(lambda row: 301 if pd.isna(row['CRP_1']) and row['CRP_W1'] == "GREATER THAN 300" else row['CRP_1'], axis=1)
CRP_2['CRP_1'] = CRP_2.apply(lambda row: 31 if pd.isna(row['CRP_1']) and row['CRP_W1'] == "GREATER THAN 30" else row['CRP_1'], axis=1)
CRP_2['CRP_1'] = CRP_2.apply(lambda row: 31 if pd.isna(row['CRP_1']) and row['CRP_W1'] == ">30" else row['CRP_1'], axis=1)
CRP_2['CRP_1'] = CRP_2.apply(lambda row: 301 if pd.isna(row['CRP_1']) and row['CRP_W1'] == "> 300" else row['CRP_1'], axis=1)

CRP_2['CRP_2'] = CRP_2.apply(lambda row: 301 if pd.isna(row['CRP_2']) and row['CRP_W2'] == "GREATER THAN 300" else row['CRP_2'], axis=1)
CRP_2['CRP_2'] = CRP_2.apply(lambda row: 31 if pd.isna(row['CRP_2']) and row['CRP_W2'] == "GREATER THAN 30" else row['CRP_2'], axis=1)
CRP_2['CRP_2'] = CRP_2.apply(lambda row: 31 if pd.isna(row['CRP_2']) and row['CRP_W2'] == ">30" else row['CRP_2'], axis=1)
CRP_2['CRP_2'] = CRP_2.apply(lambda row: 301 if pd.isna(row['CRP_2']) and row['CRP_W2'] == "> 300" else row['CRP_2'], axis=1)

WBC_USE = USE_LAB[(USE_LAB['ITEMID'] == 51300) | (USE_LAB['ITEMID'] == 51301)]
WBC_USE.loc[:, 'VALUENUM'] = WBC_USE['VALUENUM'].replace(0, np.nan)
WBC_USE = WBC_USE.dropna(subset = 'VALUENUM', axis = 0)
WBC_USE_1 = WBC_USE.sort_values('CHARTTIME').groupby('HADM_ID').first().reset_index()
WBC_USE_2 = WBC_USE.sort_values('CHARTTIME').groupby('HADM_ID').last().reset_index()
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'WBC_1'
FROM CRP_2 A 
LEFT JOIN WBC_USE_1 B
ON A.HADM_ID = B.HADM_ID
'''
WBC_1 = sqldf(q, globals())
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'WBC_2'
FROM WBC_1 A 
LEFT JOIN WBC_USE_2 B
ON A.HADM_ID = B.HADM_ID
'''
WBC_2 = sqldf(q, globals())

NEU_USE = USE_LAB[USE_LAB['ITEMID'] == 51256]
NEU_USE.loc[:, 'VALUENUM'] = NEU_USE['VALUENUM'].replace(0, np.nan)
NEU_USE = NEU_USE.dropna(subset = 'VALUENUM', axis = 0)
NEU_USE_1 = NEU_USE.sort_values('CHARTTIME').groupby('HADM_ID').first().reset_index()
NEU_USE_2 = NEU_USE.sort_values('CHARTTIME').groupby('HADM_ID').last().reset_index()
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'NEU_1'
FROM WBC_2 A 
LEFT JOIN NEU_USE_1 B
ON A.HADM_ID = B.HADM_ID
'''
NEU_1 = sqldf(q, globals())
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'NEU_2'
FROM NEU_1 A 
LEFT JOIN NEU_USE_2 B
ON A.HADM_ID = B.HADM_ID
'''
NEU_2 = sqldf(q, globals())

ESR_USE = USE_LAB[USE_LAB['ITEMID'] == 51288]
ESR_USE.loc[:, 'VALUENUM'] = ESR_USE['VALUENUM'].replace(0, np.nan)
ESR_USE = ESR_USE.dropna(subset = 'VALUENUM', axis = 0)
ESR_USE_1 = ESR_USE.sort_values('CHARTTIME').groupby('HADM_ID').first().reset_index()
ESR_USE_2 = ESR_USE.sort_values('CHARTTIME').groupby('HADM_ID').last().reset_index()
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'ESR_1'
FROM NEU_2 A 
LEFT JOIN ESR_USE_1 B
ON A.HADM_ID = B.HADM_ID
'''
ESR_1 = sqldf(q, globals())
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'ESR_2'
FROM ESR_1 A 
LEFT JOIN ESR_USE_2 B
ON A.HADM_ID = B.HADM_ID
'''
ESR_2 = sqldf(q, globals())

PLT_USE = USE_LAB[USE_LAB['ITEMID'] == 51265]
PLT_USE.loc[:, 'VALUENUM'] = PLT_USE['VALUENUM'].replace(0, np.nan)
PLT_USE = PLT_USE.dropna(subset = 'VALUENUM', axis = 0)
PLT_USE_1 = PLT_USE.sort_values('CHARTTIME').groupby('HADM_ID').first().reset_index()
PLT_USE_2 = PLT_USE.sort_values('CHARTTIME').groupby('HADM_ID').last().reset_index()
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'PLT_1'
FROM ESR_2 A 
LEFT JOIN PLT_USE_1 B
ON A.HADM_ID = B.HADM_ID
'''
PLT_1 = sqldf(q, globals())
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'PLT_2'
FROM PLT_1 A 
LEFT JOIN PLT_USE_2 B
ON A.HADM_ID = B.HADM_ID
'''
PLT_2 = sqldf(q, globals())

LAC_USE = USE_LAB[USE_LAB['ITEMID'] == 50813]
LAC_USE.loc[:, 'VALUENUM'] = LAC_USE['VALUENUM'].replace(0, np.nan)
LAC_USE = LAC_USE.dropna(subset = 'VALUENUM', axis = 0)
LAC_USE_1 = LAC_USE.sort_values('CHARTTIME').groupby('HADM_ID').first().reset_index()
LAC_USE_2 = LAC_USE.sort_values('CHARTTIME').groupby('HADM_ID').last().reset_index()
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'LAC_1'
FROM PLT_2 A 
LEFT JOIN LAC_USE_1 B
ON A.HADM_ID = B.HADM_ID
'''
LAC_1 = sqldf(q, globals())
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'LAC_2'
FROM LAC_1 A 
LEFT JOIN LAC_USE_2 B
ON A.HADM_ID = B.HADM_ID
'''
LAC_2 = sqldf(q, globals())

MAG_USE = USE_LAB[USE_LAB['ITEMID'] == 50960]
MAG_USE.loc[:, 'VALUENUM'] = MAG_USE['VALUENUM'].replace(0, np.nan)
MAG_USE = MAG_USE.dropna(subset = 'VALUENUM', axis = 0)
MAG_USE_1 = MAG_USE.sort_values('CHARTTIME').groupby('HADM_ID').first().reset_index()
MAG_USE_2 = MAG_USE.sort_values('CHARTTIME').groupby('HADM_ID').last().reset_index()
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'MAG_1'
FROM LAC_2 A 
LEFT JOIN MAG_USE_1 B
ON A.HADM_ID = B.HADM_ID
'''
MAG_1 = sqldf(q, globals())
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'MAG_2'
FROM MAG_1 A 
LEFT JOIN MAG_USE_2 B
ON A.HADM_ID = B.HADM_ID
'''
MAG_2 = sqldf(q, globals())

K_USE = USE_LAB[(USE_LAB['ITEMID'] == 50822) | (USE_LAB['ITEMID'] == 50833) | (USE_LAB['ITEMID'] == 50971)]
K_USE.loc[:, 'VALUENUM'] = K_USE['VALUENUM'].replace(0, np.nan)
K_USE = K_USE.dropna(subset = 'VALUENUM', axis = 0)
K_USE_1 = K_USE.sort_values('CHARTTIME').groupby('HADM_ID').first().reset_index()
K_USE_2 = K_USE.sort_values('CHARTTIME').groupby('HADM_ID').last().reset_index()
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'K_1'
FROM MAG_2 A 
LEFT JOIN K_USE_1 B
ON A.HADM_ID = B.HADM_ID
'''
K_1 = sqldf(q, globals())
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'K_2'
FROM K_1 A 
LEFT JOIN K_USE_2 B
ON A.HADM_ID = B.HADM_ID
'''
K_2 = sqldf(q, globals())

MONO_USE = USE_LAB[(USE_LAB['ITEMID'] == 51254) | (USE_LAB['ITEMID'] == 51253)]
MONO_USE.loc[:, 'VALUENUM'] = MONO_USE['VALUENUM'].replace(0, np.nan)
MONO_USE = MONO_USE.dropna(subset = 'VALUENUM', axis = 0)
MONO_USE_1 = MONO_USE.sort_values('CHARTTIME').groupby('HADM_ID').first().reset_index()
MONO_USE_2 = MONO_USE.sort_values('CHARTTIME').groupby('HADM_ID').last().reset_index()
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'MONO_1'
FROM K_2 A 
LEFT JOIN MONO_USE_1 B
ON A.HADM_ID = B.HADM_ID
'''
MONO_1 = sqldf(q, globals())
q = '''
SELECT DISTINCT A.*, B.VALUENUM AS 'MONO_2'
FROM MONO_1 A 
LEFT JOIN MONO_USE_2 B
ON A.HADM_ID = B.HADM_ID
'''
MONO_2 = sqldf(q, globals())

time_columns = ['ADMITTIME', 'DISCHTIME', 'DOB', 'DOD']
for col in time_columns:
    MONO_2[col] = pd.to_datetime(MONO_2[col])

MONO_2 = MONO_2[~(MONO_2['DOD'] <= MONO_2['DISCHTIME'])]

MONO_2['DEAD_DAYS'] = (MONO_2['DOD'] - MONO_2['DISCHTIME']).dt.days 
MONO_2['90DAYS_DEAD'] = np.where(MONO_2['DEAD_DAYS'] <= 180, 1, 0)

MONO_2['AGE'] = MONO_2.apply(lambda x: 
                     x['ADMITTIME'].year - x['DOB'].year - 
                     ((x['ADMITTIME'].month, x['ADMITTIME'].day) < 
                     (x['DOB'].month, x['DOB'].day)), axis=1)
MONO_2['HOS_DAYS'] = (MONO_2['DISCHTIME'] - MONO_2['ADMITTIME']).dt.days  

MONO_2.loc[MONO_2['AGE'] >= 90, 'AGE'] = 91.4

inch_mask = MONO_2['INCH/CM'] == 'In'
MONO_2.loc[inch_mask, 'HEIGHT'] = MONO_2.loc[inch_mask, 'HEIGHT'] * 2.54

MONO_2 = MONO_2[MONO_2['AGE'] >= 18]

MONO_2.loc[[1370,874,1305], 'HEIGHT'] = np.nan
MONO_2.loc[[933,640,1233,1371], 'WEIGHT_2'] = np.nan
MONO_2.loc[[155], 'WEIGHT_1'] = np.nan

SMOKING_1 = SMOKING[SMOKING['ITEMID'] == 225108]
SMOKING_2 = SMOKING[SMOKING['ITEMID'] == 227687]
smoke_map = {
    'Never us' : 0,
    'Former u' : 1,  
    'Current'  : 1, 
    'Stopped' : 1}
SMOKING_2['VALUE'] = SMOKING_2['VALUE'].map(smoke_map)
FINAL_SMOKE = pd.concat([SMOKING_1, SMOKING_2], axis = 0, ignore_index = True)
FINAL_SMOKE['VALUE'] = FINAL_SMOKE['VALUE'].astype(int)
FINAL_SMOKE = FINAL_SMOKE.sort_values('VALUE').groupby('HADM_ID').last().reset_index()

q = '''
SELECT DISTINCT HADM_ID
FROM DIAGNOSE
WHERE ICD9_CODE IN ('42731')
'''
AFIB = sqldf(q, globals())

q = '''
SELECT DISTINCT HADM_ID
FROM DIAGNOSE
WHERE ICD9_CODE IN ('2720', '2721', '2722', '2724', '2728', '2729')
'''
HYPERLIPIDEMIA = sqldf(q, globals())

q = '''
SELECT DISTINCT HADM_ID
FROM DIAGNOSE
WHERE ICD9_CODE LIKE '410%'
OR ICD9_CODE LIKE '411%'
OR ICD9_CODE LIKE '412%'
'''
MI = sqldf(q, globals())

q = '''
SELECT DISTINCT HADM_ID
FROM DIAGNOSE
WHERE ICD9_CODE IN ('4373', '4431',
                    '4439', '4471',
                    '5571', '5579', 'V434')
OR ICD9_CODE LIKE '440%'
OR ICD9_CODE LIKE '441%'
OR ICD9_CODE LIKE '4432%'
OR ICD9_CODE LIKE '4438%'
'''
PER_VASCULAR_DIS = sqldf(q, globals())

q = '''
SELECT DISTINCT HADM_ID
FROM DIAGNOSE
WHERE ICD9_CODE IN ('36234')
OR ICD9_CODE LIKE '430%'
OR ICD9_CODE LIKE '431%'
OR ICD9_CODE LIKE '432%'
OR ICD9_CODE LIKE '433%'
OR ICD9_CODE LIKE '434%'
OR ICD9_CODE LIKE '435%'
OR ICD9_CODE LIKE '436%'
OR ICD9_CODE LIKE '437%'
OR ICD9_CODE LIKE '438%'
'''
CERE_VASCULAR_DIS = sqldf(q, globals())

q = '''
SELECT DISTINCT HADM_ID
FROM DIAGNOSE
WHERE ICD9_CODE IN ('3312')
OR ICD9_CODE LIKE '290%'
OR ICD9_CODE LIKE '2941%'
'''
DEMENTIA = sqldf(q, globals())

q = '''
SELECT DISTINCT HADM_ID
FROM DIAGNOSE
WHERE ICD9_CODE IN ('4168', '4169', '501', '502', '503', 
                    '504', '505', '5064', '5081', '5088') 
OR ICD9_CODE LIKE '290%'
OR ICD9_CODE LIKE '2941%'
OR ICD9_CODE LIKE '4930%'
OR ICD9_CODE LIKE '4931%'
OR ICD9_CODE LIKE '4932%'
OR ICD9_CODE LIKE '4938%'
OR ICD9_CODE LIKE '4939%'
OR ICD9_CODE LIKE '494%'
OR ICD9_CODE LIKE '495%'
OR ICD9_CODE LIKE '496%'
'''
COPD = sqldf(q, globals())

q = '''
SELECT DISTINCT HADM_ID
FROM DIAGNOSE
WHERE ICD9_CODE IN ('4465', '7100', '7101', '7102',
                    '7103', '7104', '7140', '7141',
                    '7142', '7143', '71481', '725')
'''
RHEUMATIC = sqldf(q, globals())

q = '''
SELECT DISTINCT HADM_ID
FROM DIAGNOSE
WHERE ICD9_CODE LIKE '5330%'
OR ICD9_CODE LIKE '5331%'
OR ICD9_CODE LIKE '5332%'
OR ICD9_CODE LIKE '5333%'
OR ICD9_CODE LIKE '5334%'
OR ICD9_CODE LIKE '5335%'
OR ICD9_CODE LIKE '5336%'
OR ICD9_CODE LIKE '5337%'
OR ICD9_CODE LIKE '5339%'
OR ICD9_CODE LIKE '5340%'
OR ICD9_CODE LIKE '5341%'
OR ICD9_CODE LIKE '5342%'
OR ICD9_CODE LIKE '5343%'
OR ICD9_CODE LIKE '5344%'
OR ICD9_CODE LIKE '5345%'
OR ICD9_CODE LIKE '5346%'
OR ICD9_CODE LIKE '5347%'
OR ICD9_CODE LIKE '5349%'
'''
PEPTIC = sqldf(q, globals())

q = '''
SELECT DISTINCT HADM_ID
FROM DIAGNOSE
WHERE ICD9_CODE IN ('570', '5733', '5734', '5738', '5739',
                    'V427')
OR ICD9_CODE LIKE '571%'
'''
MILD_LIVER = sqldf(q, globals())

q = '''
SELECT DISTINCT HADM_ID
FROM DIAGNOSE
WHERE ICD9_CODE LIKE '401%'
OR ICD9_CODE LIKE '402%'
OR ICD9_CODE LIKE '403%'
OR ICD9_CODE LIKE '404%'
OR ICD9_CODE LIKE '405%'
'''
HYPERTENSION = sqldf(q, globals())

q = '''
SELECT DISTINCT HADM_ID
FROM DIAGNOSE
WHERE ICD9_CODE LIKE '2500%'
OR ICD9_CODE LIKE '2501%'
OR ICD9_CODE LIKE '2502%'
OR ICD9_CODE LIKE '2503%'
OR ICD9_CODE LIKE '2508%'
OR ICD9_CODE LIKE '2509%'
'''
DIABETES_WITHOUT_CH = sqldf(q, globals())

q = '''
SELECT DISTINCT HADM_ID
FROM DIAGNOSE
WHERE ICD9_CODE LIKE '2504%'
OR ICD9_CODE LIKE '2505%'
OR ICD9_CODE LIKE '2506%'
OR ICD9_CODE LIKE '2507%'
'''
DIABETES_WITH_CH = sqldf(q, globals())

q = '''
SELECT DISTINCT HADM_ID
FROM DIAGNOSE
WHERE ICD9_CODE IN ('3341', '3430', '3431', '3432', '3433',
                    '3434', '3438', '3439', '3441', '3442', '34440',
                    '3449')
OR ICD9_CODE LIKE '3420%'
OR ICD9_CODE LIKE '3421%'
OR ICD9_CODE LIKE '3428%'
OR ICD9_CODE LIKE '3429%'
OR ICD9_CODE LIKE '3440%'
OR ICD9_CODE LIKE '3443%'
'''
PARAPLEGIA = sqldf(q, globals())

q = '''
SELECT DISTINCT HADM_ID
FROM DIAGNOSE
WHERE ICD9_CODE IN ('5880', 'V420')
OR ICD9_CODE LIKE '4030%'
OR ICD9_CODE LIKE '4031%'
OR ICD9_CODE LIKE '4039%'
OR ICD9_CODE LIKE '4040%'
OR ICD9_CODE LIKE '4041%'
OR ICD9_CODE LIKE '4049%'
OR ICD9_CODE LIKE '582%'
OR ICD9_CODE LIKE '583%'
OR ICD9_CODE LIKE '585%'
OR ICD9_CODE LIKE '586%'
OR ICD9_CODE LIKE 'V451%'
OR ICD9_CODE LIKE 'V56%'
'''
RENAL = sqldf(q, globals())

q = '''
SELECT DISTINCT HADM_ID
FROM DIAGNOSE
WHERE ICD9_CODE IN ('2386')
OR ICD9_CODE LIKE '14%'
OR ICD9_CODE LIKE '15%'
OR ICD9_CODE LIKE '16%'
OR ICD9_CODE LIKE '17%'
OR ICD9_CODE LIKE '18%'
OR ICD9_CODE LIKE '190%'
OR ICD9_CODE LIKE '191%'
OR ICD9_CODE LIKE '192%'
OR ICD9_CODE LIKE '193%'
OR ICD9_CODE LIKE '194%'
OR ICD9_CODE LIKE '195%'
OR ICD9_CODE LIKE '203%'
'''
TUMOR = sqldf(q, globals())

q = '''
SELECT DISTINCT HADM_ID
FROM DIAGNOSE
WHERE ICD9_CODE LIKE '200%'
OR ICD9_CODE LIKE '201%'
OR ICD9_CODE LIKE '202%'
'''
LEUKE = sqldf(q, globals())

q = '''
SELECT DISTINCT HADM_ID
FROM DIAGNOSE
WHERE ICD9_CODE LIKE '204%'
OR ICD9_CODE LIKE '205%'
OR ICD9_CODE LIKE '206%'
OR ICD9_CODE LIKE '207%'
OR ICD9_CODE LIKE '208%'
'''
LYMPH = sqldf(q, globals())

q = '''
SELECT DISTINCT HADM_ID
FROM DIAGNOSE
WHERE ICD9_CODE LIKE '196%'
OR ICD9_CODE LIKE '197%'
OR ICD9_CODE LIKE '198%'
OR ICD9_CODE LIKE '199%'
'''
MOD_LIVER = sqldf(q, globals())

q = '''
SELECT DISTINCT HADM_ID
FROM DIAGNOSE
WHERE ICD9_CODE IN ('4560', '4561', '5722',
                    '5724', '5728')
OR ICD9_CODE LIKE '4562%'
'''
META_TUMOR = sqldf(q, globals())

q = '''
SELECT DISTINCT HADM_ID
FROM DIAGNOSE
WHERE ICD9_CODE IN ('29620', '29621', '29622', '29623', '29624',
                    '29625', '29626', '29630', '29631', '29632',
                    '29633', '311', '30112', '30113', '30110')
'''
DEPRESSION = sqldf(q, globals())

# q = '''
# SELECT DISTINCT HADM_ID
# FROM DIAGNOSE
# WHERE ICD9_CODE IN ('42', '43', '44')
# '''
# AIDS = sqldf(q, globals())

q = '''
SELECT DISTINCT A.*,
  CASE WHEN B.HADM_ID IS NOT NULL THEN 1 ELSE 0 END AS 'MI',
  CASE WHEN B1.HADM_ID IS NOT NULL THEN 1 ELSE 0 END AS 'PER_VASCULAR_DIS',
  CASE WHEN B2.HADM_ID IS NOT NULL THEN 1 ELSE 0 END AS 'DEMENTIA',
  CASE WHEN B3.HADM_ID IS NOT NULL THEN 1 ELSE 0 END AS 'COPD',
  CASE WHEN B4.HADM_ID IS NOT NULL THEN 1 ELSE 0 END AS 'RHEUMATIC',
  CASE WHEN B5.HADM_ID IS NOT NULL THEN 1 ELSE 0 END AS 'PEPTIC',
  CASE WHEN B6.HADM_ID IS NOT NULL THEN 1 ELSE 0 END AS 'MILD_LIVER',
  CASE WHEN B7.HADM_ID IS NOT NULL THEN 1 ELSE 0 END AS 'DIABETES_WITHOUT_CH',
  CASE WHEN B8.HADM_ID IS NOT NULL THEN 1 ELSE 0 END AS 'DIABETES_WITH_CH',
  CASE WHEN B9.HADM_ID IS NOT NULL THEN 1 ELSE 0 END AS 'PARAPLEGIA',
  CASE WHEN B10.HADM_ID IS NOT NULL THEN 1 ELSE 0 END AS 'RENAL',
  CASE WHEN B11.HADM_ID IS NOT NULL THEN 1 ELSE 0 END AS 'TUMOR',
  CASE WHEN B12.HADM_ID IS NOT NULL THEN 1 ELSE 0 END AS 'MOD_LIVER',
  CASE WHEN B13.HADM_ID IS NOT NULL THEN 1 ELSE 0 END AS 'META_TUMOR',
  CASE WHEN B14.HADM_ID IS NOT NULL THEN 1 ELSE 0 END AS 'SMOKE',
  CASE WHEN B15.HADM_ID IS NOT NULL THEN 1 ELSE 0 END AS 'CERE_VASCULAR_DIS',
  CASE WHEN B16.HADM_ID IS NOT NULL THEN 1 ELSE 0 END AS 'LYMPH',
  CASE WHEN B17.HADM_ID IS NOT NULL THEN 1 ELSE 0 END AS 'LEUKE',
  CASE WHEN B18.HADM_ID IS NOT NULL THEN 1 ELSE 0 END AS 'HYPERLIPIDEMIA',
  CASE WHEN B19.HADM_ID IS NOT NULL THEN 1 ELSE 0 END AS 'HYPERTENSION',
  CASE WHEN B20.HADM_ID IS NOT NULL THEN 1 ELSE 0 END AS 'DEPRESSION',
  CASE WHEN B21.HADM_ID IS NOT NULL THEN 1 ELSE 0 END AS 'AFIB'
FROM MONO_2 A
LEFT JOIN MI B
    ON A.HADM_ID = B.HADM_ID
LEFT JOIN PER_VASCULAR_DIS B1
    ON A.HADM_ID = B1.HADM_ID
LEFT JOIN DEMENTIA B2
    ON A.HADM_ID = B2.HADM_ID   
LEFT JOIN COPD B3
    ON A.HADM_ID = B3.HADM_ID  
LEFT JOIN RHEUMATIC B4
    ON A.HADM_ID = B4.HADM_ID  
LEFT JOIN PEPTIC B5
    ON A.HADM_ID = B5.HADM_ID  
LEFT JOIN MILD_LIVER B6
    ON A.HADM_ID = B6.HADM_ID  
LEFT JOIN DIABETES_WITHOUT_CH B7
    ON A.HADM_ID = B7.HADM_ID  
LEFT JOIN DIABETES_WITH_CH B8
    ON A.HADM_ID = B8.HADM_ID  
LEFT JOIN PARAPLEGIA B9
    ON A.HADM_ID = B9.HADM_ID  
LEFT JOIN RENAL B10
    ON A.HADM_ID = B10.HADM_ID   
LEFT JOIN TUMOR B11
    ON A.HADM_ID = B11.HADM_ID
LEFT JOIN MOD_LIVER B12
    ON A.HADM_ID = B12.HADM_ID  
LEFT JOIN META_TUMOR B13
    ON A.HADM_ID = B13.HADM_ID  
LEFT JOIN FINAL_SMOKE B14
    ON A.HADM_ID = B14.HADM_ID  
LEFT JOIN CERE_VASCULAR_DIS B15
    ON A.HADM_ID = B15.HADM_ID
LEFT JOIN LYMPH B16
    ON A.HADM_ID = B16.HADM_ID  
LEFT JOIN LEUKE B17
    ON A.HADM_ID = B17.HADM_ID
LEFT JOIN HYPERLIPIDEMIA B18
    ON A.HADM_ID = B18.HADM_ID 
LEFT JOIN HYPERTENSION B19
    ON A.HADM_ID = B19.HADM_ID 
LEFT JOIN DEPRESSION B20
    ON A.HADM_ID = B20.HADM_ID 
LEFT JOIN AFIB B21
    ON A.HADM_ID = B21.HADM_ID     
'''
FINAL = sqldf(q, globals())

FINAL['age_score'] = np.where(FINAL['AGE'] < 50, 0,
                 np.where(FINAL['AGE'] < 60, 1,
                 np.where(FINAL['AGE'] < 70, 2,
                 np.where(FINAL['AGE'] < 80, 3, 4))))

FINAL['CCI_SCORE'] = (
    FINAL['DIABETES_WITHOUT_CH'] + FINAL['COPD'] + 2 * FINAL['RENAL'] + FINAL['MI'] + 
    FINAL['CERE_VASCULAR_DIS'] + FINAL['DEMENTIA'] + FINAL['RHEUMATIC'] + FINAL['PER_VASCULAR_DIS'] +
    FINAL['PEPTIC'] + FINAL['MILD_LIVER'] + 2 * FINAL['DIABETES_WITH_CH'] + 
    2 * FINAL['PARAPLEGIA'] + 2 * FINAL['TUMOR'] + 3 * FINAL['MOD_LIVER'] + 
    6 * FINAL['META_TUMOR'] + 2 * FINAL['LEUKE'] +
    2 * FINAL['LYMPH'] + FINAL['age_score'])

COLS_TO_DEL = ['SUBJECT_ID', 'DOB', 'DOD', 'HADM_ID', 'ADMITTIME',
       'DISCHTIME', 'CRP_W1', 'CRP_W2', 'DEAD_DAYS', 'age_score' ,'ITEMID_1', 
       'KG/GMS_1','ITEMID', 'KG/GMS','INCH/CM']
final = FINAL.drop(COLS_TO_DEL, axis = 1)

plt.figure(figsize=(10, 6))
stats.probplot(MONO_2['MONO_1'].dropna(), dist="norm", plot=plt)
plt.title("Q-Q Plot")
plt.show()

columns_with_na_mean = ['HB_1', 'ALB_1', 'ALB_2', 'NA_2', 'HEIGHT', 'HR_1', 'HR_2',
                        'RR_2', 'RR_1', 'SBP_1', 'SBP_2', 'WBC_1', 'WBC_2', 'MAG_1',
                        'MAG_2','K_2']
columns_with_na_medium = ['HB_2', 'BNP_2', 'BNP_1', 'CREA_1', 'CREA_2', 'BUN_2',
                          'BUN_1', 'NA_1', 'WEIGHT_1', 'WEIGHT_2', 'O_2', 'O_1', 'CRP_2',
                          'CRP_1','NEU_2', 'NEU_1','ESR_2','ESR_1','PLT_1','PLT_2',
                          'LAC_1', 'LAC_2','K_1', 'MONO_1', 'MONO_2']
for col in columns_with_na_mean:
    final[col] = final[col].fillna(final[col].mean())
for col in columns_with_na_medium:
    final[col] = final[col].fillna(final[col].median())

final['HEIGHT'] = final['HEIGHT'] / 100
final['BMI'] = final['WEIGHT_2'] / (final['HEIGHT'] ** 2)
final['BMI'] = final['BMI'].round(1)

k = np.where(final['GENDER'] == 1, 0.9, 0.7)
a = np.where(final['GENDER'] == 1, -0.411, -0.329)
crea_k = final['CREA_2'] / k
egfr = 142 * np.where(crea_k <= 1, (crea_k)**a, (crea_k)**(-1.209)) * (0.993)**final['AGE']
final['EGFR'] = np.where(final['GENDER'] == 0, egfr * 1.018, egfr)
final['EGFR'] = final['EGFR'].round(2)

final['WEIGHT_CHANGE'] = final['WEIGHT_2'] - final['WEIGHT_1']
final['WEIGHT_CHANGE'] = final['WEIGHT_CHANGE'].round(2)