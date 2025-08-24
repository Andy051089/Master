file_name = 'C:/研究所/1131/醫院臨床數據分析/作業/Lung Cancer/DATA/FINAL/7.csv'
df = pd.read_csv(file_name)

print('總共資料有:')
print(f'{len(df)} 筆資料')
print('重複資料有:')
print(f'{df.duplicated().sum()} 筆資料')
df = df.drop_duplicates()
df = df[df['SEQ_NO'] == 1]

df['HISTBEH_4'] = df['HISTBEH'].astype(str).str[:4]
small_cell = ['8002', '8033'] + [str(i) for i in range(8041, 8046)]
adenocarcinoma = ['8050', '8130', '8140', '8141', '8143', '8144', '8146', 
                  '8201', '8211', '8213', '8230', '8574', '8500'] + \
                 [str(i) for i in range(8250, 8258)] + \
                 ['8260', '8262', '8263', '8265', '8290', '8310', '8320', 
                  '8323', '8333'] + \
                 [str(i) for i in range(8480, 8482)] + \
                 ['8490', '8503'] + \
                 [str(i) for i in range(8550, 8553)] + \
                 ['8572']
squamous = [str(i) for i in range(8051, 8053)] + \
           [str(i) for i in range(8070, 8077)] + \
           [str(i) for i in range(8083, 8085)]
large = ['8013', '8012']
conditions = [
    df['HISTBEH_4'].isin(small_cell),
    df['HISTBEH_4'].isin(adenocarcinoma),
    df['HISTBEH_4'].isin(squamous),
    df['HISTBEH_4'].isin(large)]
choices = [0, 1, 2, 3]
df['CANCER_TYPE'] = np.select(conditions, choices, default = 4)
df = df.drop('HISTBEH_4', axis = 1)

df = df[(df['CANCER_TYPE'] == 1) | (df['CANCER_TYPE'] == 2) | (df['CANCER_TYPE'] == 3)]

columns_to_replace = ['WEIGHT', 'WEIGHT_1', 'HEIGHT']
df[columns_to_replace] = df[columns_to_replace].replace(999, np.nan)

df['FINAL_WEIGHT'] = np.where(df['WEIGHT'].notna(), 
                             df['WEIGHT'], 
                             df['WEIGHT_1'])
print('體重最小值:')
print(min(df['FINAL_WEIGHT']))
print('體重最大值:')
print(max(df['FINAL_WEIGHT']))
print('體重空值:')
print(f"{df['FINAL_WEIGHT'].isna().sum()} 筆資料")
print('--------------')
print('身高最小值:')
print(min(df['HEIGHT']))
print('身高最大值:')
print(max(df['HEIGHT']))
print('身高空值:')
print(f"{df['HEIGHT'].isna().sum()} 筆資料")

date_columns = ['FC_DT', 'STS_DT', 'OP_DT', 'CH_DT',
                'HORM_DT', 'IMMU_DT', 'TARGET_DT', 'RTB_DT']
for col in date_columns:
    df[col] = df[col].fillna(0).astype('int64')  
    df[col] = pd.to_datetime(df[col], 
                             format = '%Y%m%d',
                             errors = 'coerce')

date_columns = ['DOB', 'CONT', 'DEAD_DATE_1', 'D_DATE_1', 'IPD_DATE_1',
                'CPD_DATE_1', 'DIAG_DT_1']  
for col in date_columns:
   df[col] = pd.to_datetime(df[col],
                            format = '%Y/%m/%d',
                            errors = 'coerce')

df['AGE'] = df.apply(lambda x: 
    x['IPD_DATE_1'].year - x['DOB'].year - 
    ((x['IPD_DATE_1'].month, x['IPD_DATE_1'].day) < 
      (x['DOB'].month, x['DOB'].day)), axis = 1)
print('年齡空值的有:')
print(f"{df['AGE'].isna().sum()} 筆資料")    

df['DIAGAGE'] = df.apply(lambda x: 
    x['CONT'].year - x['DOB'].year - 
    ((x['CONT'].month, x['CONT'].day) < 
      (x['DOB'].month, x['DOB'].day)), axis = 1)    
print('診斷年齡空值的有:')
print(f"{df['DIAGAGE'].isna().sum()} 筆資料")
print('年齡小於診斷年齡的有:')
print(f"{(df['AGE'] < df['DIAGAGE']).sum()} 筆資料")
AGE_ROWS = df[df['AGE'] < df['DIAGAGE']].index
df = df.drop(AGE_ROWS, axis = 0)

mapping = {
    '1A': 1, '1A1': 1, '1A2': 1, '1A3': 1,
    '1B': 1, '2A': 2, '2B': 2, '3A': 3, '3B': 3,
    '3C': 3, '4': 4, '4A': 4, '4B': 4, '4C' : 4}
df['CSTAGE_GROUP'] = df['CSTAGE'].map(mapping)
df['CSTAGE'].value_counts()
print('臨床分期組合人數:')
for index, value in df['CSTAGE_GROUP'].value_counts().items():
    print(f"{index}:  {value}")
print('臨床分期組合空值:')
print(f"{df['CSTAGE_GROUP'].isna().sum()} 筆資料")
Stage_row = df[df['CSTAGE_GROUP'].isna()].index
df = df.drop(Stage_row, axis = 0)

df['FINAL_USE_DEAD_DATE'] = np.where(df['D_DATE_1'].notna(), 
                                     df['D_DATE_1'], 
                                     df['DEAD_DATE_1'])

print('抽菸空值有 :')
print(f"{df['SMOKING'].isna().sum()}人")
print('抽菸無紀錄有 :')
print(f"{(df['SMOKING'] == 999999).sum()}人")
SMOKING_ROWS = df[df['SMOKING'].isna()].index
df = df.drop(SMOKING_ROWS, axis = 0)
SMOKING_ROWS = df[df['SMOKING'] == 999999].index
df = df.drop(SMOKING_ROWS, axis = 0)
df['SMOKING'] = df['SMOKING'].apply(
    lambda x: str(int(x)).zfill(6) if pd.notna(x) and len(str(int(x))) < 6 else x)
df['SMOKING'] = df['SMOKING'].astype(str)
df['smoking_1'] = df['SMOKING'].str[0:2]
df['smoking_2'] = df['SMOKING'].str[2:4]
df['smoking_3'] = df['SMOKING'].str[4:6]
df['SMOKING_STATUS'] = 1
df.loc[(df['smoking_3'] != '00') & (df['smoking_3'] != '88'), 'SMOKING_STATUS'] = 2
df.loc[(df['smoking_1'] == '00'), 'SMOKING_STATUS'] = 0
print('抽菸狀況人數:')
for index, value in df['SMOKING_STATUS'].value_counts().items():
    print(f"{index}:  {value}")
print('抽菸狀況空值:')
print(f"{df['SMOKING_STATUS'].isna().sum()} 筆資料")

print('檳榔空值有 :')
print(f"{df['BTCHEW'].isna().sum()}人")
print('檳榔無紀錄有 :')
print(f"{(df['BTCHEW'] == 999999).sum()}人")
BTCHEW_ROWS = df[df['BTCHEW'] == 999999].index
df = df.drop(BTCHEW_ROWS, axis = 0)

df['BTCHEW'] = df['BTCHEW'].apply(
    lambda x: str(int(x)).zfill(6) if pd.notna(x) and len(str(int(x))) < 6 else x)
df['BTCHEW'] = df['BTCHEW'].astype(str)
df['BTCHEW_1'] = df['BTCHEW'].str[0:2]
df['BTCHEW_2'] = df['BTCHEW'].str[2:4]
df['BTCHEW_3'] = df['BTCHEW'].str[4:6]
df['BTCHEW_STATUS'] = 1
df.loc[(df['BTCHEW_3'] != '00') & (df['BTCHEW_3'] != '88'), 'BTCHEW_STATUS'] = 2
df.loc[(df['BTCHEW_1'] == '00'), 'BTCHEW_STATUS'] = 0
print('吃檳榔狀況人數:')
for index, value in df['BTCHEW_STATUS'].value_counts().items():
    print(f"{index}:  {value}")
print('吃檳榔狀況空值:')
print(f"{df['BTCHEW_STATUS'].isna().sum()} 筆資料")

print('喝酒空值有 :')
print(f"{df['DRINKING'].isna().sum()}人")
print('喝酒無紀錄有 :')
print(f"{(df['DRINKING'] == 999).sum()}人")
DRINKING_ROWS = df[df['DRINKING'] == 999].index
df = df.drop(DRINKING_ROWS, axis = 0)

drink_map = {0 : 0,
             1 : 1,
             2 : 2,
             3 : 2,
             4 : 2,
             9 : 2}
df['DRINKING_GROUP'] = df['DRINKING'].map(drink_map)
print('喝酒狀況人數:')
for index, value in df['DRINKING_GROUP'].value_counts().items():
    print(f"{index}:  {value}")
print('喝酒狀況空值:')
print(f"{df['DRINKING_GROUP'].isna().sum()} 筆資料")

LATERAL_ROWS = df[df['LATERAL'] == 9].index
df = df.drop(LATERAL_ROWS, axis = 0)
LATERAL_ROWS = df[df['LATERAL'] == 3].index
df = df.drop(LATERAL_ROWS, axis = 0)
lateral_map = {1 : 1,
               2 : 2,
               4 : 3}
df['LATERAL'] = df['LATERAL'].map(lateral_map)
print('側性狀況人數:')
for index, value in df['LATERAL'].value_counts().items():
    print(f"{index}:  {value}")
print('側性狀況空值:')
print(f"{df['LATERAL'].isna().sum()} 筆資料")

df['OP'] = 1
df.loc[df['OP_DT'].isna(), 'OP'] = 0
print('手術狀況人數:')
for index, value in df['OP'].value_counts().items():
    print(f"{index}:  {value}")
print('手術狀況空值:')
print(f"{df['OP'].isna().sum()} 筆資料")

ch_map = {
    86 : 0,
    0 : 1,
    82 : 1,
    85 : 1,
    87 : 1, 
    1 : 2,
    2 : 2,
    3 : 2}
df['temp_CH'] = df['CH'].map(ch_map)
df['temp_CH_O'] = df['CH_O'].map(ch_map)
df['CHEMO'] = df['temp_CH']
mask_86 = df['CH'] == 86
df.loc[mask_86, 'CHEMO'] = df.loc[mask_86, 'temp_CH_O']
mask_different = df['temp_CH'] != df['temp_CH_O']
df.loc[mask_different, 'CHEMO'] = df.loc[
    mask_different, ['temp_CH', 'temp_CH_O']].max(axis = 1)
df = df.drop(columns=['temp_CH', 'temp_CH_O'])
df['CHEMO'] = df['CHEMO'] - 1
print('化療狀況人數:')
for index, value in df['CHEMO'].value_counts().items():
    print(f"{index}:  {value}")
print('化療狀況空值:')
print(f"{df['CHEMO'].isna().sum()} 筆資料")

immu_map = {
    0 : 0,
    1 : 1,
    20 : 1,
    30 : 1}
df['temp_IMMU'] = df['IMMU'].map(immu_map)
df['temp_IMMU_O'] = df['IMMU_O'].map(immu_map)
df['IMMU_THERAPY'] = df[['temp_IMMU', 'temp_IMMU_O']].max(axis = 1)
df = df.drop(columns=['temp_IMMU', 'temp_IMMU_O'])
print('免疫治療狀況人數:')
for index, value in df['IMMU_THERAPY'].value_counts().items():
    print(f"{index}:  {value}")
print('免疫治療狀況空值:')
print(f"{df['IMMU_THERAPY'].isna().sum()} 筆資料")

target_map = {
    86 : 0,
    85 : 1,
    0 : 1,
    1 : 2}
df['temp_TARGET'] = df['TARGET'].map(ch_map)
df['temp_TARGET_O'] = df['TARGET_O'].map(ch_map)
df['TARGET_THERAPY'] = df['temp_TARGET']
mask_86 = df['TARGET'] == 86
df.loc[mask_86, 'TARGET_THERAPY'] = df.loc[mask_86, 'temp_TARGET_O']
mask_different = df['temp_TARGET'] != df['temp_TARGET_O']
df.loc[mask_different, 'TARGET_THERAPY'] = df.loc[
    mask_different, ['temp_TARGET', 'temp_TARGET_O']].max(axis = 1)
df = df.drop(columns = ['temp_TARGET', 'temp_TARGET_O'])
df['TARGET_THERAPY'] = df['TARGET_THERAPY'] - 1
print('標靶治療狀況人數:')
for index, value in df['TARGET_THERAPY'].value_counts().items():
    print(f"{index}:  {value}")
print('標靶治療狀況空值:')
print(f"{df['TARGET_THERAPY'].isna().sum()} 筆資料")

df['RT'] = 1
df.loc[df['RTB_DT'].isna(), 'RT'] = 0
print('放射治療狀況人數:')
for index, value in df['RT'].value_counts().items():
    print(f"{index}:  {value}")
print('放射治療狀況空值:')
print(f"{df['RT'].isna().sum()} 筆資料")

df['age_score'] = np.where(df['AGE'] < 50, 0,
                 np.where(df['AGE'] < 60, 1,
                 np.where(df['AGE'] < 70, 2,
                 np.where(df['AGE'] < 80, 3, 4))))
df['CCI_SCORE'] = (
    df['DM_WITHOUT_1'] + df['CPD_1'] + 2 * df['RENAL_2'] + df['MI_1'] + 
    df['CHF_1'] + df['CERE_1'] + df['DEME_1'] + df['RHEUM_1'] + 
    df['PEPTIC_1'] + df['MILD_LIVER_1'] + 2 * df['DM_WITH_2'] + 
    2 * df['PARA_2'] + 2 * df['TUMOR_2'] + 3 * df['MOD_LIVER_3'] + 
    6 * df['META_TUMOR_6'] + 6 * df['AIDS_6'] + 2 * df['LEUKE_2'] +
    2 * df['LYMPH_2'] + df['age_score'])
print('CCI分數人數:')
for index, value in df['CCI_SCORE'].value_counts().items():
    print(f"{index}:  {value}")
print('臨床分期組合空值:')
print(f"{df['CCI_SCORE'].isna().sum()} 筆資料")
print('CCI最小值:')
print(min(df['CCI_SCORE']))
print('CCI最大值:')
print(max(df['CCI_SCORE']))

df['SURVIVED_DAYS'] = (df['FINAL_USE_DEAD_DATE'] - df['DIAG_DT_1']).dt.days 

df['HALF_YEAR_SURVIVED'] = np.where(
    df['FINAL_USE_DEAD_DATE'].isna(), 1,
    np.where(df['SURVIVED_DAYS'] >= 1852, 1, 0))
print('半年後存活的有:')
print(f"{(df['HALF_YEAR_SURVIVED'] == 1).sum()} 筆資料")
print('佔所有的:')
print(f"{((df['HALF_YEAR_SURVIVED'] == 1).sum() / len(df) * 100):.2f}%")
print('------------------')
print('半年後死亡的有:')
print(f"{(df['HALF_YEAR_SURVIVED'] == 0).sum()} 筆資料")
print('佔所有的:')
print(f"{((df['HALF_YEAR_SURVIVED'] == 0).sum() / len(df) * 100):.2f}%")

df.duplicated(subset='FEE_NO').sum()
df = df.drop_duplicates(subset = 'FEE_NO')

df['HEIGHT'] = df['HEIGHT'] / 100
df['BMI'] = df['FINAL_WEIGHT'] / (df['HEIGHT'] ** 2)
df['BMI'] = df['BMI'].round(2)

cols_to_drop = ['CHR_NO', 'ID_NO', 'FEE_NO', 'DEAD_DATE', 'D_DATE', 'IPD_DATE',
 'CPD_DATE', 'EDIAG_CODE', 'MDIAG_CODE', 'ICD10_CODE1', 'ICD10_CODE1_OUT',
 'WEIGHT_1', 'BIRTH_DT', 'CONT_DT', 'HISTBEH', 'CSTAGE', 'SMOKING', 'BTCHEW',
 'DRINKING', 'DIAG_DT', 'LATERAL', 'WEIGHT', 'FC_DT', 'STS_DT', 'OP_DT', 'CH',
 'CH_O', 'CH_DT', 'HORM_O', 'HORM', 'HORM_DT', 'IMMU_O', 'IMMU', 'IMMU_DT',
 'TARGET', 'TARGET_O', 'TARGET_DT', 'RTB_DT', 'DOB', 'CONT', 'DEAD_DATE_1',
 'D_DATE_1', 'DIAG_DT_1', 'FINAL_USE_DEAD_DATE', 'smoking_1', 'smoking_2',
 'smoking_3', 'BTCHEW_1', 'BTCHEW_2', 'BTCHEW_3', 'age_score',
 'IMMU', 'IMMU_O', 'HORM', 'HORM_O', 'SURVIVED_DAYS', 'RECUR', 'FTEEN_FLAG', 
 'SEQ_NO', 'DIAGAGE', 'FST_OPD_DATE']
df = df.drop(cols_to_drop, axis = 1)
df.duplicated().sum()
df = df.drop_duplicates()
df = df.drop('ALB', axis = 1)
df = df.dropna(axis = 0)

cols_to_drop=['HEIGHT', 'FINAL_WEIGHT']
df = df.drop(cols_to_drop, axis = 1)

train_df = df[(df['HOS_GROUP'] == 'T') | (df['HOS_GROUP'] == 'W')]
test_df = df[df['HOS_GROUP'] == 'S']

train_df = train_df.drop('HOS_GROUP', axis = 1)
test_df = test_df.drop('HOS_GROUP', axis = 1)