train_file_name = ('C:/ASA/kaggle_train_dataset.csv')
test_file_name = ('C:/ASA/kaggle_test_dataset.csv')
train_df = pd.read_csv(train_file_name)
test_df = pd.read_csv(test_file_name)

df_to_cal = [train_df, test_df]
for df in df_to_cal:
    df['Anesthesia_Method'] = df['Anesthesia_Method'].str.strip().str.upper()

strcols_tolow = ['Anesthesia_Method', 'Surgery_Name',
                 'Medication_Usage', 'Lab_Values',
                 'properties_display', 'Catheter_Use']
for col in strcols_tolow:
    mask = ~train_df[col].isna()
    train_df.loc[mask, col] = train_df.loc[mask, col].str.lower()
    mask = ~test_df[col].isna()
    test_df.loc[mask, col] = test_df.loc[mask, col].str.lower()

counts = train_df['Surgery_Name'].value_counts()
mapping_dict = {}
current_group = 0
current_count = 0
for surgery, count in counts.items():
   if (current_count + count) > 250:  
       current_group += 1 
       current_count = count  
   else:
       current_count += count  
   mapping_dict[surgery] = current_group
last_group = max(mapping_dict.values()) + 1
for df in df_to_cal:
   df['Surgery_Group'] = df['Surgery_Name'].fillna('NA').map(lambda x: mapping_dict.get(x, last_group))

counts = train_df['Medication_Usage'].value_counts()
mapping_dict = {}
current_group = 0
current_count = 0
for surgery, count in counts.items():
   if (current_count + count) > 1300:  
       current_group += 1 
       current_count = count  
   else:
       current_count += count  
   mapping_dict[surgery] = current_group
last_group = max(mapping_dict.values()) + 1
for df in df_to_cal:
    df['Medication_Group'] = df['Medication_Usage'].fillna('NA').map(lambda x: mapping_dict.get(x, last_group))

counts = train_df['Catheter_Use'].value_counts()
mapping_dict = {}
current_group = 0
current_count = 0
for surgery, count in counts.items():
   if (current_count + count) > 760:  
       current_group += 1 
       current_count = count  
   else:
       current_count += count  
   mapping_dict[surgery] = current_group
for df in df_to_cal:
    df['Catheter_Group'] = df['Catheter_Use'].fillna('NA').map(lambda x: mapping_dict.get(x, last_group))

for df in df_to_cal:
    mask1 = (df['Medication_Usage'].str.contains('zosin', na = False) |
            df['Medication_Usage'].str.contains('aspirin', na = False) |
            df['Medication_Usage'].str.contains('enoxaparin', na = False) |
            (df['Medication_Usage'].str.contains('heparin', na = False) &
             df['Medication_Usage'].str.contains('subcutaneous', na = False)) |
            df['Medication_Usage'].str.contains('clopidogrel', na = False)) 
    
    mask2 = (df['Medication_Usage'].str.contains('hydrochlorothiazide', na = False) |
            df['Medication_Usage'].str.contains('digoxin', na = False) |
            df['Medication_Usage'].str.contains('furosemide', na = False) |
            df['Medication_Usage'].str.contains('spironolactone', na = False))
    
    mask3 = ((df['Medication_Usage'].str.contains('lidocaine', na = False) &
             (df['Medication_Usage'].str.contains('oral', na = False) |
             df['Medication_Usage'].str.contains('intravenous', na = False))) |
            df['Medication_Usage'].str.contains('diltiazem', na = False) |
            df['Medication_Usage'].str.contains('amiodarone', na = False) |
            df['Medication_Usage'].str.contains('dofetilide', na = False))
    df['heart'] = (mask1 | mask2 | mask3).astype(int)

for df in df_to_cal:
    mask1 = (df['Medication_Usage'].str.contains('sartan', na = False) |
            df['Medication_Usage'].str.contains('statin', na = False) |
            df['Medication_Usage'].str.contains('clonidine', na = False) |
            df['Medication_Usage'].str.contains('hydralazine', na = False) |
        df['Medication_Usage'].str.contains('(?<!-)pril', regex = True, na = False) |
        df['Medication_Usage'].str.contains('osin', na = False) |
        (df['Medication_Usage'].str.contains('lol', na = False) & 
         ~df['Medication_Usage'].str.contains('timolol', na = False)) |
        (df['Medication_Usage'].str.contains('pine', na = False) & 
         ~df['Medication_Usage'].str.contains('pilocarpine', na = False) &
         ~df['Medication_Usage'].str.contains('epinephrine', na = False)))
    mask2 = ((df['Medication_Usage'].str.contains('insulin', na=False)))
    df['two_hi'] = (mask1 | mask2).astype(int)

for df in df_to_cal:
    mask = (df['Medication_Usage'].str.contains('gabapentin', na = False) |
            df['Medication_Usage'].str.contains('pregabalin', na = False) |
            df['Medication_Usage'].str.contains('levetiracetam', na = False) |
            df['Medication_Usage'].str.contains('carbamazepine', na = False))
    df['EPILEPSY'] = mask.astype(int)

for df in df_to_cal:
    mask = (df['Medication_Usage'].str.contains('ondansetron', na = False)) 
    df['ALLERGY'] = mask.astype(int)
    
for df in df_to_cal:
    mask = (df['Medication_Usage'].str.contains('morphine', na = False) |
            df['Medication_Usage'].str.contains('codone', na = False)) 
    df['OPIOID'] = mask.astype(int)

for df in df_to_cal:
    heart_mask1 = (
        df['Medication_Usage'].str.contains('zosin', na=False) |
        df['Medication_Usage'].str.contains('aspirin', na=False) |
        df['Medication_Usage'].str.contains('enoxaparin', na=False) |
        (df['Medication_Usage'].str.contains('heparin', na=False) &
         df['Medication_Usage'].str.contains('subcutaneous', na=False)) |
        df['Medication_Usage'].str.contains('clopidogrel', na=False))
    
    heart_mask2 = (
        df['Medication_Usage'].str.contains('hydrochlorothiazide', na=False) |
        df['Medication_Usage'].str.contains('digoxin', na=False) |
        df['Medication_Usage'].str.contains('furosemide', na=False) |
        df['Medication_Usage'].str.contains('spironolactone', na=False))
    
    heart_mask3 = (
        (df['Medication_Usage'].str.contains('lidocaine', na=False) &
         (df['Medication_Usage'].str.contains('oral', na=False) |
          df['Medication_Usage'].str.contains('intravenous', na=False))) |
        df['Medication_Usage'].str.contains('diltiazem', na=False) |
        df['Medication_Usage'].str.contains('amiodarone', na=False) |
        df['Medication_Usage'].str.contains('dofetilide', na=False))
    
    bp_mask = (
        df['Medication_Usage'].str.contains('sartan', na=False) |
        df['Medication_Usage'].str.contains('statin', na=False) |
        df['Medication_Usage'].str.contains('clonidine', na=False) |
        df['Medication_Usage'].str.contains('hydralazine', na=False) |
        df['Medication_Usage'].str.contains('(?<!-)pril', regex=True, na=False) |
        df['Medication_Usage'].str.contains('osin', na=False) |
        (df['Medication_Usage'].str.contains('lol', na=False) & 
         ~df['Medication_Usage'].str.contains('timolol', na=False)) |
        (df['Medication_Usage'].str.contains('pine', na=False) & 
         ~df['Medication_Usage'].str.contains('pilocarpine', na=False) &
         ~df['Medication_Usage'].str.contains('epinephrine', na=False)))
    
    diabetes_mask = (
        df['Medication_Usage'].str.contains('insulin', na=False))
    
    epilepsy_mask = (
        df['Medication_Usage'].str.contains('gabapentin', na=False) |
        df['Medication_Usage'].str.contains('pregabalin', na=False) |
        df['Medication_Usage'].str.contains('levetiracetam', na=False) |
        df['Medication_Usage'].str.contains('carbamazepine', na=False))
    
    allergy_mask = df['Medication_Usage'].str.contains('ondansetron', na=False)
    
    opioid_mask = (
        df['Medication_Usage'].str.contains('morphine', na=False) |
        df['Medication_Usage'].str.contains('codone', na=False))
    
    df['medication_category'] = 0  
    df.loc[heart_mask1, 'medication_category'] = 1
    df.loc[heart_mask2 & ~heart_mask1, 'medication_category'] = 2
    df.loc[heart_mask3 & ~heart_mask1 & ~heart_mask2, 'medication_category'] = 3
    df.loc[bp_mask & ~heart_mask1 & ~heart_mask2 & ~heart_mask3, 'medication_category'] = 4
    df.loc[diabetes_mask & ~heart_mask1 & ~heart_mask2 & ~heart_mask3 & ~bp_mask, 'medication_category'] = 5
    df.loc[epilepsy_mask & ~heart_mask1 & ~heart_mask2 & ~heart_mask3 & ~bp_mask & ~diabetes_mask, 'medication_category'] = 6
    df.loc[allergy_mask & ~heart_mask1 & ~heart_mask2 & ~heart_mask3 & ~bp_mask & ~diabetes_mask & ~epilepsy_mask, 'medication_category'] = 7
    df.loc[opioid_mask & ~heart_mask1 & ~heart_mask2 & ~heart_mask3 & ~bp_mask & ~diabetes_mask & ~epilepsy_mask & ~allergy_mask, 'medication_category'] = 8

keywords_dict = {
    0: ['av', 'cabg', 'angio', 'vess', 'aort', 'vascu', 'carotid', 
        'varicose', 'shunt', 'valve', 'aneurysm', 'vein', 'ablation', 'cardio',
        'arter', 'venous', 'thromb', 'sheath', 'pacemaker'],
    1: ['crani', 'cns', 'pituitary', 'brain', 'head', 'arteriovenous',
        'seeg', 'nerve', 'csf', 'neuro', 'temporal', 'skull', 'endarterectomy'],
    2: ['bronch', 'egd', 'port', 'breast', 'mast', 'rib', 'vats', 
        'lobe', 'brachy', 'thora', 'lung', 'chest', 'esopha', 'sternum'],
    3: ['cholecystectomy', 'lapa', 'anorectal', 'kidney', 'prostate',
        'bladder', 'turbt', 'uret', 'herni', 'gastr', 'abd', 'ileo',            
        'peritoneal', 'whipple', 'penile', 'cystos', 'nephro', 'endoscop',        
        'urin', 'pancre', 'splen', 'tace', 'rectum', 'hemorr', 'colo',            
        'liver', 'anus', 'renal', 'sigmoid', 'anal', 'perirectal', 'ano',
        'penis', 'scrotum', 'gi', 'jejun', 'cyst', 'enteroscopy', 'duct',
        'stomach'],
    4: ['uter','curettage', 'hyste','gynecology', 'sacrocolpopexy', 
        'cervix', 'colpocleisis', 'section', 'myom', 'ovary', 'vagina', 'vulv'],
    5: ['spin', 'lumb', 'cervical'],
    6: ['nose', 'thyroid', 'gland', 'laryn', 'neck', 'trach', 'aspira', 
        'gloss', 'sinus', 'lymph', 'rhino', 'maxill', 'tonsi', 'nasal',
        'lip', 'transoral', 'ear', 'fess', 'naso'],
    7: ['femur', 'tibia', 'ankle', 'hip', 'toe', 'knee', 'hemiar', 
        'acetabulum', 'foot', 'sacral', 'pelvix', 'lower', 'orif', 'allograft',
        'amputat', 'joint', 'arthro', 'fasciotomy', 'ligament', 'acl'],
    8: ['humerus', 'radi', 'hand', 'wrist', 'carpal',
        'thumb', 'finger', 'shoulder', 'mandible', 'facial',
        'clavicle', 'arm', 'upper', 'elbow'],
    9: ['retinal', 'globe', 'vitrectomy', 'cata', 'glaucoma', 
        'orbit', 'eye', 'canaliculus', 'blepharoplasty', 'ocular']}

for df in df_to_cal:
    df['Surgery_Name'] = df['Surgery_Name'].fillna('')
    df['Surgery_Label'] = df['Surgery_Name'].apply(
        lambda name: next((label for label, keywords in keywords_dict.items() if any(keyword in name for keyword in keywords)), 10))

catheter_items = [
    'indwelling urinary catheter', 'external urinary catheter',
    'arterial line', 'vac', 'non-surgical airway', 'nasogastric', 'picc',
    'chest tube', 'hemodialysis', 'ecmo','cvc',
    'ventricular drainage catheter', 'closed', 'port a', 'intracranial', 
    'fecal', 'impaired', 'open', 'sheath', 'pressure ulcer', 'burn', 'lumbar',
    'microdialysis', 'nephrostomy', 'epidural', 'gastrostomy', 'ileostomy',
    'jejunostomy', 'surgical airway', 'peripheral']
train_df['TUBE'] = 0
test_df['TUBE'] = 0
for item in catheter_items:
    mask = (
        train_df['properties_display'].str.contains(item, case=False, na=False) |
        train_df['Catheter_Use'].str.contains(item, case=False, na=False))
    train_df['TUBE'] += mask.astype(int)
    mask = (
        test_df['properties_display'].str.contains(item, case=False, na=False) |
        test_df['Catheter_Use'].str.contains(item, case=False, na=False))
    test_df['TUBE'] += mask.astype(int)

pattern = r'([\w\^\.\s]+): ([\d\.]+) ([a-zA-Z%\/]+)? \((\w)\)'
for df in df_to_cal:
    df['Lab_Values'] =df['Lab_Values'].fillna('').astype(str)
    df['extracted_data'] = df['Lab_Values'].apply(lambda x: re.findall(pattern, x))

for df in df_to_cal:
    df['troponin i'] = df['extracted_data'].apply(
        lambda items: 1 if any(item[0] == 'troponin i.cardiac' and float(item[1]) > 0.16 for item in items)else 0)
    df['NA'] = df['extracted_data'].apply(
        lambda items: 1 if any(item[0] == 'sodium' and (float(item[1]) < 136 or float(item[1]) > 145) for item in items)else 0)
    df['K'] = df['extracted_data'].apply(
        lambda items: 1 if any(item[0] == 'potassium' and (float(item[1]) < 3.5 or float(item[1]) > 5.1) for item in items)else 0)
    df['WBC'] = df['extracted_data'].apply(
        lambda items: 1 if any(item[0] == 'leukocytes^^corrected for nucleated erythrocytes' and (float(item[1]) < 3 or float(item[1]) > 10) for item in items)else 0)
    df['HB'] = df['extracted_data'].apply(
        lambda items: 1 if any(item[0] == 'hemoglobin' and float(item[1]) < 10 for item in items)else 0)
    df['PLT'] = df['extracted_data'].apply(
        lambda items: 1 if any(item[0] == 'platelets' and (float(item[1]) < 150 or float(item[1]) > 400) for item in items)else 0)
    df['INR'] = df['extracted_data'].apply(
        lambda items: 1 if any(item[0] == 'coagulation tissue factor induced.inr' and (float(item[1]) < 0.8 or float(item[1]) > 1.2) for item in items)else 0)
    df['CREAT'] = df['extracted_data'].apply(
        lambda items: 1 if any(item[0] == 'creatinine' and (float(item[1]) < 0.6 or float(item[1]) > 1.2) for item in items)else 0)
    df['ESRD'] = df.apply(
        lambda row : 1 if any(
            item[0] == 'creatinine' and 
            (186 * (float(item[1]) ** -1.154) * (row['Age'] ** -0.203)) < 45 
            for item in row['extracted_data']
        ) else 0, axis=1)
    df['O'] = df['extracted_data'].apply(
        lambda items: 1 if any(item[0] == 'oxygen saturation' and float(item[1]) < 90 for item in items)else 0)
    df['GLU'] = df['extracted_data'].apply(
        lambda items: 1 if any(item[0] == 'glucose' and float(item[1]) > 200 for item in items) else 0)
    df['CRP'] = df['extracted_data'].apply(
        lambda items: 1 if any(item[0] == 'c reactive protein' and float(item[1]) > 1 for item in items) else 0)
    df['ALB'] = df['extracted_data'].apply(
        lambda items: 1 if any(item[0] == 'albumin' and float(item[1]) < 3.5 for item in items) else 0)

for df in df_to_cal:
    df['LAB_OK'] = np.where(df[['NA', 'K', 'WBC', 'HB', 'PLT', 'INR',
                                'CREAT','GLU', 'CRP', 'ALB', 'troponin i',
                                'ESRD', 'O']].any(axis=1), 0, 1)

for df in df_to_cal:
    df['AGE_GROUP'] = pd.cut(df['Age'], 
                            bins = [-float('inf'), 30, 45, 65, 75, float('inf')], 
                            labels=[0, 1, 2, 3, 4],
                            include_lowest = True)
        
for df in df_to_cal:
    df['TOLTAL_DISEASES'] = df[['heart', 'two_hi', 'EPILEPSY',   
                                'ESRD', 'OPIOID']].sum(axis = 1)    

cols_to_drop = ['extracted_data']
train_df = train_df.drop(cols_to_drop, axis = 1)
test_df = test_df.drop(cols_to_drop, axis = 1)