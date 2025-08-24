import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
import nlpaug.augmenter.word as naw


data= pd.read_csv('D:/kaggle_trainset.csv')
test_data= pd.read_csv('D:/kaggle_testset.csv')

label_map = {"neoplasms": 1,
             "digestive system diseases": 2,
             "nervous system diseases": 3,
             "cardiovascular diseases": 4,
             "general pathological conditions": 5}
data["label"] = data["label"].map(label_map)
data["label"] = data["label"]-1

test_size = 0.1111
random_state = 4
train_df, test_df = train_test_split(data, 
                                     test_size=test_size,
                                     random_state=random_state,
                                     stratify=data["label"])

# nltk.download("wordnet")

aug_src = "wordnet"
aug_p = 0.1
synonym_augment = naw.SynonymAug(aug_src=aug_src, aug_p=aug_p)

new_0= train_df[train_df['label']== 0]
new_1= train_df[train_df['label']== 1]
new_2= train_df[train_df['label']== 2]
new_3= train_df[train_df['label']== 3]
new_4= train_df[train_df['label']== 4]

new_1["condition_augment"] = synonym_augment.augment(new_1["condition"].to_list())
new_1_original= new_1.drop("condition_augment", axis= 1)
new_1_augmentation= new_1.drop("condition", axis= 1)
new_1_augmentation= new_1_augmentation.rename(columns={'condition_augment': 'condition'})
final_new_1= pd.concat([new_1_original, new_1_augmentation], axis= 0, ignore_index= True)

new_2["condition_augment"] = synonym_augment.augment(new_2["condition"].to_list())
new_2_original= new_2.drop("condition_augment", axis= 1)
new_2_augmentation= new_2.drop("condition", axis= 1)
new_2_augmentation= new_2_augmentation.rename(columns={'condition_augment': 'condition'})
final_new_2= pd.concat([new_2_original, new_2_augmentation], axis= 0, ignore_index= True)

new_0_total_rows = len(new_0)
new_0_half_rows = new_0_total_rows // 2
new_0_sampled = new_0.sample(n=new_0_half_rows, replace=False, random_state=4)
new_0_sampled["condition_augment"] = synonym_augment.augment(new_0_sampled["condition"].to_list())
new_0_sampled_augmentation= new_0_sampled.drop("condition", axis= 1)
new_0_sampled_augmentation= new_0_sampled_augmentation.rename(columns={'condition_augment': 'condition'})
final_new_0= pd.concat([new_0, new_0_sampled_augmentation], axis= 0, ignore_index= True)

new_3_total_rows = len(new_3)
new_3_half_rows = new_3_total_rows // 2
new_3_sampled = new_3.sample(n=new_3_half_rows, replace=False, random_state=4)
new_3_sampled["condition_augment"] = synonym_augment.augment(new_3_sampled["condition"].to_list())
new_3_sampled_augmentation= new_3_sampled.drop("condition", axis= 1)
new_3_sampled_augmentation= new_3_sampled_augmentation.rename(columns={'condition_augment': 'condition'})
final_new_3= pd.concat([new_3, new_3_sampled_augmentation], axis= 0, ignore_index= True)

final_total= pd.concat([final_new_1, final_new_2, final_new_0, final_new_3, new_4], axis= 0)
final_total.to_csv('D:/研究所/1132/自然語言處理/作業/newaug.csv', index= False)

train_df["condition_augment"] = synonym_augment.augment(train_df["condition"].to_list())

original= train_df.drop("condition_augment", axis= 1)
augmentation= train_df.drop("condition", axis= 1)
augmentation = augmentation.rename(columns={'condition_augment': 'condition'})

final= pd.concat([original, augmentation], axis= 0, ignore_index= True)
