import pandas as pd
from google import genai
from google.genai import types
import time 

train_df= pd.read_csv('D:/研究所/1132/自然語言處理/作業/kaggle_trainset.csv')
test_df= pd.read_csv('D:/研究所/1132/自然語言處理/作業/kaggle_testset.csv')

random_state= 42


label_map = {'neoplasms': 1,
             'digestive system diseases': 2,
             'nervous system diseases': 3,
             'cardiovascular diseases': 4,
             'general pathological conditions': 5
             }
train_df['label'] = train_df['label'].map(label_map)

client = genai.Client(api_key = 'api_key')

train_df_1 = train_df[:1500]
train_df_2 = train_df[1500:3000]
train_df_3 = train_df[3000:4500]
train_df_4 = train_df[4500:6000]
train_df_5 = train_df[6000:7500]
train_df_6 = train_df[7500:9000]
train_df_7 = train_df[9000:10500]
train_df_8 = train_df[10500:12000]
train_df_9 = train_df[12000:]

def data(df):
    train_1 = df[:100]
    train_2 = df[100:200]
    train_3 = df[200:300]
    train_4 = df[300:400]
    train_5 = df[400:500]
    train_6 = df[500:600]
    train_7 = df[600:700]
    train_8 = df[700:800]
    train_9 = df[800:900]
    train_10 = df[900:1000]
    train_11 = df[1000:1100]
    train_12 = df[1100:1200]
    train_13 = df[1200:1300]
    train_14 = df[1300:1400]
    train_15 = df[1400:]

    return train_1, train_2, train_3, train_4, train_5, train_6, train_7, train_8, train_9, train_10, train_11, train_12, train_13, train_14, train_15

train_1, train_2, train_3, train_4, train_5, train_6, train_7, train_8, train_9, train_10, train_11, train_12, train_13, train_14, train_15 = data(train_df_1)

def generate_embeddings_dataframe_separate_cols_batched(client, model_name, df, text_column, batch_size=1500, sleep_duration=60):
    all_embeddings = []
    original_texts = []
    num_rows = len(df)
    num_batches = (num_rows + batch_size - 1) // batch_size

    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, num_rows)
        batch_df = df.iloc[start_index:end_index]
        texts_to_embed = batch_df[text_column].tolist()
        original_texts.extend(texts_to_embed)

        try:
            result = client.models.embed_content(
                model=model_name,
                contents=texts_to_embed
            )
            embeddings = [embedding.values for embedding in result.embeddings]
            all_embeddings.extend(embeddings)
            print(f"已處理完第 {i+1}/{num_batches} 批，共 {len(embeddings)} 筆資料。")

        except Exception as e:
            print(f"處理第 {i+1}/{num_batches} 批資料時發生錯誤：{e}")
            print("將已成功處理的資料輸出。")
            break  

        if i < num_batches - 1:
            print(f"等待 {sleep_duration} 秒後處理下一批...")
            time.sleep(sleep_duration)

    if all_embeddings:
        embeddings_df = pd.DataFrame(all_embeddings)
        num_dimensions = embeddings_df.shape[1] if not embeddings_df.empty else 0
        embeddings_df.columns = [f'embedding_dim_{i+1}' for i in range(num_dimensions)]
        final_df = pd.DataFrame({df.columns[df.columns.get_loc(text_column)]: original_texts})
        final_df = pd.concat([final_df, embeddings_df.reset_index(drop=True)], axis=1)
        return final_df
    else:
        print("沒有成功生成任何嵌入向量。")
        return pd.DataFrame()

embedding_model_name = 'models/text-embedding-004'
text_column_name = 'condition'

embeddings_df_batched_1 = generate_embeddings_dataframe_separate_cols_batched(
    client=client, 
    model_name=embedding_model_name,
    df=test_df,
    text_column=text_column_name,
    batch_size=100,
    sleep_duration=60)

y = train_df['label']
final_df = pd.concat([y, embeddings_df_batched],axis = 1)

embeddings_df_batched_1.to_csv('D:/研究所/1132/自然語言處理/作業/gemini_emb_test.csv', index= False)
embeddings_df_batched_1.to_pickle('D:/研究所/1132/自然語言處理/作業/gemini_emb_test.pkl')

embedding_model_name = 'models/text-embedding-004'
embedding_dimension = 3072
embedding_task_type = 'classification'
text_column_name = 'condition'

test1_1 = generate_embeddings_dataframe_separate_cols(
    client=client, 
    model_name=embedding_model_name,
    df=train_1,
    text_column=text_column_name,
    task_type=embedding_task_type,
    output_dimensionality=embedding_dimension
)

total_test_8 = pd.concat([test1_1, test1_2, test1_3, test1_4, test1_5, test1_6,
                        test1_7, test1_8, test1_9, test1_10, test1_11, test1_12,
                        test1_13, test1_14, test1_15], axis = 0)

final_total_test = pd.concat([df_merged1, df_merged2, df_merged3, df_merged4,
                              df_merged5, df_merged6, df_merged7, df_merged8,
                              df_merged9_1], axis= 0)