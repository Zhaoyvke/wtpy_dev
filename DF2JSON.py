import pandas as pd
import json
import re

def parquet_bond_to_json(parquet_file, json_file):
    df = pd.read_parquet(parquet_file)
    df = df.sort_values('date')
    df = df.groupby('order_book_id').last().reset_index()

    #做正股股票的处理部分
    # 删除非英文字符转大写
   #wordpart = (re.sub(r'[^a-zA-Z]', '', ('order_book_id'))).upper()
   #numeric_part = (re.findall(r'\d+', df.loc[i, "exchange"]))
    result = {}
    for i in range(len(df)):
        numeric_part = re.findall(r'\d+', df.loc[i, "order_book_id"])
        numeric_part = ''.join(numeric_part)


        #BOND的处理部分
        df.loc[df['exchange'] == 'XSHG', 'exchange'] = 'SSE'
        df.loc[df['exchange'] == 'XSHE', 'exchange'] = 'SZSE'
        name = {
            "name": df.loc[i, "symbol"],
            "code": numeric_part,
            "exchg": df.loc[i, "exchange"],
            "product": "bond"
        }
        result[numeric_part] = name

    with open(json_file, "w") as f:
        json.dump(result, f, ensure_ascii=False,indent=1)


def parquet_stk_to_josn(parquet_file, json_file):
    df = pd.read_parquet(parquet_file)
    df = df.sort_values('datetime')
    df = df.groupby('order_book_id').last().reset_index()
    df['exchange'] = df['order_book_id'].str.split('.', expand=True)[1]
            # 将XSHG替换为SSE，将XSHE替换为SZSE
    df.loc[df['exchange'] == 'XSHG', 'exchange'] = 'SSE'
    df.loc[df['exchange'] == 'XSHE', 'exchange'] = 'SZSE'
    #做正股股票的处理部分
    # 删除非英文字符转大写
    wordpart = (re.sub(r'[^a-zA-Z]', '', ('order_book_id'))).upper()
   #numeric_part = (re.findall(r'\d+', df.loc[i, "exchange"]))
    result = {}
    for i in range(len(df)):
        numeric_part = re.findall(r'\d+', df.loc[i, "order_book_id"])
        numeric_part = ''.join(numeric_part)
        #做正股股票的处理部分
        #df.loc[df['order_book_id']] = df['order_book_id'].str.replace('XSHG', 'SSE').str.replace('XSHE', 'SZSE')
        #wordpart = (re.sub(r'[^a-zA-Z]', '', df.loc[i, "order_book_id"])).upper()
        #wordpart =''.join(wordpart)
        #if(df.loc[df[wordpart] == 'XSHG']):
        #    wordpart = 'SSE'
        #elif (df.loc[df[wordpart] == 'XSHE']):
        #    wordpart = 'SZSE'    
        name = {
            "name": df.loc[i, "order_book_id"],
            "code": numeric_part,
            "exchg": df.loc[i, "exchange"],
            "product": "STK"
        }
        result[numeric_part] = name

    with open(json_file, "w") as f:
        json.dump(result, f, ensure_ascii=False,indent=1)


def readfile(path_file):
    # 读取Parquet文件
    df = pd.read_parquet(path_file)
    # 打印第一行数据
    print(df.head(1))

# Example usage:
#parquet_bond_to_json(r"E:\\wtpy_master\\wtpy-master\\DATA\\stock_1min.parquet", "E:\\wtpy_master\\wtpy-master\\DATA\\stock_1min.json.json")
parquet_stk_to_josn(r"E:\\wtpy_master\\wtpy-master\\DATA\\stock_1min.parquet", "E:\\wtpy_master\\wtpy-master\\DATA\\stock_1min.json")

#readfile("E:\\wtpy_master\\wtpy-master\\DATA\\stock_1min.parquet")
