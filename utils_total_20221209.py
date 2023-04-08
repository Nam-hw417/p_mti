import pyodbc 
import os
import glob
import pandas as pd
import numpy as np
import pickle
import re
from tqdm import tqdm
from ast import literal_eval
from itertools import chain
from collections import Counter, defaultdict
from konlpy.tag import Komoran
import nltk

import time
from datetime import datetime

from config import *

tqdm.pandas()

ko_stopwords = ['직접', '경영', '사회', '수출입', '상품', '앞쪽', '업체', '은닉', '현황', '구입', '다툼', '예외', '효능', '전문가', '국내외', '취급', '토대', '애원', '범위', '자신감', '정보', '소요', '조사', '결정', '최고', '조언', '향상', '허용', '도달', '목록', '홈페이지', '대부분', '형태', '추진구축', '임시', '지정', '이메일', '목표', '유출', '법인', '퇴장', '포함', '조치', '희망', '업무', '도모', '방법', '투자', '극복', '방해', '각종', '제안', '위협', '혼잡', '전부', '판매', '해당', '손실', '인증', '완전', '실현', '다수', '지점', '제시', '초과', '검토', '목적', '고급', '브랜드', '요망', '각광', '상담주선', '제호', '참가', '부탁', '신경', '경우', '특징', '부분', '정책', '완화', '사업', '분야', '관련', '기대', '학년', '그룹', '평소', '참조', '존재', '단계', '그중', '회의', '올해', '직영', '대응', '즉각', '본사', '자영업자', '설립', '소개', '코드', '대책', '제작', '과장', '행복', '비교', '과장', '일자', '일반', '연결', '관리자', '제품', '설명서', '관심', '과거', '분류', '접근', '이번', '용도', '설정', '주최', '주선', '바람', '내용', '고객', '정부', '개최', '발신', '수단', '획득', '프로젝트', '기타로', '대체', '조금', '메일', '바이어', '첨부', '상담', '요청', '드림']

en_stopwords=['yourselves', 'devi', 'alll', 'they', 'anna', 'nor', 'Korea', 'good', 'yours', 'stee', 'yourself', 'does', 'X', 'fund', 'andrea', 'd', 'INCEN', 'coco', 'hasn', 'customer', 'his', 'nces', 'DZEUS', 'be', 'on', "that'll", 'click', 'do', "needn't", 'epcm', 'below', 'did', 'to', 'just', 'valu', 'any', "couldn't", 'applemobilegameskr', 'until', 'mightn', 'isn', 'was', 'which', 'needn', 'Ltd', "she's", 'ourselves', 'hindu', 'own', 'what', "shouldn't", 's', 'our', 'ools', 'nori', 'eriak', 'themselves', 'cpnp', 'poly', 'each', 'dd', 'further', 'applia', 'weren', 'We', "hadn't", 'esco', 'having', 'farma', "you'll", 'consumabl', 'other', 'schoool', 'for', 'the', 'doing', 'we', 'where', "weren't", 'KOTRA', 'if', 'few', 'leisur', 'x', 'all', 'once', 'dcor', 'y', "doesn't", 'this', 't', 'between', 'otocs', 'before', 'elec', "hasn't", "don't", 'now', 'cisr', 'over', 'himself', 'only', 'To', "you're", 'such', 'there', 'engg', 'Inc', 'swcc', 'wouldn', 'SongWha', 'should', 'cond', 'off', 'me', 'Zeus', 'most', 'will', "it's", 'won', 'manuf', 'then', 'hers', 'll', 'ma', 'lase', 'herself', "shan't", 'don', 'noon', 'syst', 'ap', 'prop', 'semi', 'wwwakgreentechcokr', 'chia', 'are', 'pmma', 'no', 'down', 'didn', 'into', 'mustn', 'am', 'veau', 'oinp', 'that', 'under', 'suppleme', 'consu', "you'd", 'how', "didn't", 'doesn', 're', 'CC', 'cali', 'aren', "wouldn't", 'Company', 'couldn', 'an', 'he', 'wasn', "won't", 'electro', 'you', 'and', 'same', 'stud', 'LG', 'by', 'POSCO', 'shouldn', 'itself', 'company', 'AAAAA', 'above', 'held', 'Other', 'why', 'tracotr', 'sungl', 'with', 'etcwwwwoorirocom', 'those', 've', 'Yumins', 'very', 'o', 'coopera', 'conte', 'supe', 'after', 'as', "aren't", 'than', 'too', 'again', 'but', 'been', 'Hansung', 'ETC', 'nsk', 'haven', 'when', 'tteopoki', 'wwwisavecokr', 'theirs', 'para', 'feng', "mustn't", "haven't", 'her', 'solu', 'its', 'tele', 'ldpe', 'shan', 'ain', 'has', 'oems', 'had', 'i', 'ghee', 'combi', 'whom', 'hadn', 'reade', 'info', 'your', 'elect', 'wwwmobilegameskr', 'whol', 'basf', 'm', 'poct', 'Co', 'AAAAAAAAAA', 'out', 'is', 'yang', 'reed', "isn't", 'odor', 'being', 'supp', 'etc', 'acces', 'up', 'against', 'so', 'const', 'while', 'these', 'Others', "mightn't", 'them', 'prod', 'nec', 'from', 'in', 'othe', 'drin', 'inje', 'JEUS', "wasn't", 'their', 'EX', 'some', 'have', 'more', "you've", 'juan', 'my', 'it', 'mfrs', 'both', 'myself', 'through', 'of', 'were', 'engin', 'scre', 'during', 'not', 'hipaa', 'eqip', 'rulo', 'cosm', 'trans', 'tran', 'produ', 'can', 'she', 'Japan', 'here', 'a', 'secur', 'sudarat', 'him', 'equi', 'schoo', 'mport', 'at', 'or', 'about', 'because', "should've", 'who', 'khai', 'toppokki', 'ours', 'mgmt', 'enable', 'Woorim']

#=======================================================  공통 =======================================================     

def collect(sql):
    conn = pyodbc.connect('DSN=BP_DB;UID=bpuser;PWD=kotrabp')
    conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
    conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
    conn.setdecoding(pyodbc.SQL_WMETADATA, encoding='utf-32le')
    conn.setencoding(encoding='utf-8')

    data= pd.read_sql_query(sql, conn)
    
    conn.close()
    
    return data

def data_preprocessing_one(data): # 데이터 중복제거, 인덱스 초기화
    data = data.drop_duplicates()
    data = data.reset_index(drop=True)
    data = data.fillna('') 
    
    return data



# data_preprocessing_kor_one
def df_newline_tab_replace(data, columns):
    """
    개행 및 tab 문자열 -> '' 변경 함수
    Args:
        data=dataframe
        columns=column_name
    Returns:
        dataframe
    """
    data[columns] = data[columns].apply(lambda x: x.replace('\n', '').replace('\r','').replace('\t','').strip())
    
    return data

# data = df_newline_tab_replace(data, '대표품목명')
# data = df_newline_tab_replace(data, '대한관심품목명')


def final_preprocessing(final_result):
    """
    Buyer ID 별로 매핑된 list 내 MTICODE를 하나씩 하나의 row로 변경해 
    MTICODE 별 데이터프레임으로 반환해주는 함수 
    Args:
        final_result=최종 병합된 MTIcode list가 들어있는 dataframe
    Returns:
        dataframe
    """    
    
    df = pd.DataFrame(columns=['MTICD','BUYERID','대표품목명','대한관심품목명'])
    index = 0
    for idx, i in tqdm(final_result.iterrows()):
        bid = i['BUYERID']
        d_name = i['대표품목명']
        name = i['대한관심품목명']
        for j in i['MTICD']:
            df.loc[index] = [j, bid, d_name, name]
            index+=1
    
    return df
#=======================================================  동오 =======================================================   
# data_preprocessing_one
def fill_na(data, substitute_word):
    """
    Null 값 대체 함수
    Args:
        data=dataframe
        substitute_word=str
    Returns:
        dataframe
    """
    data = data.fillna(substitute_word) # ''

    return data


# data = fill_na(data, '')

# data_preprocessing_kor_two
def ko_noun_ext(data): 
    """
    한글 명사 추출 함수
    Args:
        data=dataframe
    Returns:
        dataframe
    """
    k = Komoran()
    data["대표품목명_ext_ko_nouns"] = data["대표품목명_ext_ko"].apply(lambda x: k.nouns(x))
    data["대한관심품목명_ext_ko_nouns"] = data["대한관심품목명_ext_ko"].apply(lambda x: k.nouns(x))
    
    return data
    
    
# data_preprocessing_kor_three
def columns_concat(data):
    """
    대표품목명(명사)과 대한관심품목명(명사) 컬럼 병합 및 중복제거
    Args:
        data=dataframe
    Returns:
        dataframe
    """

    data['hap']=''
    data = data.reset_index(drop=True)
    for i in range(data.shape[0]):
        li = []
        li1 = data.loc[i:i, '대표품목명_ext_ko_nouns'].values[0]
        li2 = data.loc[i:i, '대한관심품목명_ext_ko_nouns'].values[0]

        if li1 is not None:
            li += li1
        if li2 is not None:
            li += li2

        data.loc[i:i, 'hap'].values[0] = list(set(li))

        
    return data


# data_preprocessing_kor_four
def empty_list_drop(data):
    """
    hap 컬럼 값이 빈 리스트인 경우 제외
    Args:
        data=dataframe
    Returns:
        dataframe
    """
    data['len'] = data['hap'].apply(lambda x: len(x))
    data = data[data['len'] >= 1]
    data = data.reset_index(drop=True)
    data = data[['BUYERID','hap']]
    return data

# data_preprocessing_kor_five
def separation_col_values(data, column_name):
    """
    리스트로 된 hap 컬럼 값들을 분리
    Args:
        data=dataframe
    Returns:
        dataframe
    """
    data2 = data[column_name].apply(lambda x: pd.Series(x))
    data2 = data2.stack().reset_index(level=1, drop=True).to_frame('nouns')
    
    return data2

# data_preprocessing_kor_six
def buyerid_nouns(data_bf, data_af):
    """
    separation_col_values 결과 값에 바이어ID를 붙이는 작업 
    Args:
        data_bf=separation_col_values 처리 전 데이터
        data_af=separation_col_values 처리 결과 데이터
    Returns:
        dataframe
    """
    data = data_bf.merge(data_af, left_index=True, right_index=True, how='left')
    data = data[['BUYERID','nouns']]
    
    return data



# data_preprocessing_kor_seven
def drop_dup_and_2wnoun_ext(data):
    """
    중복제거, 명사의 길이가 2이상인 것만 추출하는 함수
    Args:
        data=dataframe
    Returns:
        dataframe
    """
    data = data.dropna()
    data = data.reset_index(drop=True)
    data_noun = data[["nouns"]]
    data_noun = data_noun.drop_duplicates()
    data_noun = data_noun.reset_index(drop=True)
    data_noun['len'] = data_noun['nouns'].apply(lambda x: len(x))
    data_noun = data_noun[data_noun['len'] >= 2]
    data_noun['result'] = 'x'
    data_noun = data_noun.reset_index(drop=True)
                     
    return data_noun


def drop_dup_and_4wnoun_ext(data):
    """
    중복제거, 명사의 길이가 4이상인 것만 추출하는 함수
    Args:
        data=dataframe
    Returns:
        dataframe
    """
    data = data.dropna()
    data = data.reset_index(drop=True)
    data_noun = data[["nouns"]]
    data_noun = data_noun.drop_duplicates()
    data_noun = data_noun.reset_index(drop=True)
    data_noun['len'] = data_noun['nouns'].apply(lambda x: len(x))
    data_noun = data_noun[data_noun['len'] >= 4]
    data_noun['result'] = 'x'
    data_noun = data_noun.reset_index(drop=True)
                     
    return data_noun

# data_preprocessing_kor_eight
def mti_map(data, mti_ko_en):
    """
    MTI_KO_EN.csv 파일 검색 결과 매핑 함수
    Args:
        data=dataframe
        mti_ko_en=dataframe(MTI_KO_EN.csv)
    Returns:
        dataframe
    """
    for i in tqdm(range(data.shape[0])):
        nu = data.loc[i:i, "nouns"].values[0]
        if len(mti_ko_en[mti_ko_en["HS_DESC_KO"].str.contains(nu)]) != 0:
            li = list(mti_ko_en[mti_ko_en["HS_DESC_KO"].str.contains(nu)]["MTICD"])
            data.loc[i:i, "result"].values[0] = li
    data = data[["nouns", "result"]]
    data["result"] = data["result"].apply(lambda x: ', '.join(x))
    data = data[data["result"] != 'x']
    data = data.reset_index(drop=True)
    data = data.astype('str')
    
   
    return data
                         
                         
# data_preprocessing_kor_nine
def buyerid_mticode(data_kor_hap, data_kor_hap_noun):
    """
    기업별 명사 추출한 테이블과 MTICODE 매핑 함수
    Args:
        data=dataframe
        data_noun=dataframe
    Returns:
        dataframe
    """
    data_kor_hap['mti'] = 'x'
    data_kor_result = pd.merge(data_kor_hap, data_kor_hap_noun, how='left', on='nouns')
    data_kor_result = data_kor_result[["BUYERID", "nouns", "result"]]
    data_kor_result.columns = ['BUYERID', "NOUNS", "MTICD"]
    data_kor_result = data_kor_result.dropna()
    
    return data_kor_result
                         
# data_preprocessing_kor_ten
def buyerid_mticode_conc(data):
    """
    바이어ID별 매핑되는 MTICODE 출력 함수
    Args:
        data=dataframe
    Returns:
        dataframe
    """
    data = data.groupby("BUYERID")["MTICD"].agg(lambda x: ', '.join(x))
    data = pd.DataFrame(data, columns =["MTICD"])
    data = data.reset_index()
    data['MTICD'] = data['MTICD'].apply(lambda x: x.split(', '))
    data['len'] = data['MTICD'].apply(lambda x: len(x))
    #data_kor_result["MTICD"] = data_kor_result["MTICD"].apply(lambda x: [x[0] for x in list(Counter.most_common(Counter(x), n=3))])
    data = data.drop('len', axis=1)
    
    return data
    
def pos_noun(x):
    """
    명사형 단어만 추출['NNG','NNB','NNP']
    Args:
        x=string
    Returns:
        list
    """
                         
    text=[]
    if isinstance(x, list):
        for i in x:
            if i[1] in ['NNG','NNB','NNP']:
                text.append(i[0])

    return text 


# data_preprocessing_eng_four
def merge_and_drop_dup(data_eng):
    """
    [대표품목명 + 대한관심품목명] + 중복제거 = hap
    Args:
        data_eng=dataframe
    Returns:
        dataframe
    """    
    data_eng['hap'] = data_eng['대표품목명_token'] + data_eng['대한관심품목명목_token']
    data_eng['hap'] = data_eng['hap'].apply(lambda x: list(set(x)))
    
    return data_eng

# data_preprocessing_eng_five
def ext_eng_drop_stopw(data_eng):
    """
    영어 추출 함수
    Args:
        data_eng=dataframe
    Returns:
        dataframe    
    """
    data_eng["대표품목명_token"] = data_eng["대표품목명"].apply(lambda x: re.sub('〮|,|/|\.|-|&|\(|\)|，|;', ', ', x))
    data_eng["대한관심품목명목_token"] = data_eng["대한관심품목명"].apply(lambda x: re.sub('〮|,|/|\.|-|&|\(|\)|，|;', ', ', x))
    data_eng["대표품목명_token"] = data_eng["대표품목명_token"].apply(lambda x: re.sub('[^a-zA-Z\s]+', '', x).strip())
    data_eng["대한관심품목명목_token"] = data_eng["대한관심품목명목_token"].apply(lambda x: re.sub('[^a-zA-Z\s]+', '', x).strip())
    con1 = data_eng['대표품목명_token'] != ''
    con2 = data_eng['대한관심품목명목_token'] != ''
    data_eng = data_eng[con1 | con2]
    data_eng = data_eng.reset_index(drop=True)
    
    return data_eng
# def ext_eng_drop_stopw(data_eng):
#     """
#     영어 추출 및 불용어 제거 함수
#     Args:
#         data_eng=dataframe
#     Returns:
#         dataframe    
#     """
#     data_eng['hap'] = data_eng['hap'].apply(lambda x: [re.sub('[^a-zA-Z]', '', i) for i in x])
#     data_eng['hap'] = data_eng['hap'].apply(lambda x: [i for i in x if i])
#     data_eng['hap_len'] = data_eng['hap'].apply(lambda x: len(x))
#     data_eng = data_eng[data_eng['hap_len'] >= 1]
#     data_eng = data_eng.reset_index(drop=True)
#     data_eng = data_eng.drop('hap_len', axis=1)
    
#     data_eng['hap'] = data_eng['hap'].apply(lambda x: [i for i in x if i not in en_stopwords])
    
#     return data_eng


# data_preprocessing_eng_seven_one
def tokenization_intritem(data_eng):
    """
    tokenize 함수
    Args:
        data_eng=dataframe
    Returns:
        dataframe  
    """
    data_eng["대표품목명_token"] = data_eng["대표품목명_token"].apply(lambda x: nltk.word_tokenize(x))
    data_eng["대한관심품목명목_token"] = data_eng["대한관심품목명목_token"].apply(lambda x: nltk.word_tokenize(x))    
    
#     data_eng["대표품목명_token"] = data_eng["대표품목명"].apply(lambda x: nltk.word_tokenize(x))
#     data_eng["대한관심품목명목_token"] = data_eng["대한관심품목명"].apply(lambda x: nltk.word_tokenize(x))
#     data_eng['hap'] = data_eng['hap'].apply(lambda x: [word for word in x if word not in en_stopwords])

#     data_eng = data_eng.reset_index(drop=True)
    
    return data_eng

def hap_len_drop(data_eng): # 변경됨2
    """
    hap의 길이가 1 이상인 행 추출 함수
    Args:
        data_eng=dataframe
    Returns:
        dataframe  
    """
    #data_eng['hap'] = data_eng['hap'].apply(lambda x: [re.sub('[^a-zA-Z\s]', '', i) for i in x])
    #data_eng['hap'] = data_eng['hap'].apply(lambda x: [i for i in x if i])
    data_eng['hap_len'] = data_eng['hap'].apply(lambda x: len(x))
    data_eng = data_eng[data_eng['hap_len'] >= 1]
    data_eng = data_eng.reset_index(drop=True)
    data_eng = data_eng.drop('hap_len', axis=1)
    
    return data_eng


# data_preprocessing_eng_seven_two
def en_noun_ext(data_eng):
    """
    품사 태깅으로 명사가 아닌 것들에 대해서 lemmatization 후 명사 추출 함수
    Args:
        data_eng=dataframe
    Returns:
        dataframe
    
    """
    data_eng['hap_lemma'] = data_eng['hap']
    for i in tqdm(range(data_eng.shape[0])):
        li = []
        for word in data_eng.loc[i:i, "hap"].values[0]: # 리스트
            for w, t in nltk.pos_tag([word]): # 리스트 내에서 하나씩
                if t not in ["NN", "NNS", "NNP", "NNPS", "JJ"]:
                    w = nltk.WordNetLemmatizer().lemmatize(w, pos='n')
                    if nltk.pos_tag([w])[0][1] in ["NN", "NNS", "NNP", "NNPS", "JJ"]:
                        #w = nltk.WordNetLemmatizer().lemmatize(w, pos='n')
                        li.append(w)
                elif t in ["NN", "NNS", "NNP", "NNPS", "JJ"]:
                    w = nltk.WordNetLemmatizer().lemmatize(w, pos='n')
                    li.append(w)
        data_eng.loc[i:i, 'hap_lemma'].values[0] = li
    
    data_eng['hap_lemma'] = data_eng['hap_lemma'].apply(lambda x: [i for i in x if i not in en_stopwords])
    
    return data_eng

# data_preprocessing_eng_ten
def buyerid_noun_list(data_eng_subone, data_eng_subtwo):
    """
    기업별 명사 목록 추출 함수
    Args:
        data_eng_subone=drop_blank 결과 dataframe
        data_eng_subtwo=separation_col_values 결과 dataframe
    Returns:
        dataframe
    
    """

    data_eng_hap = data_eng_subone.merge(data_eng_subtwo, left_index=True, right_index=True, how='left')
    data_eng_hap = data_eng_hap[["BUYERID", "nouns"]]
   
    return data_eng_hap


# data_preprocessing_eng_twelve
def noun_mti_map(data_eng_hap_noun, mti_ko_en):
    """
    명사별로 MTI_KO_EN.csv와 비교하여 결과가 존재하는 명사만 출력하는 함수
    Args:
        data_eng_hap_noun=drop_dup_and_3wnoun_ext 결과 dataframe
        mti_ko_en=dataframe(MTI_KO_EN.csv)
    Returns:
        dataframe    
    """
    for i in tqdm(range(data_eng_hap_noun.shape[0])):
        nu = data_eng_hap_noun.loc[i:i, "nouns"].values[0]
        if len(mti_ko_en[mti_ko_en["HS_DESC_EN"].str.contains(nu)]) != 0:
            li = list(mti_ko_en[mti_ko_en["HS_DESC_EN"].str.contains(nu)]["MTICD"])
            data_eng_hap_noun.loc[i:i, "result"].values[0] = li
    data_eng_hap_noun = data_eng_hap_noun[["nouns", "result"]]
    data_eng_hap_noun["result"] = data_eng_hap_noun["result"].apply(lambda x: ', '.join(x))
    data_eng_hap_noun = data_eng_hap_noun[data_eng_hap_noun["result"] != 'x']
    data_eng_hap_noun = data_eng_hap_noun.reset_index(drop=True)
    data_eng_hap_noun = data_eng_hap_noun.astype('str')
    
   
    return data_eng_hap_noun

# data_preprocessing_kor_eleven
def final_result(data, data_result):
    """
    BUYERID를 이용하여 기존 data에 한글 or 영어 처리결과 매핑
    Args:
        data_kor=columns_concat 결과 dataframe
        data_kor_result=data_res 결과 dataframe
    Returns:
        dataframe 
    """
    data = data[["BUYERID", "대표품목명", "대한관심품목명", "hap"]]
    result = pd.merge(data_result, data, how='left', on='BUYERID')
    result = result[["BUYERID", "대표품목명", "대한관심품목명", "MTICD"]]
    
    return result


# final_data_preprocessing_one
def data_ko_en_merge(data, kor_result, eng_result):
    """
    한글처리결과, 영어처리결과를 맨 처음 data 형태에 left join
    Args:
        data=preprocessing 함수의 input data dataframe
        kor_result=한글 매핑 결과 dataframe
        eng_result=영어 매핑 결과 dataframe
    Returns:
        dataframe 
    """
    kor_result_sub = kor_result[["BUYERID", "MTICD"]]
    kor_result_sub.columns = ["BUYERID", "KOR_MTICD"]
    eng_result_sub = eng_result[["BUYERID", "MTICD"]]
    eng_result_sub.columns = ["BUYERID", "ENG_MTICD"]     
    
    final_result = pd.merge(data, kor_result_sub, how='left', on='BUYERID')
    final_result = pd.merge(final_result, eng_result_sub, how='left', on='BUYERID')
    
    return final_result

# final_data_preprocessing_two
def merge_ko_en_mticd(final_result):
    """
    hap = 한글 결과 + 영어 결과
    Args:
        final_result=한글결과+영어결과 dataframe
    Returns:
        dataframe     
    """
    final_result['hap'] = None
    
    for i in tqdm(range(final_result.shape[0])):
        if (final_result.loc[i:i, 'KOR_MTICD'].notnull()[i] == True): # null이 아니면
            if (final_result.loc[i:i, 'ENG_MTICD'].notnull()[i] == True): # null이 아니면
                final_result.loc[i:i, 'hap'].values[0] = final_result.loc[i:i, 'KOR_MTICD'].values[0] + final_result.loc[i:i, 'ENG_MTICD'].values[0]
            
        if (final_result.loc[i:i, 'KOR_MTICD'].isnull()[i] == True):
            if (final_result.loc[i:i, 'ENG_MTICD'].isnull()[i] == True):
                final_result.loc[i:i, 'hap'].values[0] = None
            else:
                final_result.loc[i:i, 'hap'].values[0] = final_result.loc[i:i, 'ENG_MTICD'].values[0]
                
        if (final_result.loc[i:i, 'ENG_MTICD'].isnull()[i] == True):
            if (final_result.loc[i:i, 'KOR_MTICD'].isnull()[i] == True):
                final_result.loc[i:i, 'hap'].values[0] = None
            else:
                final_result.loc[i:i, 'hap'].values[0] = final_result.loc[i:i, 'KOR_MTICD'].values[0]
            
    return final_result

  
    

def buyer_hs4_mti(data, data2):
    """
    hscode 4단위로 MTI코드 매핑하는 함수
    Args:
        data=dataframe
        data2=dataframe
    Returns:
        dataframe

    """    
    
    # 초기 정제작업
    data = data[["BUYERID", "HSCD"]]
    data = data.drop_duplicates()
    data = data.dropna()
    data2 = data2.drop_duplicates()
    data2 = data2.reset_index(drop=True)
    
    # hscd4와 mticd6 매핑
    data2_dict = defaultdict()
    for i in range(data2.shape[0]):
        hs = data2.loc[i:i, "HSCD4"].values[0]
        mti = data2.loc[i:i, "MTICD"].values[0]

        if hs not in data2_dict:
            data2_dict[hs] = [mti]
        else:
            data2_dict[hs].append(mti)
    data2_groupby_hscd4 = pd.DataFrame({"HSCD":list(data2_dict.keys()), "MTICD":list(data2_dict.values())})
    data2_groupby_hscd4["MTICD"] = data2_groupby_hscd4["MTICD"].apply(lambda x: ', '.join(x))
    
    # buyerid별로 mticd6 매핑
    result = pd.merge(data, data2_groupby_hscd4, how='left', on="HSCD")
    result = result.astype(str)
    result = result.groupby("BUYERID")["MTICD"].agg(lambda x: ', '.join(x))
    result = pd.DataFrame(result, columns =["MTICD"])
    result = result.reset_index()
    
    return result    

# final_data_preprocessing_three
def intritem_hs_map(data, hscd4_mticd6):
    """
    앞서 만든 HSCD4자리와 MTICD6자리 매핑 테이블과 매핑하는 함수
    Args:
        data=전처리 결과 dataframe
        data2=hscd4_mticd6 dataframe
    Returns:
        dataframe
    """
    
    hscd4_mticd6.columns = ['BUYERID', "HSCD4_MTICD6"]
    hscd4_mticd6 = hscd4_mticd6.fillna('')
    hscd4_mticd6['HSCD4_MTICD6'] = hscd4_mticd6['HSCD4_MTICD6'].apply(lambda x: x.split(', '))
    data = pd.merge(data, hscd4_mticd6, how='left', on='BUYERID')
    
    return data


# final_data_preprocessing_four
def repl_none(data):
    """
    HSCD4자리와 MTICD6자리 매핑 결과 컬럼이 비어있는 경우 None으로 변경하는 함수
    Args:
        data=dataframe
    Returns:
        dataframe
    """
    for i in tqdm(range(data.shape[0])):
        word = data.loc[i:i, "HSCD4_MTICD6"].values[0]
        if word == list(['']):
            data.loc[i:i, "HSCD4_MTICD6"].values[0] = None
    
    return data


# final_data_preprocessing_five
def blank_hap_repl(data):
    """
    real_final_hap = hap의 값. 하지만 hap이 비어있는 경우 HSCD4_MTICD6의 값으로 대체하는 함수
    Args:
        data=dataframe
    Returns:
        dataframe
    """
    data['real_final_hap'] = 'x'
    for i in tqdm(range(data.shape[0])):
        if (data.loc[i:i, "hap"].isnull()[i] == True):
            data.loc[i:i, "real_final_hap"].values[0] = data.loc[i:i, "HSCD4_MTICD6"].values[0]
        elif (data.loc[i:i, "HSCD4_MTICD6"].values[0] == ''):
            data.loc[i:i, "real_final_hap"].values[0] = data.loc[i:i, "hap"].values[0]
        else:
            data.loc[i:i, "real_final_hap"].values[0] = data.loc[i:i, "hap"].values[0]
            
    return data


# final_data_preprocessing_six
def assign_mticode(data):
    """
    조건별 MTIcode 추출 함수 
    real_final_hap의 mti4자리에 대해 Counter하고, Counter의 결과가 2개 이상일 경우에만 해당하는 mti6자리 출력,
    real_final_hap의 길이가 4이하라면 MTI4_Counting_Result에 real_final_hap의 결과가 들어간다.
    Args:
        data=dataframe
    Returns:
        dataframe    
    """
    data['hscd4_li'] = 'x'
    data['MTI4_Counting_Result'] = 'x' 
    data = data.replace({np.nan:None})
    for i in tqdm(range(data.shape[0])):
        li = []
        if (data.loc[i:i, "real_final_hap"].values[0] == None):
            pass
        elif (data.loc[i:i, "real_final_hap"].values[0] != None):
            if (len(data.loc[i:i, "real_final_hap"].values[0]) <= 4):
                data.loc[i:i, "MTI4_Counting_Result"].values[0] = data.loc[i:i, "real_final_hap"].values[0]
            else:
                four_li = [num[:4] for num in data.loc[i:i, "real_final_hap"].values[0]]
                data.loc[i:i, "hscd4_li"].values[0] = Counter(four_li)
                four_li_li = [i[0] for i in Counter(four_li).items() if i[1] >= 2]

                for num in data.loc[i:i, "real_final_hap"].values[0]:
                    for f in four_li_li:
                        if str(num).startswith(str(f)):
                            li.append(num)
            data.loc[i:i, "MTI4_Counting_Result"].values[0] = li
            
    return data


def preprocessing_intriitem(data, mti_ko_en):
    """
    대표 품목명, 대한관심품목명 컬럼에서 한글과 영어 텍스트 데이터 전처리 함수
    Args:
        data=dataframe
    Returns:
        dataframe

    """     
    
    data_ko = fill_na(data, '')
    
    data_ko['대표품목명_ext_ko'] = data_ko['대표품목명'].progress_apply(lambda x : korean_extract(x))
    data_ko['대한관심품목명_ext_ko'] = data_ko['대한관심품목명'].progress_apply(lambda x : korean_extract(x))
    
    data_ko = df_newline_tab_replace(data_ko, '대표품목명')
    data_ko = df_newline_tab_replace(data_ko, '대한관심품목명')
    
    data_ko = data_ko[(data_ko['대표품목명_ext_ko'].str.strip()!='')|(data_ko['대한관심품목명_ext_ko'].str.strip()!='')] # 두 컬럼 모두 NUll drop

    
#     data_dp = ko_noun_ext(data_dp) 
    morph_file_name = sorted(glob.glob("morph_dict*"), reverse=True)[0]
    with open(morph_file_name, 'rb') as f:
        tagging = pickle.load(f)
        
    data_ko['대표품목명_ext_ko_nouns'] = data_ko['대표품목명_ext_ko'].apply(lambda x : tagging.get(x))
    data_ko['대한관심품목명_ext_ko_nouns'] = data_ko['대한관심품목명_ext_ko'].apply(lambda x : tagging.get(x))
    
    data_ko['대표품목명_ext_ko_nouns'] = data_ko['대표품목명_ext_ko_nouns'].progress_apply(pos_noun)  # extract noun 
    data_ko['대한관심품목명_ext_ko_nouns'] = data_ko['대한관심품목명_ext_ko_nouns'].progress_apply(pos_noun)  # extract noun 
# ****
# **** 실제 서비스 운영시 변경 예정
    
    data_ko = columns_concat(data_ko)
    data_one = empty_list_drop(data_ko)
    data_two = separation_col_values(data_ko, 'hap')
    data_2 = buyerid_nouns(data_one, data_two)
    data_3 = drop_dup_and_2wnoun_ext(data_2)
    data_mti_map = mti_map(data_3, mti_ko_en)
    data_res = buyerid_mticode(data_2, data_mti_map)
    data_res = buyerid_mticode_conc(data_res)
    
    ko_result = final_result(data_ko, data_res)

    # ================================== 영어
    data_en = fill_na(data, '')
    data_en["대표품목명"] = data_en["대표품목명"].apply(lambda x: x.lower().strip())
    data_en["대한관심품목명"] = data_en["대한관심품목명"].apply(lambda x: x.lower().strip())
    
    data_en = ext_eng_drop_stopw(data_en)
    data_en = tokenization_intritem(data_en)
    data_en = merge_and_drop_dup(data_en)
    data_en = hap_len_drop(data_en)
    data_en['hap'] = data_en['hap'].apply(lambda x: [i for i in x if i not in en_stopwords])
    
    data_en = en_noun_ext(data_en)
    data_eng_subone = data_en[['BUYERID', 'hap_lemma']]
    data_eng_subtwo = separation_col_values(data_eng_subone, 'hap_lemma')
    data_eng_hap = buyerid_noun_list(data_eng_subone, data_eng_subtwo)
    data_eng_hap_noun = drop_dup_and_4wnoun_ext(data_eng_hap)
    
    data_eng_hap_noun = noun_mti_map(data_eng_hap_noun, mti_ko_en)
    data_eng_res = buyerid_mticode(data_eng_hap, data_eng_hap_noun)
    data_eng_res = buyerid_mticode_conc(data_eng_res)
    en_result = final_result(data_en, data_eng_res)
    
    # ================================== 한글 영어 병합
    result = data_ko_en_merge(data, ko_result, en_result)
    result = merge_ko_en_mticd(result)
    
    return result  

#=======================================================희서=======================================================                   
def drop_null(data, subset_colname):
    """
    Drop null or blank values 함수
    Args:
        data=dataframe
        subset_colname=str
    Returns:
        dataframe
    """                        
    data = data.dropna(subset=[subset_colname], axis = 0)
    data = data[data[subset_colname].str.strip()!='']
    data = data.reset_index(drop=True)                         
    return data
                         
# data = drop_null(data, '코드품목명_영문')


def korean_extract(x):
    """
    한글 추출 함수
    Args:
        x=string
    Returns:
        string
    """                         
    p = re.compile('[^가-힣\s]*')
    x = p.sub('', x)
    x = x.strip()
    return x
                         
def morpheme_analysis(x):
    """
    형태소 분석 함수
    Args:
        x=string
    Returns:
        list(tuple)
        ex) [(레이저, NNP), (소자, NNP)]
    """
                         
    dics = '사용자정의사전.txt'

    ko_words=[]
    if x.strip()!='':
        ko = Komoran(userdic=dics)
        ko_res_li = ko.pos(x)
        for val in ko_res_li:
            if val[0] not in ko_stopwords:
                ko_words.append(val)
                
    return ko_words                         

def noun_ext(x):
    """
    명사형 단어만 추출['NNG','NNB','NNP']
    1글자 및 '기타' 단어 제외
    Args:
        x=string
    Returns:
        string
    """
                         
    text=''
    if isinstance(x, list):
        for i in x:
            if i[1] in ['NNG','NNB','NNP']:
                if len(text)!=0:
                    if (len(i[0]) > 1)&(i[0]!='기타'):
                        text+= ', '+i[0]
                else:
                    if (len(i[0]) > 1)&(i[0]!='기타'):
                        text+= i[0]

    return text    
                         
def noun_ext2(x): 
    """
    명사형 단어만 추출['NNG','NNB','NNP','NA'] # NA 분석불능단어 태깅 포함
    1글자 및 '기타' 단어 제외
    Args:
        x=string
    Returns:
        string
    """                           
                         
    text=''
    if isinstance(x, list):
        for i in x:
            if i[1] in ['NNG','NNB','NNP','NA']:
                if len(text)!=0:
                    if (len(i[0]) > 1)&(i[0]!='기타'):
                        text+= ', '+i[0]
                else:
                    if (len(i[0]) > 1)&(i[0]!='기타'):
                        text+= i[0]

    return text   
                         
def contains_func(pattern, data, column_name):
    """
    MTI_KO_EN.csv 내 DESC 컬럼에서 패턴에 해당하는 데이터 추출 함수
    Args:
        pattern=DESC내 찾고자하는 문자열의 패턴
        data=dataframe(MTI_KO_EN.csv)
        column_name=DESC 컬럼명
    Returns:
        dataframe

    """                         
    return data[data[column_name].str.contains(pattern)]
                         
def output(input_value1, input_value2, input_value3, data, column_name):
    """
    contains_func 실행해 mticode list 추출 함수
    Args:
        input_value1, input_value2, input_value3= 조회 조건 키워드
        data=dataframe(MTI_KO_EN.csv)
        column_name=DESC 컬럼명
    Returns:
        list : MITCODE list

    """                         
    search_words_list = [input_value1, input_value2, input_value3]
    pattern1 = '^'+''.join(fr'(?=.*{x})' for x in search_words_list if x is not None) # and 조건
    
    result = contains_func(pattern1, data, column_name = column_name) 
                    
    result = result[['MTICD','MTI_NAME']]
    result_mticd = result['MTICD'].tolist()

    return result_mticd                         

# mti_co_li = output_ko(input_value1, input_value2, input_value3, data, 'HS_DESC_KO')  
                         
###  run                          
def mapping_MTICODE_ko(data, mti_ko_en):
    """
    전처리된 텍스트 데이터에서 해당하는 컬럼의 
    단어에 매칭되는 MTI code 찾아 매핑하는 함수
    Args:
        data=dataframe
    Returns:
        dataframe

    """                         
    results = []
    for idx, i in enumerate(tqdm(data['코드품목명_영문_ext_ko_pos_noun'])):

        if isinstance(i, str):
            i_li = i.split(', ')


            if (len(i_li)<= 3)&(i_li[0]!=''):

                input_values = [None, None, None]
                                                  
                for idx in range(len(i_li)):
                    input_values[idx] = i_li[idx]

                input_value1, input_value2, input_value3 = input_values[0], input_values[1], input_values[2]

                mti_co_li = output(input_value1, input_value2, input_value3, mti_ko_en, 'HS_DESC_KO')           


                if len(mti_co_li)==0:  # 매칭 mti code가 없는 경우
                    tok_txt = data['코드품목명_영문_ext_ko_pos_noun_2'].iloc[idx] # NA (분석불능범주) 까지 사용
                    tok_txt_li = tok_txt.split(', ')
                    tok_txt_li = list(map(lambda x: x.strip().replace(' ',''), tok_txt_li))
                         
                    input_value1, input_value2, input_value3 = '|'.join(tok_txt_li), None, None # or 매핑
                    mti_co_li = output(input_value1, input_value2, input_value3, mti_ko_en, 'HS_DESC_KO')
 

            elif (len(i_li)> 3)&(i_li[0]!=''):
                i_li = list(map(lambda x: x.strip().replace(' ',''), i_li))
                         
                input_value1, input_value2, input_value3 = '|'.join(i_li), None, None
                mti_co_li = output(input_value1, input_value2, input_value3, mti_ko_en, 'HS_DESC_KO')

            results.append(mti_co_li)
                         
        else:
            mti_co_li = []
            results.append(mti_co_li) 
                         
    data['MTICD'] = results
                         
    return data

###  run 
def preprocessing_itemname_ko(data):
    """
    ITEMNAME_EN (코드품목명_영문) 컬럼에서 한글에 해당하는 텍스트 데이터 전처리 함수
    Args:
        data=dataframe
    Returns:
        dataframe

    """     

    data['코드품목명_영문_ext_ko'] = data['코드품목명_영문'].progress_apply(lambda x : korean_extract(x))
    
    data = df_newline_tab_replace(data, '코드품목명_영문_ext_ko')
    data = drop_null(data, '코드품목명_영문_ext_ko')
    
#     data['코드품목명_영문_ext_ko_pos'] = data['코드품목명_영문_ext_ko'].progress_apply(morpheme_analysis)  #tagging # 95만건 - 70만건 = 약 25만건에 대해 소요시간 140h , 
    with open('morph_dict_20221207.pkl', 'rb') as f:
        tagging = pickle.load(f)
    data['코드품목명_영문_ext_ko_pos'] = data['코드품목명_영문_ext_ko'].apply(lambda x : tagging.get(x))
# ****
# **** 실제 서비스 운영시 변경 예정
    
    data['코드품목명_영문_ext_ko_pos_noun'] = data['코드품목명_영문_ext_ko_pos'].progress_apply(noun_ext)  # extract noun 
    data['코드품목명_영문_ext_ko_pos_noun_2'] = data['코드품목명_영문_ext_ko_pos'].progress_apply(noun_ext2)  # extract noun 
    # 단일 단어의 경우, 매핑 안되면 tagging 데이터 사용을 위해 코드품목명_영문_ext_ko_komoran_noun_2 사용 ( NA 분석불능단어 포함된 태깅 데이터 사용)
    
    
    data['코드품목명_영문_단어수'] = data['코드품목명_영문_ext_ko'].str.strip().str.split(' ').str.len() # 단일 단어와 복합단어 구분을 위해
    data.loc[data['코드품목명_영문_ext_ko'].str.strip()=="기타", '코드품목명_영문_단어수'] = 0  
    data.loc[data['코드품목명_영문_단어수']==1,'코드품목명_영문_ext_ko_pos_noun'] = data['코드품목명_영문_ext_ko']  # 단일단어의 경우 tagging 데이터 x , 한글추출 데이터 사용
    
    return data


def buyerid_grouping(data, key_column_name, agg_column_name):
    """
    key_column_name 기준으로 그룹화하여 agg_column_name을 합치는 함수
    Args:
        data=dataframe
        key_column_name=BUYERID에 해당하는 컬럼명 
        agg_column_name=MTICD에 해당하는 컬럼명
    Returns:
        dataframe
        
        ex)
            BUYERID	MTICD
        0	3000052	[011230, 011510, 011615, 012510, 013290, 01410...
        1	3000081	[613310, 613320, 613330, 613210, 613220, 613230]
        2	3000140	[]
        3	3000410	[511640, 733100, 733200, 814610, 814790, 950900]
        4	3000656	[011440, 011490, 013110, 013900, 015500, 01620...
    
    """ 
    return data.groupby([key_column_name], as_index=False).agg({agg_column_name:sum})


#=======================================================현주======================================================= 



def english_extract(x):
    """
    영어 추출 함수
    Args:
        x=string
    Returns:
        string
    """  
    p = re.compile('[^a-zA-Z\s]+')
    x = p.sub('', x)
    x = x.strip()
    x = x.lower()
    return x


def tokenization(data):
    """
    토큰화 후 명사(명사, 복수명사, 고유명사, 일반명사) 추출 함수
    Args:
        x=dataframe
    Returns:
        dataframe
    """    
    data['코드품목명_영문_ext_en'] = data['코드품목명_영문_ext_en'].progress_apply(lambda x : nltk.word_tokenize(x))
    data['코드품목명_영문_ext_en_pos_noun'] = data['코드품목명_영문_ext_en'].progress_apply(lambda x: [word for word, pos in nltk.pos_tag(x) if pos in ['NN','NNS','NNP','NNPS']])
    data['코드품목명_영문_ext_en_pos_noun']  = data['코드품목명_영문_ext_en_pos_noun'].progress_apply(lambda x: [nltk.WordNetLemmatizer().lemmatize(i) for i in x])
    return data

# stop
def drop_stopword_en(data):
    """
    불용어 제거 함수
    Args:
        x=dataframe
    Returns:
        dataframe
    """     
    data['코드품목명_영문_ext_en_pos_noun']  = data['코드품목명_영문_ext_en_pos_noun'].progress_apply(lambda x: [w for w in x if w not in en_stopwords])

    return data

# cleansing
def drop_blank_and_1w(data):
    """
    빈 리스트와 1글자 미만 단어 제거 함수
    Args:
        x=dataframe
    Returns:
        dataframe
    """     
    text_lii = []
    for j in data['코드품목명_영문_ext_en_pos_noun']:
        if len(j)!=0 :
            li = []
            for i in j:
                if len(i)>1 :
                    li.append(i)
            text_lii.append(li)
        else : 
            text_lii.append(np.nan)
            
    data['코드품목명_영문_ext_en_pos_noun'] = text_lii
    data = data.dropna(subset=['코드품목명_영문_ext_en_pos_noun'])
    return data 


def peeloff_list(data):
    """
    리스트 벗기기 
    Args:
        x=dataframe
    Returns:
        dataframe
    """   
    data['코드품목명_영문_ext_en_pos_noun'] = data['코드품목명_영문_ext_en_pos_noun'].progress_apply(lambda x :', '.join(x))
    data['코드품목명_영문_ext_en'] =data['코드품목명_영문_ext_en'].progress_apply(lambda x :', '.join(x))
    return data 

def preprocessing_itemname_en(data):
    """
    ITEMNAME_EN (코드품목명_영문) 컬럼에서 영어에 해당하는 텍스트 데이터 전처리 함수
    Args:
        data=dataframe
    Returns:
        dataframe

    """     

    data['코드품목명_영문_ext_en'] = data['코드품목명_영문'].progress_apply(lambda x : english_extract(x))
    
    data = df_newline_tab_replace(data, '코드품목명_영문_ext_en')
    data = drop_null(data, '코드품목명_영문_ext_en')
    
    data = tokenization(data)
    data = drop_stopword_en(data)
    
    data = drop_blank_and_1w(data)
    data = peeloff_list(data)
    
    return data


# MTI
def mapping_MTICODE_en(data, mti_ko_en):
    """
    전처리된 텍스트 데이터에서 해당하는 컬럼의 
    단어에 매칭되는 MTI code 찾아 매핑하는 함수
    Args:
        data=dataframe
    Returns:
        dataframe

    """     
    results = []
    for idx, i in enumerate(tqdm(data['코드품목명_영문_ext_en_pos_noun'])):

        i_li = i.split(', ')


        if (len(i_li)<= 3)&(i_li[0]!=''):

            input_values = [None, None, None]
            for idx in range(len(i_li)):
                input_values[idx] = i_li[idx]

            input_value1, input_value2, input_value3 = input_values[0], input_values[1], input_values[2]
       
            mti_co_li = output(input_value1, input_value2, input_value3, mti_ko_en, 'HS_DESC_EN')
        
        elif (len(i_li)> 3)&(i_li[0]!=''):
            i_li = list(map(lambda x: x.strip().replace(' ',''), i_li))
            input_value1, input_value2, input_value3 = '|'.join(i_li), None, None

            mti_co_li = output(input_value1, input_value2, input_value3, mti_ko_en, 'HS_DESC_EN')

        results.append(mti_co_li)
    data['MTICD'] = results
    
    return data

