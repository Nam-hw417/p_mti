from utils_total import *



item_sql = """
SELECT A.BUYERID,
        A.ITEMNAME_EN AS 코드품목명_영문
FROM SCRM.TB_CM_BUYER_ITEM A -- 바이어관심품목
LEFT JOIN (SELECT *
    FROM SCRM.TB_CM_BUYER -- 바이어정보
    WHERE COSTATE ='01' -- 활성기업
    ) B
ON A.BUYERID = B.BUYERID
WHERE TRIM(ITEMNAME_EN) IS NOT NULL
AND ROWNUM <= 986866
;
"""
# 986866


intri_sql = """
SELECT DISTINCT BUYERID, REPITEMNAME AS 대표품목명, INTRITEMDESC AS 대한관심품목명
FROM SCRM.TB_CM_BUYER A 
LEFT JOIN SCRM.TB_CO_ECATALOG B 
    ON (A.INTRITEMCD=B.EACODE)
LEFT JOIN BPUSER.TB_ENTP_CMDLT_MAPNG_INFO C 
    ON (B.LVL1_NM_KO=C.ENTP_CMDLT_NAME AND C.USE_YN='Y')
WHERE TRIM(INTRITEMCD) IS NOT NULL 
AND A.COSTATE ='01'
AND ROWNUM <= 435229
;
"""
# 435229


hs1_sql = """
SELECT BUYERID , C.HSCD -- 바이어정보
FROM SCRM.TB_CM_BUYER A -- 바이어정보
LEFT JOIN SCRM.TB_CO_ECATALOG B -- 코트라품목분류코드
    ON (A.INTRITEMCD=B.EACODE)
LEFT JOIN BPUSER.TB_ENTP_CMDLT_MAPNG_INFO C -- 코트라품목분류 - HS코드 4자리 매핑
    ON (B.LVL1_NM_KO=C.ENTP_CMDLT_NAME AND C.USE_YN='Y')
WHERE TRIM(INTRITEMCD) IS NOT NULL 
AND A.COSTATE ='01'
AND ROWNUM <= 435229
; -- 활성기업
"""
# 435229


hs2_sql = """
SELECT substring(HSCD, 1, 4) AS HSCD4, MTICD 
FROM BPUSER.TB_BP_KITA_HSCD_MAPNG_MTICD
;
"""
# 







def preprocess():
    
    mti_ko_en = pd.read_csv(os.path.join(base_path, "MTI_KO_EN.csv"), dtype={'MTICD':'string','HS_DESC_KO':'string','HS_DESC_EN':'string'})
    mti_ko_en = drop_null(mti_ko_en, 'HS_DESC_EN')
    
    
    # 코드성품목명
    data = collect(item_sql)

    data = data.drop_duplicates(subset=['BUYERID','코드품목명_영문'])
    data = drop_null(data, '코드품목명_영문')
    
    data_ko = preprocessing_itemname_ko(data) # 한글
    data_en = preprocessing_itemname_en(data) # 영어
    
    data_ko = mapping_MTICODE_ko(data_ko, mti_ko_en)# 한글
    data_en = mapping_MTICODE_en(data_en, mti_ko_en)# 영어

    data_item_ko = buyerid_grouping(data_ko, 'BUYERID', 'MTICD')
    data_item_en = buyerid_grouping(data_en, 'BUYERID', 'MTICD')
    
    data_item_ko = data_item_ko.rename(columns={'MTICD':'코드성품목_ko_mti'})
    data_item_en = data_item_en.rename(columns={'MTICD':'코드성품목_en_mti'})
    
    tot_item_result = data_item_ko.merge(data_item_en, how='outer', on='BUYERID')  # 모두 병합 : 596,865건
    
    tot_item_result['코드성품목_ko_mti'] = tot_item_result['코드성품목_ko_mti'].fillna('').apply(list)
    tot_item_result['코드성품목_en_mti'] = tot_item_result['코드성품목_en_mti'].fillna('').apply(list)
    
    tot_item_result['코드성품목_mti'] = tot_item_result['코드성품목_ko_mti'] + tot_item_result['코드성품목_en_mti']
    

    
    
    # 대한/대표 품목명
    data2 = collect(intri_sql)

    dt = preprocessing_intriitem(data2, mti_ko_en)

    
    # HS
    data3 = collect(hs1_sql)
    data3 = data3.astype('str')
    
    data4 = collect(hs2_sql)
    data4 = data4.astype('str')
    
    result = buyer_hs4_mti(data3, data4)
    
    dt = intritem_hs_map(dt, result)
    dt = repl_none(dt)
    dt = blank_hap_repl(dt)
    dt = assign_mticode(dt)
    
    dt = dt[['BUYERID','대표품목명','대한관심품목명','real_final_hap','MTI4_Counting_Result']]

    # data_hs_hap_2 = result.merge(data_hs_hap_2, how='inner', on='BUYERID')  # 샘플 5만건에 대해서..
    dt = dt.merge(tot_item_result, how='outer', on='BUYERID')
    dt = dt.rename(columns={'real_final_hap':'대한,대표|HS4_MTI', 'MTI4_Counting_Result':'real_final_MTI4_Counting_Result'})
    
    
    final_result = dt[['BUYERID','대표품목명','대한관심품목명','real_final_MTI4_Counting_Result','코드성품목_mti']]
    

    final_result['real_final_MTI4_Counting_Result'] = final_result['real_final_MTI4_Counting_Result'].fillna('').apply(list)
    final_result['코드성품목_mti'] = final_result['코드성품목_mti'].fillna('').apply(list)
    
    final_result['intersection_mti'] = final_result.apply(lambda x : list(set(x['real_final_MTI4_Counting_Result']).intersection(set(x['코드성품목_mti']))) , axis=1)
    
    
    final_result.loc[final_result['intersection_mti'].str.len()!=0, 'results'] = final_result['intersection_mti']
    final_result.loc[final_result['intersection_mti'].str.len()==0, 'results'] = final_result['real_final_MTI4_Counting_Result']
    
    
    final_result.loc[final_result['results'].str.len()==0, 'results'] = final_result['코드성품목_mti']
    
    final_result['results_uniq'] = final_result['results'].apply(lambda x: list(set(x)))
    
    final_result = final_result.rename(columns={'results_uniq':'MTICD'})

    final_result = final_result[['BUYERID','대표품목명','대한관심품목명','MTICD']]
    

    
    
    output = final_preprocessing(final_result)
    add = data[['BUYERID','코드품목명_영문']]
    add = add.drop_duplicates()
    add = add.groupby(['BUYERID'], as_index=False).agg({'코드품목명_영문':'/ '.join})
    
    output = output.merge(add, how='left', on='BUYERID')
    output['대표품목명'] = output['대표품목명'].fillna('')
    output['대한관심품목명'] = output['대표품목명'].fillna('')
    output['코드품목명_영문'] = output['대표품목명'].fillna('')
    
    output['대표품목명'] = output['대표품목명'].astype(str)
    output['대한관심품목명'] = output['대한관심품목명'].astype(str)
    output['코드품목명_영문'] = output['코드품목명_영문'].astype(str)
    
    
    output['품목명'] = output['대표품목명']+' & '+output['대한관심품목명']+' & '+output['코드품목명_영문']
    output = output[['MTICD','BUYERID','품목명','대표품목명','대한관심품목명','코드품목명_영문']]
    
    today = time.strftime('%Y%m%d')
    output.to_csv(f'{today}_해외바이어_MTICODE_결과.csv', index=False, encoding='utf-8-sig')
    
if __name__=='__main__':
    
    start = datetime.now()
    print("시작시간:", time.strftime('%Y-%m-%d %H:%M:%S'))
    
    preprocess()
    
    end = datetime.now()
    print("시간:", end - start)