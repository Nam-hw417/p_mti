from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import dash_table
import plotly.express as px
from utils import *
from config import *
import time

 

initial_active_cell = {'row':0, 'column':0, 'column_id':'MTICD','row_id':0}

# MTI
# data = pd.read_csv(os.path.join(base_path, "MTI_KO.csv"), dtype={'MTICD':'string'})
data = pd.read_csv(os.path.join(base_path, "MTI_KO_EN.csv"), dtype={'MTICD':'string','HS_DESC_KO':'string','HS_DESC_EN':'string'})

# HS 
hs_data = pd.read_csv(os.path.join(base_path, "HSCD_KO_EN.csv"), dtype={'MTICD':'string','HSCD':'string'})


app = Dash(suppress_callback_exceptions=True)


def inital_data_load():
    
    result = data
    
    return data_table.DataTable(data=result.to_dict('records'),
                                columns=[{'name':i, 'id':i} for i in result.columns],
                                page_size=20,
                                style_cell={'textAlign':'left'},
                                id = 'tbl')
                               

app.layout = html.Div(children=[
    html.Hr(),
    html.H1(children='MTI/HS CODE Return (Korean)', style={'color':'#004172',
                                             'display':'outline-block',
                                             'margin':'10px',
                                             'padding':'10px',
                                             'border':'3px solid',
                                             'border-color':'#d6e4ea'}),
    html.Hr(style={'margin':'10px'}),
    html.Div(className='row', children=[
        html.Div([
            html.Label('검색조건', style={'vertical-align':'middle'}),
        ], style={'width':'10%','height':'50%','display':'inline-block','textAlign':'center'}),
        html.Div([
            html.Label('MTI code ', style={'textAlign':'center','padding':'10px'}),
            dcc.Input(id='code',
                placeholder='code',
                type='text',
                style={'marginLeft':'10px'}
            )
        ], style={'width':'15%','display':'inline-block'}),
        html.Div([
            html.Label('검색어 1 ', style={'textAlign':'center','padding':'10px'}),
            dcc.Input(id='input_word1',
                placeholder='키워드',
                type='text',
                debounce=True,
                style={'marginLeft':'10px'}
            )
        ], style={'width':'15%','display':'inline-block'}),
        html.Div([
            html.Label('검색어 2 ', style={'textAlign':'center','padding':'10px'}),
            dcc.Input(id='input_word2',
                placeholder='키워드',
                type='text',
                style={'marginLeft':'10px'}
            )
        ], style={'width':'15%','display':'inline-block'}),
        html.Div([
            html.Label('검색어 3 ', style={'textAlign':'center','padding':'10px'}),
            dcc.Input(id='input_word3',
                placeholder='키워드',
                type='text',
                style={'marginLeft':'10px'}
            )
        ], style={'width':'15%','display':'inline-block'}),
        html.Button('검색', id='click', n_clicks=0),
    ], style={'width':'98%','display':'inline-block','vertical-align':'middle'}),
    html.Hr(style={'margin':'10px'}),
    html.Div([
        html.Div([
            html.H4(children='MTI CODE', style={'font-weight':'border',
                                               'vertical-align':'middle',
                                               'padding-left':'10px'}),
            html.Div(id='output_code', style={'margin':'5px', 'float':'left'})
        ], style={'width':'30%', 'display':'inline-block'}),
        html.Div([
            html.H4(children='HS CODE', style={'font-weight':'border',
                                               'vertical-align':'middle',
                                               'padding-left':'10px'}),
            html.Div(id='output_code2', style={'margin':'5px', 'float':'left'})
        ], style={'width':'70%', 'display':'inline-block'})
    ], style={'padding-left':'30px','width':'98%', 'display':'inline-block','textAlign':'left'}),
    html.Hr(style={'margin':'30px'}),
    html.Div([
        html.Div([
            html.Label('검색 키워드', style={'textAlign':'center', 'padding':'10px'}),
            html.Div(id='output_keyword', style={'marginLeft':'10px', 'border':'2px solid'})
        ], style={'width':'15%','display':'inline-block'}),
        html.Div([
            html.Label('선택한 MTI CODE', style={'textAlign':'center', 'padding':'10px'}),
            html.Div(id='output_mticode', style={'marginLeft':'10px', 'border':'2px solid'})
        ], style={'width':'15%','display':'inline-block'}),
        html.Button('선택', id='submit', n_clicks=0, loading_state={'is_loadng':False}),
        html.Div(id='DB-submit', children='Enter values and press 선택', style={'width':'15%','display':'inline-block'})
    ], style={'width':'98%','display':'inline-block','vertical-align':'middle'}),
    html.Hr(style={'margin':'30px'}),
#     html.Div([
#         html.Div([
#             html.Label('선택한 MTI CODE', style={'textAlign':'center', 'padding':'10px'}),
#             html.Div(id='output_mticode2', style={'marginLeft':'10px', 'border':'2px solid'})
#         ], style={'width':'90%','display':'inline-block'}),
#         html.Button('기업검색', id='submit2', n_clicks=0, loading_state={'is_loadng':False}),
#         html.Div(id='search_corp_buyer', style={'margin':'5px', 'float':'right'})
#     ], style={'width':'98%','display':'inline-block','vertical-align':'middle'})
    html.Div([
        html.Div([
            html.Label('선택한 MTI CODE', style={'textAlign':'center', 'padding':'10px'}),
            html.Div(id='output_mticode2', style={'marginLeft':'10px', 'border':'2px solid'})
        ], style={'width':'80%','display':'inline-block'}),
        html.Button('기업검색', id='submit2', n_clicks=0, loading_state={'is_loadng':False})
    ], style={'width':'30%','display':'inline-block','vertical-align':'middle'}),
    html.H4(children='MTI code & 국내/해외 기업', style={'font-weight':'border',
                                               'vertical-align':'middle',
                                               'padding-left':'10px'}),
    html.Div([
        html.Div(id='search_corp_buyer', style={'margin':'5px', 'float':'left'})
    ], style={'width':'98%','display':'inline-block','vertical-align':'middle'})
    
], style={'background-color':'#d6e4ea'})

@app.callback(
    Output(component_id='output_code', component_property='children'),
    Input('click','n_clicks'),
    State(component_id='input_word1', component_property='value'),
    State(component_id='input_word2', component_property='value'),
    State(component_id='input_word3', component_property='value')
)
def mti_output_div(n_click, input_value1, input_value2=None, input_value3=None):
    

    try:
        result = output(input_value1, input_value2, input_value3, data)
            
    except:
        result = pd.DataFrame(columns=['MTICD','MTI_NAME'])
    
    return dash_table.DataTable(id='tbl', data=result.to_dict('records'), 
                                columns = [{'name': i, 'id': i} for i in result.columns], 
                                page_size=15,
                                is_focused = True,
                                style_cell={'textAlign':'left', 'cursor':'pointer', 'whiteSpace' : 'normal'})





@app.callback(
    Output('output_code2', 'children'),
    Input('tbl','data'),
    Input('tbl','active_cell'),
    Input('tbl','page_current'),
    Input('tbl','page_size')
)
def hs_output_div(df_dict, active_cell, cur_page, page_size):
    
    if active_cell is None:
        return None
    
    else:
        cur_page = 0 if cur_page is None else cur_page
        row = cur_page * page_size + active_cell['row']
        col = active_cell['column_id']
        active_data = df_dict[row][col]
        
        if col == 'MTI_NAME':

            result = output_hs_nm(hs_data, active_data)
            
        else:
            result = output_hs_cd(hs_data, active_data, df_dict)
        
        result = result[['HSCD','HS_MTI_DESC','HS_MTI_DESC_EN']]
        
        ### MTI 코드 내 클릭시 영어 HSCD 테이블의 컬럼을 가지고 와야함.
        # 수정 필요
        
        return dash_table.DataTable(data=result.to_dict('records'), 
                                columns=[{'name': i, 'id': i} for i in result.columns], 
                                page_size=15,
                                is_focused=True,
                                style_cell={'textAlign':'left', 'cursor':'pointer', 'whiteSpace' : 'normal'},
                                id='tbl2')

@app.callback(
    Output('output_keyword', 'children'),
    Input('click','n_clicks'),
    State(component_id='input_word1', component_property='value'),
    State(component_id='input_word2', component_property='value'),
    State(component_id='input_word3', component_property='value')
)
def keyword(n_click, input_word1, input_word2=None, input_word3=None):
    
    if input_word2!=None:
        if input_word3!=None:
            text = input_word1+input_word2+input_word3
        else:
            text = input_word1+input_word2
    else:
        text = input_word1
    
    return text

@app.callback(
    Output('output_mticode', 'children'),
    Input('tbl','data'),
    Input('tbl','active_cell'),
    Input('tbl','page_current'),
    Input('tbl','page_size')
)
def mti_code(df_dict, active_cell, cur_page, page_size):
    if active_cell is None:
        return None
    
    else:
        cur_page = 0 if cur_page is None else cur_page
        row = cur_page * page_size + active_cell['row']
        col = active_cell['column_id']
        active_data = df_dict[row][col]
    
        if col =='MTICD':
            return active_data
        else:
            return None
        
@app.callback(
    Output('DB-submit', 'children'),
    Input('submit','n_clicks'),
    Input('tbl','data'),
    Input('tbl','active_cell'),
    Input('tbl','page_current'),
    Input('tbl','page_size'),
    State(component_id='input_word1', component_property='value'),
    State(component_id='input_word2', component_property='value'),
    State(component_id='input_word3', component_property='value')
)        
def DB_insert(n_clicks, df_dict, active_cell, cur_page, page_size, input_word1, input_word2=None, input_word3=None):
    

    if n_clicks:
        if active_cell is not None:

            cur_page = 0 if cur_page is None else cur_page
            row = cur_page * page_size + active_cell['row']
            col = active_cell['column_id']
            active_data = df_dict[row][col]
        
        if input_word2!=None:
            if input_word3!=None:
                text = input_word1+input_word2+input_word3
            else:
                text = input_word1+input_word2
        else:
            text = input_word1
        
        date = time.strftime('%Y-%m-%d', time.localtime(time.time()))
        
        sql = f'''
            INSERT INTO BPUSER.TMP_KEYWORD_MTI_TB(KEYWORD, MTICD, DATETIME)
            VALUES('{text}','{active_data}','{date}');
            '''

        update_db(sql)

        answer = f'{text}, {active_data} 정보가 저장되었습니다.'

    else:
        answer = 'Enter values and press 선택'
        
    return answer


@app.callback(
    Output('output_mticode2', 'children'),
    Input('tbl','data'),
    Input('tbl','active_cell'),
    Input('tbl','page_current'),
    Input('tbl','page_size')
)
def mti_code2(df_dict, active_cell, cur_page, page_size):
    if active_cell is None:
        return None
    
    else:
        cur_page = 0 if cur_page is None else cur_page
        row = cur_page * page_size + active_cell['row']
        col = active_cell['column_id']
        active_data = df_dict[row][col]
    
        if col =='MTICD':
            return active_data
        else:
            return None


@app.callback(
    Output('search_corp_buyer', 'children'),
    Input('submit2','n_clicks'),
    Input('tbl','data'),
    Input('tbl','active_cell'),
    Input('tbl','page_current'),
    Input('tbl','page_size')
)        
def search_corp_buyer_func(n_clicks, df_dict, active_cell, cur_page, page_size):
    

    if n_clicks:
        if active_cell is not None:

            cur_page = 0 if cur_page is None else cur_page
            row = cur_page * page_size + active_cell['row']
            col = active_cell['column_id']
            active_data = df_dict[row][col]

         # 임시
        sql = f"""SELECT A.MTICD, B.BSNO_DECRYPT, A.INTRITEM  
            FROM ( SELECT MTICD, HSCD, BSNO, INTRITEM  
                    FROM SCRM.TB_CM_EXIMHIST_NEWHSCD ) A 
            INNER JOIN SCRM.TB_CM_EXIMCORP_TOTAL B   -- 코트라 기업 통합 테이블
            ON (A.BSNO=B.BSNO)
            WHERE A.MTICD = '{active_data}'; 
            """
        
                # 해외 정보 DB 적재 후 실행
#             f"""SELECT DISTINCT C.MTICD, C.BSNO_DECRYPT, D.BUYERID, C.INTRITEM, D.INTE_PRD
#             FROM (SELECT A.MTICD, B.BSNO_DECRYPT, A.INTRITEM  
#                     FROM ( SELECT MTICD, HSCD, BSNO, INTRITEM  
#                             FROM SCRM.TB_CM_EXIMHIST_NEWHSCD ) A 
#                     INNER JOIN SCRM.TB_CM_EXIMCORP_TOTAL B   -- 코트라 기업 통합 테이블
#                     ON (A.BSNO=B.BSNO) ) C
#             INNER JOIN BPUSER.TMP_MTI_CODE D   -- 코트라 기업 통합 테이블
#             ON C.MTICD=D.MTICD
#             WHERE C.MTICD = '229000';"""
        
        
        data = collect(sql)

        
    return dash_table.DataTable(data=data.to_dict('records'), 
                                columns=[{'name': i, 'id': i} for i in data.columns], 
                                page_size=15,
                                is_focused=True,
                                style_cell={'textAlign':'left', 'cursor':'pointer', 'whiteSpace' : 'normal'},
                                id='tbl3')
        
# @app.callback(
#     Output('output_keyword','children'),
#     Input('')

# )


if __name__ == '__main__':
    app.run_server(debug=True, host='192.168.70.230', port=18885)
