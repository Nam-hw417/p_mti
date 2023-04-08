from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import dash_table
import plotly.express as px
from utils import *
from config import *

 

initial_active_cell = {'row':0, 'column':0, 'column_id':'MTICD','row_id':0}

# MTI
# data = pd.read_csv(os.path.join(base_path, "MTI_KO.csv"), dtype={'MTICD':'string'})
data = pd.read_csv(os.path.join(base_path, "MTI_KO_EN.csv"), dtype={'MTICD':'string','HS_DESC_KO':'string','HS_DESC_EN':'string'})

# HS 
hs_data = pd.read_csv(os.path.join(base_path, "HSCD_KO_EN.csv"), dtype={'MTICD':'string','HSCD':'string'})


app = Dash(suppress_callback_exceptions=True)


def inital_data_load():
    
    reult = data
    
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
        ], style={'width':'50%', 'display':'inline-block'}),
        html.Div([
            html.H4(children='HS CODE', style={'font-weight':'border',
                                               'vertical-align':'middle',
                                               'padding-left':'10px'}),
            html.Div(id='output_code2', style={'margin':'5px', 'float':'left'})
        ], style={'width':'50%', 'display':'inline-block'})
    ], style={'padding-left':'30px','width':'98%', 'display':'inline-block','textAlign':'left'})
    
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
        result = output(input_value1, input_value2, input_value2, data)
            
    except:
        result = pd.DataFrame(columns=['MTICD','MTI_NAME'])
    
    return dash_table.DataTable(id='tbl', data=result.to_dict('records'), 
                                columns = [{'name': i, 'id': i} for i in result.columns], 
                                page_size=20,
                                is_focused = True,
                                style_cell={'textAlign':'left', 'cursor':'pointer'})

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
        
        result = result[['HSCD','HS_MTI_DESC']]
        
        ### MTI 코드 내 클릭시 영어 HSCD 테이블의 컬럼을 가지고 와야함.
        # 수정 필요
        
        return dash_table.DataTable(data=result.to_dict('records'), 
                                columns=[{'name': i, 'id': i} for i in result.columns], 
                                page_size=20,
                                is_focused=True,
                                style_cell={'textAlign':'left', 'cursor':'pointer'},
                                id='tbl2')

if __name__ == '__main__':
    app.run_server(debug=True, host='192.168.70.230', port=18885)
