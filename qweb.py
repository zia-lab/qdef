#           _.                  __                  ,_.
#        _=Tg2\s.            ,v/LsZe_             _=Tg/+s.
#     _z5,mmms/_X+s.      ,g/|_mmmK,2Ze_       _zT,gmmm|_X+s
#   iML!M!@__m@*T!,8s    ZK.*[*b__WAf|-2W.   iMLY@!@__m@*T-,8s
#   / 'VeL=/*T-!GW@@@i  ]` ~+K)cVfL=Xm@@@b   P 'Ve7=/*Tv!gW@@@.
#  ]` _. '\c7Lm@@@@@@W  P ,_  ~=2TgW@@@@@@i ]  _. '\c7LW@@@@@@W
#  [ dPVW.  Z@@@@@@@@@id  @~Ys  iM@@@@@@@@W,[ ]PVW.  Z@@@@@@@@@i
#                         web-app for QDEF
# d  !Wg@  i`M@@@@@@@@W[  Msd[  /!@@@@@@@@@Z  !W_@  ]`M@@@@@@@@W
# 'Vc. W!  P '@@@@@@@@@@m_ i@  ]` M@@@@@@@@@Ws. W[  P !@@@@@@@@@@m_
#    '\c. ]   Y@@@@@@@@@@@@m_ ,[  '@@@@@@@@@@@@Ws. d   Y@@@@@@@@@@@@m_
#       '\!   'M@@@@@@@@@@@@@@Z    V@@@@@@@@@@@@@@W!   'M@@@@@@@@@@@@@@i
#               'V@@@@@@@@@Af`       ~M@@@@@@@@@*~       '*@@@@@@@@@Af`
#                  ~*@@@@f`            'VM@@@*~             ~*@@@Af`
#                     ~~                  '~`                  ~~

from flask import Flask
from dash import Dash, callback_context
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from sympy import latex, symbols
import base64
from qdef import *

MATHJAX_CDN = '''
https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/
MathJax.js?config=TeX-MML-AM_CHTML'''

external_scripts = [
                    {'type': 'text/javascript',
                     'id': 'MathJax-script',
                     'src': MATHJAX_CDN,
                     },
                    ]

server = Flask(__name__)
app = Dash('Symmetry ahoy!',
            server=server,
            external_scripts=external_scripts,
            external_stylesheets=[dbc.themes.BOOTSTRAP])

app.config['suppress_callback_exceptions']=True
app.title='Groups ahoy!'

banner_fname = './images/qdef-banner.png'
encoded_png = base64.b64encode(open(banner_fname, 'rb').read())

title_fname = './images/title.png'
encoded_title = base64.b64encode(open(title_fname, 'rb').read())

pretty_group_labels = []
for label in list(sorted(CPG.AllGroupLabels)):
    clear_label = label
    if '_' in label:
        label_parts = label.split('_')
        label = '%s_{%s}' % tuple(label_parts)
    pretty_group_labels.append({'label':'\(%s\)' % label,
    'value': clear_label,
    'btn_label': 'btn-'+clear_label.replace('{','(').replace('}',')')})

btn_labels = [gl['btn_label'] for gl in pretty_group_labels]

btn_style = {'width':'50px','margin':'1px'}

app.layout = html.Div(id='power',
    children = ([html.Img(src='data:image/png;base64,{}'.format(encoded_png.decode()),style={'width': '400px','marginBottom':'5px','marginTop':'20px'}),
    html.Br(),
    html.Img(src='data:image/png;base64,{}'.format(encoded_title.decode()),style={'width': '400px','marginBottom':'10px','marginTop':'5px'}),
    html.Br(),
    dbc.ButtonGroup([dbc.Button(gl['label'], style=btn_style, id= gl['btn_label']) for gl in pretty_group_labels[:8]]),
    html.Br(),
    dbc.ButtonGroup([dbc.Button(gl['label'], style=btn_style, id= gl['btn_label']) for gl in pretty_group_labels[8:16]]),
    html.Br(),
    dbc.ButtonGroup([dbc.Button(gl['label'], style=btn_style, id= gl['btn_label']) for gl in pretty_group_labels[16:24]]),
    html.Br(),
    dbc.ButtonGroup([dbc.Button(gl['label'], style=btn_style, id= gl['btn_label']) for gl in pretty_group_labels[24:]]),
    html.Br(),
    dcc.Markdown('-----'),
    # character table
    dbc.Button(
            "Character Table",
            id="collapse-button-char-table",
            className="mb-3",
            color="primary",
        ),
    dbc.Collapse(
            html.Div(
                html.Table(id='character-table',
                style={'margin-left':'auto',
                    'margin-right':'auto',
                    'border':'1px solid black'}
                ),
            ),
            id="collapse-char-table",
        ),
    dcc.Markdown('-----'),

    # group operations
    dbc.Button(
            "Group Operations (Euler Angles)",
            id="collapse-button-group-ops",
            className="mb-3",
            color="primary",
        ),
    dbc.Collapse(
            html.Div(
                html.Table(id='ops-table',style={'margin-left':'auto',
                'margin-right':'auto',
                'border':'1px solid black'}),
            ),
            id="collapse-grp-ops",
        ),
    dcc.Markdown('-----'),

    # direct product table
    dbc.Button(
            "Direct Product Table",
            id="collapse-button-ptable",
            className="mb-3",
            color="primary",
        ),
    dbc.Collapse(
            html.Div(
                html.Table(id='ptable',
                style={'margin-left':'auto',
                    'margin-right':'auto',
                    'border':'1px solid black'}
                ),
            ),
            id="collapse-ptable",
        ),
    dcc.Markdown('-----')
    ]
    ),
    style = {'text-align':'center'})

@app.callback(
    [Output(gl['btn_label'], "color") for gl in pretty_group_labels],
    [Input(gl['btn_label'], "n_clicks") for gl in pretty_group_labels]
)
def on_button_click(*btns):
    ctx = callback_context
    if not ctx.triggered:
        button_id = 'btn-O'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    colors = ["primary" if gl['btn_label'] == button_id else "secondary" for gl in pretty_group_labels]
    return colors

@app.callback(
    Output("collapse-ptable", "is_open"),
    [Input("collapse-button-ptable", "n_clicks")],
    [State("collapse-ptable", "is_open")],
)
def toggle_collapse_ptable(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("collapse-grp-ops", "is_open"),
    [Input("collapse-button-group-ops", "n_clicks")],
    [State("collapse-grp-ops", "is_open")],
)
def toggle_collapse_grp_ops(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("collapse-char-table", "is_open"),
    [Input("collapse-button-char-table", "n_clicks")],
    [State("collapse-char-table", "is_open")],
)
def toggle_collapse_char_table(n, is_open):
    if n:
        return not is_open
    return not is_open

def make_dash_table(alist, headers, leaders):
    ''' Return an HTML table for a given list of lists
    headers and leaders'''
    cell_style = {'margin-left':'auto',
    'margin-right':'auto',
    'border':'1px solid black',
    'text-align':'center',
    'border-collapse': 'collapse'}
    table = []
    table.append(html.Thead([html.Td('')]+[html.Td('\(%s\)' % (h)) for h in headers],style=cell_style))
    # table.append(html.Tr([html.Td(header) for header in alist]))
    for row, row_leader in zip(alist, leaders):
        html_row = [html.Td('\(%s\)' % row_leader)]
        for i in range(len(row)):
            html_row.append(html.Td(['\(%s\)' % latex(row[i])],style=cell_style))
        table.append(html.Tr(html_row))
    return table


# update the character table
@app.callback(
    Output('character-table', 'children'),
    [Input(gl['btn_label'], "n_clicks") for gl in pretty_group_labels])
def update_char_table(*btns):
    ctx = callback_context
    if not ctx.triggered:
        button_id = 'btn-O'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    value = pretty_group_labels[btn_labels.index(button_id)]['value']
    grup = CPG.Groups[CPG.AllGroupLabels.index(value)]
    char_table = grup.CharacterTable
    grup_classes = grup.Classes
    grup_irreps =  grup.IrrReps
    grup_ops = grup.Elements
    grup_op_params = grup.ParameterTable
    grup_op_euler_angs_and_det = [g[:-1] for g in grup_op_params]
    grup_op_rot_axis = [g[-1] for g in grup_op_params]
    return make_dash_table(char_table, grup_classes, grup_irreps)

# update the group operations table
@app.callback(
    Output('ops-table', 'children'),
    [Input(gl['btn_label'], "n_clicks") for gl in pretty_group_labels])
def update_ops_table(*btns):
    ctx = callback_context
    if not ctx.triggered:
        button_id = 'btn-O'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    value = pretty_group_labels[btn_labels.index(button_id)]['value']
    grup = CPG.Groups[CPG.AllGroupLabels.index(value)]
    char_table = grup.CharacterTable
    grup_classes = grup.Classes
    grup_irreps =  grup.IrrReps
    grup_ops = grup.Elements
    grup_op_params = grup.ParameterTable
    grup_op_euler_angs_and_det = [g[:-1] for g in grup_op_params]
    grup_op_rot_axis = [g[-1] for g in grup_op_params]
    return make_dash_table(grup_op_euler_angs_and_det, [r"\alpha",r'\beta',r'\gamma',r'\textrm{det}',r'\phi'], grup_ops)

# update the direct product table
@app.callback(
    Output('ptable', 'children'),
    [Input(gl['btn_label'], "n_clicks") for gl in pretty_group_labels])
def update_product_table(*btns):
    ctx = callback_context
    if not ctx.triggered:
        button_id = 'btn-O'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    value = pretty_group_labels[btn_labels.index(button_id)]['value']
    grup = CPG.Groups[CPG.AllGroupLabels.index(value)]
    grup_irreps =  grup.IrrReps
    ptable = CPG.direct_product_table(value)
    pretty_table = ptable.list_parse()
    return make_dash_table(pretty_table, grup_irreps, grup_irreps)

if __name__ == '__main__':
    app.run_server(host= '0.0.0.0', port=8000, debug=True)
