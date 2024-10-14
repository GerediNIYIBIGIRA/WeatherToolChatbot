
from dash import Dash, html, dcc, ctx
from dash import Dash, dcc, html, Input, Output, State
from dash import Dash, dash_table

tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold',
    'width': '600px',
    'font-size':20
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px',
    'width': '600px',
    'font-size':20
}

col_style = {'display':'grid', 'grid-auto-flow': 'row', 'font-size':20, 'margin-top':'20px', 'width':'90px'}
row_style = {'display':'grid', 'grid-auto-flow': 'column', 'font-size':20}

import plotly.express as px
import pandas as pd
import requests

df = []

# external JavaScript files
external_scripts = [
    'http://localhost:8000/copilot/index.js'
]

app = Dash(__name__, external_scripts=external_scripts)

app.layout = html.Div(children=[
    html.H1(children='Weather Infromation Generator'),
    dcc.Tabs([
        dcc.Tab(label="Input form", style=tab_style, selected_style=tab_selected_style, children=[

            html.H2(children='Weather Information Form', style={'font-family': 'sans-serif'}),

            # fields for course add/drop
            html.Div([
                html.Div([
                    html.Label(['Country Name'], style={'font-family': 'sans-serif'}),
                    html.Div(dcc.Input(id='fieldA', type='text', style={"font-family": 'sans-serif', 'font-size': 20}))
                ], style=col_style),
                                   
                html.Div([
                    html.Label(['City Name'], style={'font-family': 'sans-serif'}),
                    html.Div(dcc.Input(id='fieldB', type='text', style={"font-family": 'sans-serif', 'font-size': 20}))
                ], style=col_style),

                html.Div([
                    html.Label(['Weather Information Results'], style={'font-family': 'sans-serif'}),
                    html.Div(dcc.Input(id='fieldC', type='text', style={"font-family": 'sans-serif', 'font-size': 20}))
                ], style=col_style),

                html.Div([
                    html.Button('Submit', id='add-val', style={"width":"90px", "height":"30px", "font-family": 'sans-serif', 'font-size': 20}),
                    html.Div(id='submit-response', children='Click to submit')
                ], style=col_style)

            ], style=col_style)

        ])
    ])
])

if __name__ == '__main__':
    app.run_server(debug=False)

