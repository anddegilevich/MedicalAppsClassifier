import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

#-----------------------------------------------------------------------------------------------------------------------
#Интерфейс и графики

RatingS = pd.read_pickle('RatingS.pkl')

AndroidS = pd.read_pickle('AndroidS.pkl')

SizeS = pd.read_pickle('SizeS.pkl')

df_top10S = pd.read_pickle('df_top10S.pkl')
hf='Health&Care'
m='Medical'

figRating = px.box(RatingS, x="Month", y="Rating",color='Category')

figAndroid = px.box(AndroidS.loc[AndroidS['Android'] < 13], x="Month", y="Android",color='Category')

figSize = px.box(SizeS, x="Month", y="Size",color='Category')

figTop10S = px.line(df_top10S.query('Category in @hf'), x="Month", y="Rating", color='Name')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='Диплом'),

    html.Div([
        html.P('Тема Диплома:'),
        html.P('Разработка программного комплекса для оценки качества и безопасности мобильных приложений медицинского назначения')
    ]),

    html.H2(children='Динамика распеределения рейтинга приложений:'),

    dcc.Dropdown(
        id='DropdownRating',
        options=[
            {'label': 'Health&Hitness', 'value': 'Health&Care'},
            {'label': 'Medical', 'value': 'Medical'}
        ],
        multi=True,
        value=["Health&Care", "Medical"],
        placeholder="Select category",
        clearable=False,
    ),

    dcc.Graph(id="GraphRating",figure=figRating),

html.H2(children='Динамика распеределения рейтинга Android:'),

    dcc.Dropdown(
        id='DropdownAndroid',
        options=[
            {'label': 'Health&Hitness', 'value': 'Health&Care'},
            {'label': 'Medical', 'value': 'Medical'}
        ],
        multi=True,
        value=["Health&Care", "Medical"],
        placeholder="Select category",
        clearable=False,
    ),

    dcc.Graph(id="GraphAndroid",figure=figAndroid),

html.H2(children='Динамика распеределения размера приложений:'),

    dcc.Dropdown(
        id='DropdownSize',
        options=[
            {'label': 'Health&Hitness', 'value': 'Health&Care'},
            {'label': 'Medical', 'value': 'Medical'}
        ],
        multi=True,
        value=["Health&Care", "Medical"],
        placeholder="Select category",
        clearable=False,
    ),

    dcc.Graph(id="GraphSize",figure=figSize),

    html.H2(children='Динамика рейтинга 10 самых скачиваемых приложений:'),

    dcc.Dropdown(
        id='DropdownTop10S',
        options=[
            {'label': 'Health&Hitness', 'value': 'Health&Care'},
            {'label': 'Medical', 'value': 'Medical'}
        ],
        multi=False,
        value="Health&Care",
        placeholder="Select category",
        clearable=False,
    ),

    dcc.Graph(id="GraphTop10S",figure=figTop10S),

])

#Callback DropdownRating
@app.callback(Output('GraphRating', 'figure'), [Input('DropdownRating', 'value')])
def update_graph_rating(selected_dropdown_value):
    categories=selected_dropdown_value
    df=RatingS.query("Category in @categories")
    figRating = px.box(df, x="Month", y="Rating", color='Category')
    return figRating

#Callback DropdownAndroid
@app.callback(Output('GraphAndroid', 'figure'), [Input('DropdownAndroid', 'value')])
def update_graph_android(selected_dropdown_value):
    categories=selected_dropdown_value
    df=AndroidS.query("Category in @categories")
    figAndroid = px.box(df.loc[df['Android'] < 13], x="Month", y="Android", color='Category')
    return figAndroid

#Callback DropdownSize
@app.callback(Output('GraphSize', 'figure'), [Input('DropdownSize', 'value')])
def update_graph_size(selected_dropdown_value):
    categories=selected_dropdown_value
    df=SizeS.query("Category in @categories")
    figSize = px.box(df, x="Month", y="Size", color='Category')
    return figSize

#Callback DropdownTop10S
@app.callback(Output('GraphTop10S', 'figure'), [Input('DropdownTop10S', 'value')])
def update_graph_top10(selected_dropdown_value):
    categories=selected_dropdown_value
    df=df_top10S.query("Category in @categories")
    figTop10S = px.line(df, x="Month", y="Rating", color='Name')
    return figTop10S

if __name__ == '__main__':
    app.run_server(debug=True)