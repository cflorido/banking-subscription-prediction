from dash import dcc, html, Input, Output, Dash
import dash_bootstrap_components as dbc
import psycopg2
import pandas as pd
import plotly.express as px
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go


app = Dash(__name__)

#Cargar modelos:
# Cargar el modelo default
#model3 = keras.models.load_model('models/modeloproy_03.keras')
model1 = keras.models.load_model('models/modeloproy_01.keras')
#model2 = keras.models.load_model('models/modeloproy_02.keras')
#model4 = keras.models.load_model('models/modeloproy_04.keras')
#model5 = keras.models.load_model('models/modeloproy_05.keras')
#model6 = keras.models.load_model('models/modeloproy_06.keras')
#model7 = keras.models.load_model('models/modeloproy_07.keras')
#model8 = keras.models.load_model('models/modeloproy_08.keras')
#model9 = keras.models.load_model('models/modeloproy_09.keras')

# Conectar a la base de datos
engine = psycopg2.connect(
    dbname="prod",
    user="postgres",
    password="manolo21",
    host="database-2.cw2fwyuh7xdb.us-east-1.rds.amazonaws.com",
    port="5432"
)

# Cargar modelo
model1 = keras.models.load_model('models/modeloproy_01.keras')

# Datos para la gráfica
resultados = {
    0.1: {"Ingreso esperado": 98213.46, "Accuracy": 0.30},
    0.2: {"Ingreso esperado": 59827.85, "Accuracy": 0.58},
    0.3: {"Ingreso esperado": 52050.41, "Accuracy": 0.63},
    0.4: {"Ingreso esperado": 42358.89, "Accuracy": 0.68},
    0.5: {"Ingreso esperado": 19292.33, "Accuracy": 0.76},
    0.6: {"Ingreso esperado": 9156.47, "Accuracy": 0.81},
    0.7: {"Ingreso esperado": -13594.80, "Accuracy": 0.84},
    0.8: {"Ingreso esperado": -27413.94, "Accuracy": 0.86},
    0.9: {"Ingreso esperado": -27624.71, "Accuracy": 0.86},
}
# Extrayendo valores para las listas
indices = list(resultados.keys())
ingresos = [resultados[key]["Ingreso esperado"] for key in resultados]
accuracy = [resultados[key]["Accuracy"] for key in resultados]

# Estilos generales
app.layout = html.Div(
    style={
        'backgroundColor': '#f4f8fb',
        'fontFamily': 'Arial, sans-serif',
        'padding': '30px'
    },
    children=[
        html.H1(
            "Análisis de Datos de Clientes",
            style={
                'textAlign': 'center',
                'color': '#00254a',
                'fontSize': '36px',
                'fontWeight': 'bold',
                'marginBottom': '30px'
            }
        ),
        
        # Umbral Card
        dbc.Card(
            style={
                'padding': '25px',
                'marginBottom': '20px',
                'borderRadius': '15px',
                'backgroundColor': '#ffffff',
                'boxShadow': '0px 4px 8px rgba(0, 0, 0, 0.1)'
            },
            children=[
                html.Label(
                    "Ingrese el Umbral",
                    style={'fontSize': '22px', 'fontWeight': 'bold', 'color': '#003366'}
                ),
                dcc.Slider(
                    min=0.1, max=0.9, step=0.1, value=0.3,
                    marks={i: str(i) for i in [0.1, 0.3, 0.5, 0.7, 0.9]},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ]
        ),
        
        # Perfil del Cliente Card
        dbc.Card(
            style={
                'padding': '25px',
                'marginBottom': '20px',
                'borderRadius': '15px',
                'backgroundColor': '#ffffff',
                'boxShadow': '0px 4px 8px rgba(0, 0, 0, 0.1)'
            },
            children=[
                html.P(
                    "Seleccione el perfil del cliente que desea analizar",
                    style={
                        'fontSize': '18px',
                        'fontWeight': 'bold',
                        'marginBottom': '20px'
                    }
                ),
                
                html.Label("Edad (18 a 99):", style={'fontSize': '18px', 'fontWeight': 'bold'}),
                dcc.Input(type='number', value=21, min=18, max=99, style={'width': '100%', 'height': '30px', 'marginBottom': '30px'}),
                
                html.Label("Ocupación:", style={'fontSize': '18px', 'fontWeight': 'bold'}),
                dcc.Dropdown(
                    options=[
                        {'label': 'Desconocido', 'value': 'unknown'},
                        {'label': 'Desempleado', 'value': 'unemployed'},
                        {'label': 'Estudiante', 'value': 'student'},
                        {'label': 'Gerente', 'value': 'management'},
                        {'label': 'Técnico', 'value': 'technician'},
                        {'label': 'Empresario', 'value': 'entrepreneur'},
                        {'label': 'Obrero', 'value': 'blue-collar'},
                        {'label': 'Jubilado', 'value': 'retired'},
                        {'label': 'Administrativo', 'value': 'admin.'},
                        {'label': 'Servicios', 'value': 'services'},
                        {'label': 'Autónomo', 'value': 'self-employed'},
                        {'label': 'Ama de casa', 'value': 'housemaid'}
                    ],
                    style={'width': '100%', 'marginBottom': '20px'}
                ),
                
                html.Label("Estado Civil:", style={'fontSize': '18px', 'fontWeight': 'bold'}),
                dcc.Dropdown(
                    options=[
                        {'label': 'Soltero', 'value': 'single'},
                        {'label': 'Casado', 'value': 'married'},
                        {'label': 'Divorciado', 'value': 'divorced'}
                    ],
                    style={'width': '100%', 'marginBottom': '15px'}
                ),
                
                html.Label("Nivel Educativo:", style={'fontSize': '18px', 'fontWeight': 'bold'}),
                dcc.Dropdown(
                    options=[
                        {'label': 'Desconocido', 'value': 'unknown'},
                        {'label': 'Secundaria', 'value': 'high_school'}
                    ],
                    style={'width': '100%'}
                ),
            ]
        ),
        
        # Estadísticas Card
        dbc.Card(
            style={
                'padding': '25px',
                'marginBottom': '20px',
                'borderRadius': '15px',
                'backgroundColor': '#ffffff',
                'boxShadow': '0px 4px 8px rgba(0, 0, 0, 0.1)'
            },
            children=[
                html.H4(
                    "Estadísticas",
                    style={'fontSize': '22px', 'fontWeight': 'bold', 'color': '#003366'}
                ),
                html.P(
                    "Número de coincidencias exactas:\nNúmero de coincidencias (Sí):\nNúmero de coincidencias (No):\nNúmero total de coincidencias:",
                    style={'fontSize': '18px'}
                ),
                html.H4(
                    "Predicción:",
                    style={'fontSize': '22px', 'fontWeight': 'bold', 'color': '#003366'}
                ),
                html.P("Predicción (Sí):\nPredicción (No):", style={'fontSize': '18px'}),
            ]
        ),
        
        html.Br(),

        # Gráficos
        html.Div(
            [
                html.H4(
                    "Gráficos de Coincidencias",
                    style={'color': '#00254a', 'textAlign': 'center', 'fontSize': '22px'}
                ),
                dcc.Graph(id='pie-chart-coincidence', config={'displayModeBar': False}),
                dcc.Graph(id='pie-chart-figLH', config={'displayModeBar': False})
            ],
            style={
                'padding': '25px',
                'border': '2px solid #003366',
                'borderRadius': '15px',
                'backgroundColor': '#E8F4FF',
                'boxShadow': '0px 4px 8px rgba(0, 0, 0, 0.1)'
            }
        ),
        html.Br(),
        html.Div(
            [
                html.H1(children='Análisis de Ingreso Esperado y Accuracy'),

                dcc.Graph(
                    id='ingreso-accuracy-graph',
                    figure={
                        'data': [
                            go.Scatter(
                                x=indices,
                                y=ingresos,
                                mode='lines+markers',
                                name='Ingreso Esperado',
                                line=dict(color='blue')
                            ),
                            go.Scatter(
                                x=indices,
                                y=accuracy,
                                mode='lines+markers',
                                name='Accuracy',
                                yaxis="y2",
                                line=dict(color='green')
                            )
                        ],
                        'layout': go.Layout(
                            title='Relación entre Ingreso Esperado y Accuracy',
                            xaxis={'title': 'Índice'},
                            yaxis={'title': 'Ingreso Esperado', 'side': 'left'},
                            yaxis2={
                                'title': 'Accuracy',
                                'overlaying': 'y',
                                'side': 'right'
                            },
                            legend={'x': 0.1, 'y': 1.1},
                        )
                    }
                )
            ]
        )
    ]
)
      



@app.callback(
    Output('output-coincidenceE', 'children'),
    Output('output-coincidenceY', 'children'),
    Output('output-coincidenceN', 'children'),
    Output('output-coincidence', 'children'),
    Output('pie-chart-coincidence', 'figure'),  # Output para el gráfico de torta
    Output('pie-chart-figLH', 'figure'),  # Output para el gráfico de torta
    Output("resultados-graph", "figure"),
    Input('age', 'value'),
    Input('job', 'value'),
    Input('marital', 'value'),
    Input('education', 'value'),
    Input('balance', 'value'),
    Input('housing', 'value'),
    Input('loan', 'value'),
    Input('contact', 'value'),
    Input('day', 'value'),
    Input('month', 'value'),
    Input('duration', 'value'),
    Input('campaign', 'value'),
    Input('pdays', 'value'),
    Input('previous', 'value'),
    Input('poutcome', 'value')
)

def update_output_div(age, job, marital, education, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome):
    cursor = engine.cursor()

    # Asignar valores predeterminados en caso de que alguno de los inputs sea None
    age = int(age) if age is not None else 18
    job = job if job is not None else 'unknown'
    marital = marital if marital is not None else 'single'
    education = education if education is not None else 'unknown'
    balance = balance if balance is not None else 0
    housing = housing if housing is not None else 'no'
    loan = loan if loan is not None else 'no'
    contact = contact if contact is not None else 'unknown'
    day = day if day is not None else 1
    month = month if month is not None else 'jan'
    duration = duration if duration is not None else 0
    campaign = campaign if campaign is not None else 0
    pdays = pdays if pdays is not None else -1
    previous = previous if previous is not None else 0
    poutcome = poutcome if poutcome is not None else 'unknown'

    # Consulta para ver cuántos datos de mi BD coinciden con las características dadas y tienen y = 'yes'
    query_coincidence_exact = f"""
    SELECT COUNT(*)
    FROM bank_client_data
    WHERE age = '{int(age)}' AND
        job = '{job}' AND
        marital = '{marital}' AND
        education = '{education}' AND
        balance = '{int(balance)}' AND
        housing = '{housing}' AND
        loan = '{loan}' AND
        contact = '{contact}' AND
        day = '{int(day)}' AND
        month = '{month}' AND
        duration = '{duration}' AND
        campaign = '{campaign}' AND
        pdays = '{pdays}' AND
        previous = '{previous}' AND
        poutcome = '{poutcome}';
    """
    cursor.execute(query_coincidence_exact)
    result_coincidenceEXACT = cursor.fetchone()
    coincidenceE = result_coincidenceEXACT[0] if result_coincidenceEXACT[0] is not None else 0

    # Consulta para ver cuántos datos de mi BD coinciden con las características dadas y tienen y = 'yes'
    query_coincidence_Y = f"""
    SELECT COUNT(*)
    FROM bank_client_data
    WHERE (age BETWEEN '{int(age) - 10}' AND '{int(age) + 10}') AND
        job = '{job}' AND
        marital = '{marital}' AND
        education = '{education}' AND
        housing = '{housing}' AND
        loan = '{loan}' AND
        poutcome = '{poutcome}' AND
        y = 'yes';
    """
    cursor.execute(query_coincidence_Y)
    result_coincidenceY = cursor.fetchone()
    coincidenceY = result_coincidenceY[0] if result_coincidenceY[0] is not None else 0

    # Consulta para ver cuántos datos de mi BD coinciden con las características dadas y tienen y = 'no'
    query_coinN = f"""
    SELECT COUNT(*)
    FROM bank_client_data
    WHERE (age BETWEEN '{int(age) - 10}' AND '{int(age) + 10}') AND
        job = '{job}' AND
        marital = '{marital}' AND
        education = '{education}' AND
        housing = '{housing}' AND
        loan = '{loan}' AND
        poutcome = '{poutcome}' AND
        y = 'no';
    """
    cursor.execute(query_coinN)
    result_coincidenceN = cursor.fetchone()
    coincidenceN = result_coincidenceN[0] if result_coincidenceN[0] is not None else 0

    # Consulta para ver cuántos datos de mi BD coinciden con las características dadas y tienen y = 'no'
    query_loan_housingY = f"""
    SELECT COUNT(*)
    FROM bank_client_data
    WHERE housing = '{housing}' AND
        loan = '{loan}' AND
        y = 'yes';
    """
    cursor.execute(query_loan_housingY)
    result_loan_housingY = cursor.fetchone()
    loand_housign_Y = result_loan_housingY[0] if result_loan_housingY[0] is not None else 0


    # Consulta para ver cuántos datos de mi BD coinciden con las características dadas y tienen y = 'no'
    query_loan_housingN = f"""
    SELECT COUNT(*)
    FROM bank_client_data
    WHERE housing = '{housing}' AND
        loan = '{loan}' AND
        y = 'no';
    """
    cursor.execute(query_loan_housingN)
    result_loan_housingN = cursor.fetchone()
    loand_housign_N = result_loan_housingN[0] if result_loan_housingN[0] is not None else 0


    # Número total de coincidencias
    coincidence = coincidenceY + coincidenceN

    # Gráfico de torta
    pie_data_Coincidences = {
        'Resultado': ['Sí', 'No'],
        'Coincidencias': [coincidenceY, coincidenceN]
    }
    pie_df = pd.DataFrame(pie_data_Coincidences)
    fig1 = px.pie(pie_df, names='Resultado', values='Coincidencias', title="Distribución de Coincidencias (Sí/No)")


    # Gráfico de torta
    pie_data_loan_Housing = {
        'Resultado': ['Sí', 'No'],
        'LoansHousing': [loand_housign_Y, loand_housign_N]
    }
    pie_df_loan_housing = pd.DataFrame(pie_data_loan_Housing)
    figLH = px.pie(pie_df_loan_housing, names='Resultado', values='LoansHousing', title="Distribución de (Sí/No) de acuerdo a Loan y Housing")

    fig_resultados = px.line(x=[0.1, 0.2, 0.3], y=[1000, 2000, 1500])
    

    return coincidenceE, coincidenceY, coincidenceN, coincidence, fig1, figLH, fig_resultados

# Callback for updating client prediction
@app.callback(
    Output(component_id='output-cdtY', component_property='children'),
    Output(component_id='output-cdtN', component_property='children'),
    [
        Input('umbral', 'value'),
        Input('age', 'value'),
        Input('job', 'value'),
        Input('marital', 'value'),
        Input('education', 'value'),
        Input('balance', 'value'),
        Input('contact', 'value'),
        Input('day', 'value'),
        Input('month', 'value'),
        Input('duration', 'value'),
        Input('campaign', 'value'),
        Input('pdays', 'value'),
        Input('previous', 'value'),
        Input('poutcome', 'value')
    ]
)

def update_prediccionCliente(umbral, age, job, marital, education, balance, contact, day, month, duration, campaign, pdays, previous, poutcome):
    umbral = float(umbral) if umbral is not None else 0.3

    dict = {'contact': {'cellular': 0, 'telephone': 1, 'unknown': 2},
            'education': {'primary': 0, 'secondary': 1, 'tertiary': 2, 'unknown': 3},
            'poutcome': {'failure': 0, 'other': 1, 'success': 2, 'unknown': 3},
            'marital': {'divorced': 0, 'married': 1, 'single': 2},
            'job': {'admin.': 0, 'blue-collar': 1, 'entrepreneur': 2, 'housemaid': 3, 'management': 4, 'retired': 5, 'self-employed': 6, 'services': 7, 'student': 8, 'technician': 9, 'unemployed': 10, 'unknown': 11},
            'month': {'apr': 0, 'aug': 1, 'dec': 2, 'feb': 3, 'jan': 4, 'jul': 5, 'jun': 6, 'mar': 7, 'may': 8, 'nov': 9, 'oct': 10, 'sep': 11}}

    # Convertir los valores de los inputs a los valores que el modelo espera
    job = dict['job'][job]
    marital = dict['marital'][marital]
    education = dict['education'][education]
    contact = dict['contact'][contact]
    month = dict['month'][month]
    poutcome = dict['poutcome'][poutcome]

    # ['contact', 'education', 'poutcome', 'marital', 'job', 'month', 'age', 'balance', 'campaign', 'pdays', 'previous', 'day']
    #input = [[contact, education, poutcome, marital, job, month, int(age), float(balance), int(campaign), int(pdays), int(previous), int(duration), int(day)]]
    input = [np.array([contact]), np.array([education]), np.array([poutcome]), np.array([marital]), np.array([job]),
              np.array([month]), np.array([age]), np.array([balance]), np.array([campaign]), np.array([pdays]),
              np.array([previous]), np.array([duration]), np.array([day])]

    print(input)

    ypred = model1.predict(input)
    #Seleccion del modelo:
    if umbral == 0.1:
        ypred = model1.predict(input)
    #elif umbral == 0.2:
        #ypred = model2.predict(input)
    #elif umbral == 0.4:
       # ypred = model4.predict(input)
    #elif umbral == 0.5:
       # ypred = model5.predict(input)
    #elif umbral == 0.6:
       # ypred = model6.predict(input)
    #elif umbral == 0.7:
        #ypred = model7.predict(input)
    #elif umbral == 0.8:
       # ypred = model8.predict(input)
   # elif umbral == 0.9:
        #ypred = model9.predict(input)

    # Predict using the loaded model
    return '{0:.3f}'.format(ypred[0][0]), '{0:.3f}'.format(1-ypred[0][0])



if __name__ == '__main__':
    app.run_server(debug=True)
