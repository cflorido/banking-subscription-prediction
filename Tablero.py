from dash import dcc, html, Input, Output, State, Dash
import psycopg2
import pandas as pd
import plotly.express as px
import tensorflow as tf
from tensorflow import keras
import numpy as np
import psycopg2

app = Dash(__name__)

# Conectar a la base de datos
engine = psycopg2.connect(
    dbname="prod",
    user="postgres",
    password="manolo21",
    host="database-2.cw2fwyuh7xdb.us-east-1.rds.amazonaws.com",
    port="5432"
)

model = keras.models.load_model('models/modeloproy_03.keras')


app.layout = html.Div(
    [   html.H1("Análisis de datos de clientes"),

        #Seleccione el modelo a usar:
        html.H3("Ingrese el umbral que desea utilizar de acuerdo a la gráfica anterior"),
        html.Div(["Umbral: ",
                  dcc.Slider(id ='umbral', min=0.1, max=0.9, step=0.1, value=0.3,
                            marks={i/10: str(i/10) for i in range(1, 10)}
                    )]),
        html.H6("Seleccione el perfil del cliente que quiere analizar"),
        html.Div(["Edad (18 a 99) ",
            dcc.Input(
                id='age', type='number', min=18, max=99, step=1, value=21
            )
        ]),
        html.Div(["Ocupación: ",
                  dcc.Dropdown(id='job', value='unknown',
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
                                 ])]),
        html.Div(["Estado civil: ",
                  dcc.Dropdown(id='marital', value='single',
                               options=[
                                   {'label': 'Soltero', 'value': 'single'},
                                   {'label': 'Casado', 'value': 'married'},
                                   {'label': 'Divorciado', 'value': 'divorced'}
                               ])]),
        html.Div(["Nivel educativo: ",
                  dcc.Dropdown(id='education', value='unknown',
                               options=[
                                    {'label': 'Desconocido', 'value': 'unknown'},
                                    {'label': 'Primario', 'value': 'primary'},
                                    {'label': 'Secundario', 'value': 'secondary'},
                                    {'label': 'Terciario', 'value': 'tertiary'}
                               ])]),
        html.Div(["Saldo promedio anual (euros): ",
            dcc.Input(
                id='balance', type='number', min=0, max=10000000, step=1, value=0
            )
        ]),
        html.Div(["Tiene un préstamo de vivienda: ",
                    dcc.Dropdown(id='housing', value='no',
                                 options=[
                                        {'label': 'Sí', 'value': 'yes'},
                                        {'label': 'No', 'value': 'no'}
                                 ])]),
        html.Div(["Tiene un préstamo personal: ",
                dcc.Dropdown(id='loan', value='no',
                                options=[
                                    {'label': 'Sí', 'value': 'yes'},
                                    {'label': 'No', 'value': 'no'}
                                ])]),
        html.Div(["Contacto: ",
                    dcc.Dropdown(id='contact', value='unknown',
                                 options=[
                                     {'label': 'Desconocido', 'value': 'unknown'},
                                     {'label': 'Celular', 'value': 'cellular'},
                                     {'label': 'Teléfono', 'value': 'telephone'}
                                ])]),
        html.Div(["Día del mes del último contacto (1-31): ",
                    dcc.Input(id='day', type='number', min=1, max=31, step=1, value=1
            )
        ]),
        html.Div(["Mes del último contacto: ",
                    dcc.Dropdown(id='month', value='jan',
                                options=[
                                    {'label': 'Enero', 'value': 'jan'},
                                    {'label': 'Febrero', 'value': 'feb'},
                                    {'label': 'Marzo', 'value': 'mar'},
                                    {'label': 'Abril', 'value': 'apr'},
                                    {'label': 'Mayo', 'value': 'may'},
                                    {'label': 'Junio', 'value': 'jun'},
                                    {'label': 'Julio', 'value': 'jul'},
                                    {'label': 'Agosto', 'value': 'aug'},
                                    {'label': 'Septiembre', 'value': 'sep'},
                                    {'label': 'Octubre', 'value': 'oct'},
                                    {'label': 'Noviembre', 'value': 'nov'},
                                    {'label': 'Diciembre', 'value': 'dec'}
                                ])]),
        html.Div(["Duración de la última llamada (segundos): ",
                    dcc.Input(id='duration', type='number', min=0, max=10000, step=1, value=0
            )
        ]),
        html.Div(["Número de contactos realizados durante esta campaña: ",
                    dcc.Input(id='campaign', type='number', min=0, max=100, step=1, value=0
            )
        ]),
        html.Div(["Número de días que pasaron desde el último contacto de una campaña anterior (Si no fue contactado, ingrese -1): ",
                    dcc.Input(id='pdays', type='number', min=-1, max=1000, step=1, value=-1)]),
        html.Div(["Número de contactos realizados antes de esta campaña: ",
                    dcc.Input(id='previous', type='number', min=0, max=100, step=1, value=0)]),
        html.Div(["Resultado de la campaña anterior: ",
                    dcc.Dropdown(id='poutcome', value='unknown',
                                 options=[
                                    {'label': 'Desconocido', 'value': 'unknown'},
                                    {'label': 'Otro', 'value': 'other'},
                                    {'label': 'Fracaso', 'value': 'failure'},
                                    {'label': 'Éxito', 'value': 'success'}
                                ])]),
        html.Br(),
        html.H4("Estadísticas:"),
        html.Br(),
        html.Div(["Número de coincidencias exactas:", html.Div(id='output-coincidenceE')]),
        html.Div(["Número de coincidencias (Sí):", html.Div(id='output-coincidenceY')]),
        html.Div(["Número de coincidencias (No):", html.Div(id='output-coincidenceN')]),
        html.Div(["Número total de coincidencias:", html.Div(id='output-coincidence')]),
        # Gráfico de torta para las coincidencias
        dcc.Graph(id='pie-chart-coincidence'),
        # Gráfico de torta para las coincidencias
        dcc.Graph(id='pie-chart-figLH')
    ]
)

@app.callback(
    Output('output-coincidenceE', 'children'),
    Output('output-coincidenceY', 'children'),
    Output('output-coincidenceN', 'children'),
    Output('output-coincidence', 'children'),
    Output('pie-chart-coincidence', 'figure'),  # Output para el gráfico de torta
    Output('pie-chart-figLH', 'figure'),  # Output para el gráfico de torta
    Input('umbral', 'value'),
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

def update_output_div(umbral, age, job, marital, education, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome):
    cursor = engine.cursor()
    umbral = float(umbral) if umbral is not None else 0.3

    # Cargar el modelo default
    #Seleccion del modelo:
    if umbral == 0.1:
        model = keras.models.load_model('models/modeloproy_01.keras')
    elif umbral == 0.2:
        model = keras.models.load_model('models/modeloproy_02.keras')
    elif umbral == 0.4:
        model = keras.models.load_model('models/modeloproy_04.keras')
    elif umbral == 0.5:
        model = keras.models.load_model('models/modeloproy_05.keras')
    elif umbral == 0.6:
        model = keras.models.load_model('models/modeloproy_06.keras')
    elif umbral == 0.7:
        model = keras.models.load_model('models/modeloproy_07.keras')
    elif umbral == 0.8:
        model = keras.models.load_model('models/modeloproy_08.keras')
    elif umbral == 0.9:
        model = keras.models.load_model('models/modeloproy_09.keras')

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


    return coincidenceE, coincidenceY, coincidenceN, coincidence, fig1, figLH

if __name__ == '__main__':
    app.run_server(debug=True)
