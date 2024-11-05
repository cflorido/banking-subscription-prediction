from dash import dcc, html, Input, Output, State, Dash
import psycopg2
import pandas as pd
import plotly.express as px
import tensorflow as tf
from tensorflow import keras
import numpy as np
import psycopg2
import plotly.graph_objects as go

app = Dash(__name__)

# Conectar a la base de datos
engine = psycopg2.connect(
    dbname="prod",
    user="postgres",
    password="manolo21",
    host="database-2.cw2fwyuh7xdb.us-east-1.rds.amazonaws.com",
    port="5432"
)

#Cargar modelos:
# Cargar el modelo default
model3 = keras.models.load_model('models/modeloproy_03.keras')
model1 = keras.models.load_model('models/modeloproy_01.keras')
model2 = keras.models.load_model('models/modeloproy_02.keras')
model4 = keras.models.load_model('models/modeloproy_04.keras')
model5 = keras.models.load_model('models/modeloproy_05.keras')
model6 = keras.models.load_model('models/modeloproy_06.keras')
model7 = keras.models.load_model('models/modeloproy_07.keras')
model8 = keras.models.load_model('models/modeloproy_08.keras')
model9 = keras.models.load_model('models/modeloproy_09.keras')


app.layout = html.Div(
    [   html.H1("Análisis de datos de clientes"),
        html.Div([
            dcc.Graph(id='comparison-graph'),
            dcc.Graph(id='threshold-table'),
            dcc.Graph(id='prices-graph'),
        ], style={'display': 'flex', 'flex-direction': 'column', 'gap': '20px'}),

        #Seleccione el modelo a usar:
        html.Div(style={'margin': '20px 0', 'padding': '20px', 'border': '1px solid #ccc', 'border-radius': '10px', 'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)'}, children=[
            html.H3("Ingrese el umbral que desea utilizar de acuerdo a la gráfica anterior"),
            html.Div(["Umbral: ",
                      dcc.Slider(id='umbral', min=0.1, max=0.9, step=0.1, value=0.3,
                                 marks={i/10: str(i/10) for i in range(1, 10)}
                      )]),
        ]),

        html.Div(style={'margin': '20px 0', 'padding': '20px', 'border': '1px solid #ccc', 'border-radius': '10px', 'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)'}, children=[
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
        ]),

        # Tarjeta para estadísticas
        html.Div(style={'margin': '20px 0', 'padding': '20px', 'border': '1px solid #ccc', 'border-radius': '10px', 'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)', 'text-align': 'center'}, children=[
            html.H4("Estadísticas:"),
            html.Div(["Número de coincidencias exactas:", html.Div(id='output-coincidenceE')]),
            html.Div(["Número de coincidencias (Sí):", html.Div(id='output-coincidenceY')]),
            html.Div(["Número de coincidencias (No):", html.Div(id='output-coincidenceN')]),
            html.Div(["Número total de coincidencias:", html.Div(id='output-coincidence')]),
            # Gráfico de torta para las coincidencias
            html.Div([
                dcc.Graph(id='pie-chart-coincidence', style={'width': '48%', 'display': 'inline-block'}),
                dcc.Graph(id='pie-chart-figLH', style={'width': '48%', 'display': 'inline-block'}),
            ], style={'display': 'flex', 'justify-content': 'center'}),
        ]),
        html.Div(style={'margin': '20px 0', 'padding': '20px', 'border': '1px solid #ccc', 'border-radius': '10px', 'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)', 'text-align': 'center'}, children=[
            html.Div([
            dcc.Graph(id='output-prediction-pie')
            ], style={'display': 'flex', 'justify-content': 'center'}),
        ]),
    ]
)

# Definir el callback para la gráfica de comparación
# Callback para actualizar la gráfica y la tabla
@app.callback(
    [Output('comparison-graph', 'figure'),
     Output('threshold-table', 'figure')],
    [Input('umbral', 'value')]
)
def update_output(umbral):
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
    # Crear la gráfica
    indices = list(resultados.keys())
    ingresos = [resultados[key]["Ingreso esperado"] for key in resultados]
    accuracy_values = [resultados[key]["Accuracy"] for key in resultados]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=indices, y=ingresos, mode='lines+markers', name='Ingreso esperado', yaxis="y1"))
    fig.add_trace(go.Scatter(x=indices, y=accuracy_values, mode='lines+markers', name='Accuracy', yaxis="y2"))

    fig.update_layout(
        title="Comparación entre Ingreso esperado y Accuracy según el umbral",
        xaxis=dict(title="Umbral"),
        yaxis=dict(title="Ingreso esperado (€)", side="left"),
        yaxis2=dict(title="Accuracy", overlaying="y", side="right", tickformat=".0%", range=[0.25, 1])
    )

    # Crear la tabla
    ingreso_esperado = resultados[umbral]["Ingreso esperado"]
    accuracy = resultados[umbral]["Accuracy"]

    table_fig = go.Figure(data=[go.Table(
        header=dict(values=["Umbral", "Ingreso Esperado (€)", "Accuracy"],
                    fill_color='lavender',
                    align='left',
                    font=dict(size=14)),
        cells=dict(values=[[umbral], [ingreso_esperado], [accuracy]],
                   align='left')
    )])

    table_fig.update_layout(title="Tabla de resultados para el umbral seleccionado")

    return fig, table_fig

@app.callback(
    Output('prices-graph', 'figure'),
    [Input('umbral', 'value')]
)

def update_prices(umbral):
    # Example price data
    datos_precios = {
        'TP_precio': 106.68880783994922,
        'FP_precio': -0.1238527007957243,
        'TN_precio': 0.1238527007957243,
        'FN_precio': -106.81266054074494
    }

    # Create a 2D array for the heatmap
    price_matrix = np.array([[datos_precios['TP_precio'], datos_precios['FP_precio']],
                              [datos_precios['TN_precio'], datos_precios['FN_precio']]])

    # Create a heatmap for price values
    fig = go.Figure(data=go.Heatmap(
        z=price_matrix,
        x=['Predicted Positive', 'Predicted Negative'],
        y=['Actual Positive', 'Actual Negative'],
        colorscale='RdPu',
        colorbar=dict(title='Price Value')
    ))

    # Add annotations for price values
    for i in range(price_matrix.shape[0]):
        for j in range(price_matrix.shape[1]):
            fig.add_annotation(
                x=j,
                y=i,
                text=f'{price_matrix[i, j]:.2f}',
                showarrow=False,
                font=dict(color='white' if price_matrix[i, j] > 100 else 'black')
            )

    # Update layout
    fig.update_layout(
        title='Matriz de precios',
        xaxis_title='Tipo de Precio Predecido',
        yaxis_title='Tipo de precio Actual'
    )

    return fig

@app.callback(
    Output('output-coincidenceE', 'children'),
    Output('output-coincidenceY', 'children'),
    Output('output-coincidenceN', 'children'),
    Output('output-coincidence', 'children'),
    Output('pie-chart-coincidence', 'figure'),  # Output para el gráfico de torta
    Output('pie-chart-figLH', 'figure'),  # Output para el gráfico de torta
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

    custom_colors = ['#7a0177','#f768a1']  # Dark pink for 'Sí' and dark purple for 'No'

    # Gráfico de torta
    pie_data_Coincidences = {
        'Resultado': ['Sí', 'No'],
        'Coincidencias': [coincidenceY, coincidenceN]
    }
    pie_df = pd.DataFrame(pie_data_Coincidences)
    fig1 = px.pie(pie_df, names='Resultado', values='Coincidencias', title="Distribución de Coincidencias (Sí/No)", color_discrete_sequence=custom_colors)  # Apply the custom color sequence



    # Gráfico de torta
    pie_data_loan_Housing = {
        'Resultado': ['Sí', 'No'],
        'LoansHousing': [loand_housign_Y, loand_housign_N]
    }
    pie_df_loan_housing = pd.DataFrame(pie_data_loan_Housing)
    figLH = px.pie(pie_df_loan_housing, names='Resultado', values='LoansHousing', title="Distribución de (Sí/No) de acuerdo a Loan y Housing", color_discrete_sequence=custom_colors) 

    return coincidenceE, coincidenceY, coincidenceN, coincidence, fig1, figLH

# Callback for updating client prediction
@app.callback(
    Output('output-prediction-pie', 'figure'),
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

    ypred = model3.predict(input)
    #Seleccion del modelo:
    if umbral == 0.1:
        ypred = model1.predict(input)
    elif umbral == 0.2:
        ypred = model2.predict(input)
    elif umbral == 0.4:
        ypred = model4.predict(input)
    elif umbral == 0.5:
        ypred = model5.predict(input)
    elif umbral == 0.6:
        ypred = model6.predict(input)
    elif umbral == 0.7:
        ypred = model7.predict(input)
    elif umbral == 0.8:
        ypred = model8.predict(input)
    elif umbral == 0.9:
        ypred = model9.predict(input)
# Prepare data for the pie chart
    positive_prob = ypred[0][0]
    negative_prob = 1 - positive_prob

    pie_data = {
        'Outcome': ['Yes', 'No'],
        'Probability': [positive_prob, negative_prob]
    }

    # Create the pie chart
    fig = px.pie(pie_data, names='Outcome', values='Probability', title='Distribución de la predicción',
                 color_discrete_sequence=['#7a0177', '#f768a1'])  # Custom colors

    # Return the pie chart figure
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)