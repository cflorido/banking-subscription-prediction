import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import psycopg2

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# Conectar a la base de datos
engine = psycopg2.connect(
    dbname="prod", 
    user="postgres", 
    password="manolo21", 
    host="database-2.cw2fwyuh7xdb.us-east-1.rds.amazonaws.com", 
    port="5432"
)

# Crear el diseño de la aplicación
app.layout = html.Div(
    [
        html.H6("Seleccione estado civil y respuesta de campaña"),
        html.Div(["Estado civil: ",
                  dcc.Dropdown(id='marital_status', value='single', 
                               options=[
                                   {'label': 'Soltero', 'value': 'single'},
                                   {'label': 'Casado', 'value': 'married'},
                                   {'label': 'Divorciado', 'value': 'divorced'}
                               ])]),
        html.Div(["Respuesta a campaña: ",
                  dcc.Dropdown(id='campaign_response', value='yes', 
                               options=[
                                   {'label': 'Sí', 'value': 'yes'},
                                   {'label': 'No', 'value': 'no'}
                               ])]),
        html.Br(),
        html.H6("Estadísticas:"),
        html.Br(),
        html.Div(["Saldo promedio:", html.Div(id='output-balance')]),
        html.Div(["Duración promedio de llamada (segundos):", html.Div(id='output-duration')]),
    ]
)

# Callback para actualizar los datos según el estado civil y la respuesta a la campaña
@app.callback(
    Output('output-balance', 'children'),
    Output('output-duration', 'children'),
    Input('marital_status', 'value'),
    Input('campaign_response', 'value')
)
def update_output_div(marital_status, campaign_response):
    cursor = engine.cursor()

    # Consulta para saldo promedio
    query_balance = f"""
    SELECT AVG(balance) 
    FROM bank_client_data 
    WHERE marital = '{marital_status}' AND y = '{campaign_response}';
    """
    cursor.execute(query_balance)
    result_balance = cursor.fetchone()
    balance = result_balance[0] if result_balance[0] is not None else 0

    # Consulta para duración promedio de llamadas
    query_duration = f"""
    SELECT AVG(duration) 
    FROM bank_client_data 
    WHERE marital = '{marital_status}' AND y = '{campaign_response}';
    """
    cursor.execute(query_duration)
    result_duration = cursor.fetchone()
    duration = result_duration[0] if result_duration[0] is not None else 0

    return f'{balance:.2f}', f'{duration:.2f}'

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)
