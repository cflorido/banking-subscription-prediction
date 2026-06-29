from pathlib import Path
import os

from dash import Dash, Input, Output, clientside_callback, dcc, html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
from dotenv import load_dotenv
from tensorflow import keras

app = Dash(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / 'artifacts' / 'models' / 'final'

load_dotenv(ROOT_DIR / '.env')

DB_CONFIG = {
    'dbname': os.getenv('DB_NAME', 'prod'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'manolo21'),
    'host': os.getenv('DB_HOST', 'database-2.cw2fwyuh7xdb.us-east-1.rds.amazonaws.com'),
    'port': os.getenv('DB_PORT', '5432'),
    'connect_timeout': int(os.getenv('DB_CONNECT_TIMEOUT', '5')),
}

DB_CONNECTION_ERROR = None

try:
    engine = psycopg2.connect(**DB_CONFIG)
except psycopg2.Error as exc:
    engine = None
    DB_CONNECTION_ERROR = str(exc)

def safe_load_model(path):
    try:
        return keras.models.load_model(str(path))
    except Exception:
        return None


model3 = safe_load_model(MODEL_DIR / 'modeloproy_03.keras')
model1 = safe_load_model(MODEL_DIR / 'modeloproy_01.keras')
model2 = safe_load_model(MODEL_DIR / 'modeloproy_02.keras')
model4 = safe_load_model(MODEL_DIR / 'modeloproy_04.keras')
model5 = safe_load_model(MODEL_DIR / 'modeloproy_05.keras')
model6 = safe_load_model(MODEL_DIR / 'modeloproy_06.keras')
model7 = safe_load_model(MODEL_DIR / 'modeloproy_07.keras')
model8 = safe_load_model(MODEL_DIR / 'modeloproy_08.keras')
model9 = safe_load_model(MODEL_DIR / 'modeloproy_09.keras')


def normalize_language(language):
    if isinstance(language, str) and language.lower().startswith('es'):
        return 'es'
    return 'en'


def resolve_language(selected_language, browser_language):
    if selected_language == 'auto':
        return normalize_language(browser_language)
    return normalize_language(selected_language)


def get_texts(language):
    lang = normalize_language(language)
    if lang == 'es':
        return {
            'title': 'Analisis de Datos de Clientes',
            'language_label': 'Idioma:',
            'threshold_prompt': 'Ingrese el umbral que desea utilizar de acuerdo a la grafica anterior',
            'profile_prompt': 'Seleccione el perfil del cliente que quiere analizar',
            'stats_title': 'Estadisticas:',
            'age_label': 'Edad (18 a 99):',
            'balance_label': 'Saldo promedio anual (euros):',
            'day_label': 'Dia del mes del ultimo contacto (1-31):',
            'duration_label': 'Duracion de la ultima llamada (segundos):',
            'campaign_label': 'Numero de contactos realizados durante esta campana:',
            'pdays_label': 'Dias desde el ultimo contacto de una campana anterior (si no fue contactado, ingrese -1):',
            'previous_label': 'Numero de contactos realizados antes de esta campana:',
            'job_label': 'Ocupacion:',
            'marital_label': 'Estado civil:',
            'education_label': 'Nivel educativo:',
            'housing_label': 'Tiene un prestamo de vivienda?:',
            'loan_label': 'Tiene un prestamo personal?:',
            'contact_label': 'Contacto:',
            'month_label': 'Mes del ultimo contacto:',
            'poutcome_label': 'Resultado de la campana anterior:',
            'exact_matches': 'Numero de coincidencias exactas:',
            'yes_matches': 'Numero de coincidencias (Si):',
            'no_matches': 'Numero de coincidencias (No):',
            'total_matches': 'Numero total de coincidencias:',
            'yes': 'Si',
            'no': 'No',
            'comparison_title': 'Comparacion entre ingreso esperado y accuracy segun el umbral',
            'threshold_axis': 'Umbral',
            'income_axis': 'Ingreso esperado (EUR)',
            'accuracy_axis': 'Accuracy',
            'income_trace': 'Ingreso esperado',
            'accuracy_trace': 'Accuracy',
            'table_title': 'Tabla de resultados para el umbral seleccionado',
            'price_matrix_title': 'Matriz de precios',
            'predicted_price_axis': 'Tipo de precio predicho',
            'actual_price_axis': 'Tipo de precio real',
            'coincidence_pie_title': 'Distribucion de coincidencias (Si/No)',
            'loan_housing_pie_title': 'Distribucion de (Si/No) segun loan y housing',
            'prediction_pie_title': 'Distribucion de la prediccion',
            'predicted_positive': 'Predicho positivo',
            'predicted_negative': 'Predicho negativo',
            'actual_positive': 'Real positivo',
            'actual_negative': 'Real negativo',
            'outcome': 'Resultado',
            'probability': 'Probabilidad',
            'db_warning_title': 'Aviso de conexion: ',
            'db_warning_message': 'La aplicacion se esta ejecutando en modo sin conexion. Las estadisticas en vivo no estan disponibles por ahora.',
        }

    return {
        'title': 'Customer Data Analysis',
        'language_label': 'Language:',
        'threshold_prompt': 'Select the threshold based on the previous chart',
        'profile_prompt': 'Select the customer profile to analyze',
        'stats_title': 'Statistics:',
        'age_label': 'Age (18 to 99):',
        'balance_label': 'Average yearly balance (EUR):',
        'day_label': 'Day of month of last contact (1-31):',
        'duration_label': 'Last call duration (seconds):',
        'campaign_label': 'Number of contacts during this campaign:',
        'pdays_label': 'Days since last contact from previous campaign (if never contacted, enter -1):',
        'previous_label': 'Number of contacts before this campaign:',
        'job_label': 'Job:',
        'marital_label': 'Marital status:',
        'education_label': 'Education level:',
        'housing_label': 'Has housing loan?:',
        'loan_label': 'Has personal loan?:',
        'contact_label': 'Contact type:',
        'month_label': 'Month of last contact:',
        'poutcome_label': 'Previous campaign outcome:',
        'exact_matches': 'Exact matches:',
        'yes_matches': 'Matches (Yes):',
        'no_matches': 'Matches (No):',
        'total_matches': 'Total matches:',
        'yes': 'Yes',
        'no': 'No',
        'comparison_title': 'Expected revenue vs accuracy by threshold',
        'threshold_axis': 'Threshold',
        'income_axis': 'Expected revenue (EUR)',
        'accuracy_axis': 'Accuracy',
        'income_trace': 'Expected revenue',
        'accuracy_trace': 'Accuracy',
        'table_title': 'Results table for selected threshold',
        'price_matrix_title': 'Price matrix',
        'predicted_price_axis': 'Predicted price type',
        'actual_price_axis': 'Actual price type',
        'coincidence_pie_title': 'Match distribution (Yes/No)',
        'loan_housing_pie_title': 'Distribution (Yes/No) by loan and housing',
        'prediction_pie_title': 'Prediction distribution',
        'predicted_positive': 'Predicted positive',
        'predicted_negative': 'Predicted negative',
        'actual_positive': 'Actual positive',
        'actual_negative': 'Actual negative',
        'outcome': 'Outcome',
        'probability': 'Probability',
        'db_warning_title': 'Connection notice: ',
        'db_warning_message': 'The app is running in offline mode. Live database statistics are currently unavailable.',
    }


def get_dropdown_options(language):
    lang = normalize_language(language)
    if lang == 'es':
        return {
            'language': [
                {'label': 'Ingles', 'value': 'en'},
                {'label': 'Espanol', 'value': 'es'},
                {'label': 'Automatico (navegador)', 'value': 'auto'},
            ],
            'job': [
                {'label': 'Desconocido', 'value': 'unknown'},
                {'label': 'Desempleado', 'value': 'unemployed'},
                {'label': 'Estudiante', 'value': 'student'},
                {'label': 'Gerente', 'value': 'management'},
                {'label': 'Tecnico', 'value': 'technician'},
                {'label': 'Empresario', 'value': 'entrepreneur'},
                {'label': 'Obrero', 'value': 'blue-collar'},
                {'label': 'Jubilado', 'value': 'retired'},
                {'label': 'Administrativo', 'value': 'admin.'},
                {'label': 'Servicios', 'value': 'services'},
                {'label': 'Autonomo', 'value': 'self-employed'},
                {'label': 'Ama de casa', 'value': 'housemaid'},
            ],
            'marital': [
                {'label': 'Soltero', 'value': 'single'},
                {'label': 'Casado', 'value': 'married'},
                {'label': 'Divorciado', 'value': 'divorced'},
            ],
            'education': [
                {'label': 'Desconocido', 'value': 'unknown'},
                {'label': 'Primario', 'value': 'primary'},
                {'label': 'Secundario', 'value': 'secondary'},
                {'label': 'Terciario', 'value': 'tertiary'},
            ],
            'housing': [{'label': 'Si', 'value': 'yes'}, {'label': 'No', 'value': 'no'}],
            'loan': [{'label': 'Si', 'value': 'yes'}, {'label': 'No', 'value': 'no'}],
            'contact': [
                {'label': 'Desconocido', 'value': 'unknown'},
                {'label': 'Celular', 'value': 'cellular'},
                {'label': 'Telefono', 'value': 'telephone'},
            ],
            'month': [
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
                {'label': 'Diciembre', 'value': 'dec'},
            ],
            'poutcome': [
                {'label': 'Desconocido', 'value': 'unknown'},
                {'label': 'Otro', 'value': 'other'},
                {'label': 'Fracaso', 'value': 'failure'},
                {'label': 'Exito', 'value': 'success'},
            ],
        }

    return {
        'language': [
            {'label': 'English', 'value': 'en'},
            {'label': 'Spanish', 'value': 'es'},
            {'label': 'Auto (browser)', 'value': 'auto'},
        ],
        'job': [
            {'label': 'Unknown', 'value': 'unknown'},
            {'label': 'Unemployed', 'value': 'unemployed'},
            {'label': 'Student', 'value': 'student'},
            {'label': 'Management', 'value': 'management'},
            {'label': 'Technician', 'value': 'technician'},
            {'label': 'Entrepreneur', 'value': 'entrepreneur'},
            {'label': 'Blue-collar', 'value': 'blue-collar'},
            {'label': 'Retired', 'value': 'retired'},
            {'label': 'Admin', 'value': 'admin.'},
            {'label': 'Services', 'value': 'services'},
            {'label': 'Self-employed', 'value': 'self-employed'},
            {'label': 'Housemaid', 'value': 'housemaid'},
        ],
        'marital': [
            {'label': 'Single', 'value': 'single'},
            {'label': 'Married', 'value': 'married'},
            {'label': 'Divorced', 'value': 'divorced'},
        ],
        'education': [
            {'label': 'Unknown', 'value': 'unknown'},
            {'label': 'Primary', 'value': 'primary'},
            {'label': 'Secondary', 'value': 'secondary'},
            {'label': 'Tertiary', 'value': 'tertiary'},
        ],
        'housing': [{'label': 'Yes', 'value': 'yes'}, {'label': 'No', 'value': 'no'}],
        'loan': [{'label': 'Yes', 'value': 'yes'}, {'label': 'No', 'value': 'no'}],
        'contact': [
            {'label': 'Unknown', 'value': 'unknown'},
            {'label': 'Cellular', 'value': 'cellular'},
            {'label': 'Telephone', 'value': 'telephone'},
        ],
        'month': [
            {'label': 'January', 'value': 'jan'},
            {'label': 'February', 'value': 'feb'},
            {'label': 'March', 'value': 'mar'},
            {'label': 'April', 'value': 'apr'},
            {'label': 'May', 'value': 'may'},
            {'label': 'June', 'value': 'jun'},
            {'label': 'July', 'value': 'jul'},
            {'label': 'August', 'value': 'aug'},
            {'label': 'September', 'value': 'sep'},
            {'label': 'October', 'value': 'oct'},
            {'label': 'November', 'value': 'nov'},
            {'label': 'December', 'value': 'dec'},
        ],
        'poutcome': [
            {'label': 'Unknown', 'value': 'unknown'},
            {'label': 'Other', 'value': 'other'},
            {'label': 'Failure', 'value': 'failure'},
            {'label': 'Success', 'value': 'success'},
        ],
    }


app.layout = html.Div(
    className='page-shell',
    style={'textAlign': 'center', 'padding': '20px'},
    children=[
        dcc.Location(id='url', refresh=False),
        dcc.Store(id='browser-language', data='en'),
        html.Div(
            className='topbar',
            style={
                'display': 'flex',
                'justifyContent': 'flex-end',
                'alignItems': 'center',
                'gap': '8px',
                'marginBottom': '12px',
            },
            children=[
                html.Div(id='label-language', className='topbar-label', children='Language:'),
                dcc.Dropdown(id='language-selector', options=[], value='en', clearable=False, className='lang-dropdown', style={'width': '220px'}),
            ],
        ),
        html.Div(
            className='hero-banner',
            style={'backgroundColor': '#fa9fb5', 'padding': '30px', 'borderRadius': '10px'},
            children=[
                html.H1(id='title-text', children='Customer Data Analysis', style={'fontSize': '40px', 'fontWeight': 'bold', 'color': '#333333'})
            ],
        ),
        html.Div(
            className='panel',
            style={
                'margin': '20px 0',
                'padding': '14px 20px',
                'border': '1px solid #f0b27a',
                'borderRadius': '10px',
                'backgroundColor': '#fff3e8',
                'display': 'none' if DB_CONNECTION_ERROR is None else 'block',
            },
            children=[
                html.Strong(id='db-warning-title', children='Connection notice: '),
                html.Span(id='db-warning-message', children='The app is running in offline mode. Live database statistics are currently unavailable.'),
            ],
        ),
        html.Div(
            className='panel',
            style={
                'margin': '20px 0',
                'padding': '20px',
                'border': '1px solid #ccc',
                'borderRadius': '10px',
                'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
            },
            children=[
                html.Div([
                    dcc.Graph(id='comparison-graph'),
                    dcc.Graph(id='prices-graph'),
                ], style={'display': 'flex', 'flex-direction': 'column', 'gap': '20px'})
            ],
        ),
        html.Div(
            className='panel',
            style={
                'margin': '20px 0',
                'padding': '20px',
                'border': '1px solid #ccc',
                'borderRadius': '10px',
                'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
            },
            children=[
                html.H3(id='threshold-text', children='Select the threshold based on the previous chart'),
                dcc.Slider(id='umbral', min=0.1, max=0.9, step=0.1, value=0.3, marks={i / 10: str(i / 10) for i in range(1, 10)}),
                html.Div([dcc.Graph(id='threshold-table')], style={'display': 'flex', 'flex-direction': 'column', 'gap': '20px'}),
            ],
        ),
        html.Div(
            className='panel panel-profile',
            style={
                'margin': '20px 0',
                'padding': '20px',
                'border': '1px solid #ccc',
                'borderRadius': '10px',
                'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
            },
            children=[
                html.H3(id='profile-text', children='Select the customer profile to analyze'),
                html.Div(
                    className='form-grid',
                    style={'display': 'flex', 'flex-wrap': 'wrap', 'gap': '20px'},
                    children=[
                        html.Div(
                            className='form-column',
                            style={'flex': '1', 'min-width': '200px'},
                            children=[
                                html.Div([
                                    html.Div(id='label-age', className='field-label', children='Age (18 to 99):', style={'font-weight': 'bold', 'text-align': 'left', 'margin-bottom': '5px'}),
                                    dcc.Input(id='age', className='field-input', type='number', min=18, max=99, step=1, value=21),
                                ], style={'display': 'flex', 'flex-direction': 'column'}),
                                html.Br(),
                                html.Div([
                                    html.Div(id='label-balance', className='field-label', children='Average yearly balance (EUR):', style={'font-weight': 'bold', 'text-align': 'left', 'margin-bottom': '5px'}),
                                    dcc.Input(id='balance', className='field-input', type='number', min=0, max=10000000, step=1, value=0),
                                ], style={'display': 'flex', 'flex-direction': 'column'}),
                                html.Br(),
                                html.Div([
                                    html.Div(id='label-day', className='field-label', children='Day of month of last contact (1-31):', style={'font-weight': 'bold', 'text-align': 'left', 'margin-bottom': '5px'}),
                                    dcc.Input(id='day', className='field-input', type='number', min=1, max=31, step=1, value=1),
                                ], style={'display': 'flex', 'flex-direction': 'column'}),
                                html.Br(),
                                html.Div([
                                    html.Div(id='label-duration', className='field-label', children='Last call duration (seconds):', style={'font-weight': 'bold', 'text-align': 'left', 'margin-bottom': '5px'}),
                                    dcc.Input(id='duration', className='field-input', type='number', min=0, max=10000, step=1, value=0),
                                ], style={'display': 'flex', 'flex-direction': 'column'}),
                                html.Br(),
                                html.Div([
                                    html.Div(id='label-campaign', className='field-label', children='Number of contacts during this campaign:', style={'font-weight': 'bold', 'text-align': 'left', 'margin-bottom': '5px'}),
                                    dcc.Input(id='campaign', className='field-input', type='number', min=0, max=100, step=1, value=0),
                                ], style={'display': 'flex', 'flex-direction': 'column'}),
                                html.Br(),
                                html.Div([
                                    html.Div(id='label-pdays', className='field-label', children='Days since last contact from previous campaign (if never contacted, enter -1):', style={'font-weight': 'bold', 'text-align': 'left', 'margin-bottom': '5px'}),
                                    dcc.Input(id='pdays', className='field-input', type='number', min=-1, max=1000, step=1, value=-1),
                                ], style={'display': 'flex', 'flex-direction': 'column'}),
                                html.Br(),
                                html.Div([
                                    html.Div(id='label-previous', className='field-label', children='Number of contacts before this campaign:', style={'font-weight': 'bold', 'text-align': 'left', 'margin-bottom': '5px'}),
                                    dcc.Input(id='previous', className='field-input', type='number', min=0, max=100, step=1, value=0),
                                ], style={'display': 'flex', 'flex-direction': 'column'}),
                            ],
                        ),
                        html.Div(
                            className='form-column',
                            style={'flex': '1', 'min-width': '200px'},
                            children=[
                                html.Div([
                                    html.Div(id='label-job', className='field-label', children='Job:', style={'font-weight': 'bold', 'text-align': 'left', 'margin-bottom': '5px'}),
                                    dcc.Dropdown(id='job', className='field-dropdown', value='unknown', options=[]),
                                ]),
                                html.Br(),
                                html.Div([
                                    html.Div(id='label-marital', className='field-label', children='Marital status:', style={'font-weight': 'bold', 'text-align': 'left', 'margin-bottom': '5px'}),
                                    dcc.Dropdown(id='marital', className='field-dropdown', value='single', options=[]),
                                ]),
                                html.Br(),
                                html.Div([
                                    html.Div(id='label-education', className='field-label', children='Education level:', style={'font-weight': 'bold', 'text-align': 'left', 'margin-bottom': '5px'}),
                                    dcc.Dropdown(id='education', className='field-dropdown', value='unknown', options=[]),
                                ]),
                                html.Br(),
                                html.Div([
                                    html.Div(id='label-housing', className='field-label', children='Has housing loan?:', style={'font-weight': 'bold', 'text-align': 'left', 'margin-bottom': '5px'}),
                                    dcc.Dropdown(id='housing', className='field-dropdown', value='no', options=[]),
                                ]),
                                html.Br(),
                                html.Div([
                                    html.Div(id='label-loan', className='field-label', children='Has personal loan?:', style={'font-weight': 'bold', 'text-align': 'left', 'margin-bottom': '5px'}),
                                    dcc.Dropdown(id='loan', className='field-dropdown', value='no', options=[]),
                                ]),
                                html.Br(),
                                html.Div([
                                    html.Div(id='label-contact', className='field-label', children='Contact type:', style={'font-weight': 'bold', 'text-align': 'left', 'margin-bottom': '5px'}),
                                    dcc.Dropdown(id='contact', className='field-dropdown', value='unknown', options=[]),
                                ]),
                                html.Br(),
                                html.Div([
                                    html.Div(id='label-month', className='field-label', children='Month of last contact:', style={'font-weight': 'bold', 'text-align': 'left', 'margin-bottom': '5px'}),
                                    dcc.Dropdown(id='month', className='field-dropdown', value='jan', options=[]),
                                ]),
                                html.Br(),
                                html.Div([
                                    html.Div(id='label-poutcome', className='field-label', children='Previous campaign outcome:', style={'font-weight': 'bold', 'text-align': 'left', 'margin-bottom': '5px'}),
                                    dcc.Dropdown(id='poutcome', className='field-dropdown', value='unknown', options=[]),
                                ]),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            className='panel',
            style={
                'margin': '20px 0',
                'padding': '20px',
                'border': '1px solid #ccc',
                'borderRadius': '10px',
                'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
            },
            children=[
                html.H3(id='stats-text', children='Statistics:'),
                html.Div([html.Span(id='label-exact', children='Exact matches:'), html.Div(id='output-coincidenceE')]),
                html.Br(),
                html.Div([html.Span(id='label-yes', children='Matches (Yes):'), html.Div(id='output-coincidenceY')]),
                html.Br(),
                html.Div([html.Span(id='label-no', children='Matches (No):'), html.Div(id='output-coincidenceN')]),
                html.Br(),
                html.Div([html.Span(id='label-total', children='Total matches:'), html.Div(id='output-coincidence')]),
                html.Div([
                    dcc.Graph(id='pie-chart-coincidence', style={'width': '48%', 'display': 'inline-block'}),
                    dcc.Graph(id='pie-chart-figLH', style={'width': '48%', 'display': 'inline-block'}),
                ], style={'display': 'flex', 'justify-content': 'center'}),
            ],
        ),
        html.Div(
            className='panel',
            style={
                'margin': '20px 0',
                'padding': '20px',
                'border': '1px solid #ccc',
                'borderRadius': '10px',
                'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
            },
            children=[html.Div([dcc.Graph(id='output-prediction-pie')], style={'display': 'flex', 'justify-content': 'center'})],
        ),
    ],
)


clientside_callback(
    """
    function(pathname) {
        const navLang = (navigator.language || navigator.userLanguage || 'en').toLowerCase();
        return navLang.startsWith('es') ? 'es' : 'en';
    }
    """,
    Output('browser-language', 'data'),
    Input('url', 'pathname'),
)


@app.callback(
    Output('language-selector', 'value'),
    Input('browser-language', 'data'),
)
def initialize_language(browser_language):
    del browser_language
    return 'en'


@app.callback(
    Output('label-language', 'children'),
    Output('title-text', 'children'),
    Output('threshold-text', 'children'),
    Output('profile-text', 'children'),
    Output('stats-text', 'children'),
    Output('label-age', 'children'),
    Output('label-balance', 'children'),
    Output('label-day', 'children'),
    Output('label-duration', 'children'),
    Output('label-campaign', 'children'),
    Output('label-pdays', 'children'),
    Output('label-previous', 'children'),
    Output('label-job', 'children'),
    Output('label-marital', 'children'),
    Output('label-education', 'children'),
    Output('label-housing', 'children'),
    Output('label-loan', 'children'),
    Output('label-contact', 'children'),
    Output('label-month', 'children'),
    Output('label-poutcome', 'children'),
    Output('label-exact', 'children'),
    Output('label-yes', 'children'),
    Output('label-no', 'children'),
    Output('label-total', 'children'),
    Output('db-warning-title', 'children'),
    Output('db-warning-message', 'children'),
    Output('language-selector', 'options'),
    Output('job', 'options'),
    Output('marital', 'options'),
    Output('education', 'options'),
    Output('housing', 'options'),
    Output('loan', 'options'),
    Output('contact', 'options'),
    Output('month', 'options'),
    Output('poutcome', 'options'),
    Input('language-selector', 'value'),
    Input('browser-language', 'data'),
)
def update_language(language, browser_language):
    lang = resolve_language(language, browser_language)
    texts = get_texts(lang)
    options = get_dropdown_options(lang)

    return (
        texts['language_label'],
        texts['title'],
        texts['threshold_prompt'],
        texts['profile_prompt'],
        texts['stats_title'],
        texts['age_label'],
        texts['balance_label'],
        texts['day_label'],
        texts['duration_label'],
        texts['campaign_label'],
        texts['pdays_label'],
        texts['previous_label'],
        texts['job_label'],
        texts['marital_label'],
        texts['education_label'],
        texts['housing_label'],
        texts['loan_label'],
        texts['contact_label'],
        texts['month_label'],
        texts['poutcome_label'],
        texts['exact_matches'],
        texts['yes_matches'],
        texts['no_matches'],
        texts['total_matches'],
        texts['db_warning_title'],
        texts['db_warning_message'],
        options['language'],
        options['job'],
        options['marital'],
        options['education'],
        options['housing'],
        options['loan'],
        options['contact'],
        options['month'],
        options['poutcome'],
    )


@app.callback(
    Output('comparison-graph', 'figure'),
    Output('threshold-table', 'figure'),
    Input('umbral', 'value'),
    Input('language-selector', 'value'),
    Input('browser-language', 'data'),
)
def update_output(umbral, language, browser_language):
    texts = get_texts(resolve_language(language, browser_language))
    resultados = {
        0.1: {'Ingreso esperado': 98213.46, 'Accuracy': 0.30},
        0.2: {'Ingreso esperado': 59827.85, 'Accuracy': 0.58},
        0.3: {'Ingreso esperado': 52050.41, 'Accuracy': 0.63},
        0.4: {'Ingreso esperado': 42358.89, 'Accuracy': 0.68},
        0.5: {'Ingreso esperado': 19292.33, 'Accuracy': 0.76},
        0.6: {'Ingreso esperado': 9156.47, 'Accuracy': 0.81},
        0.7: {'Ingreso esperado': -13594.80, 'Accuracy': 0.84},
        0.8: {'Ingreso esperado': -27413.94, 'Accuracy': 0.86},
        0.9: {'Ingreso esperado': -27624.71, 'Accuracy': 0.86},
    }

    indices = list(resultados.keys())
    ingresos = [resultados[key]['Ingreso esperado'] for key in resultados]
    accuracy_values = [resultados[key]['Accuracy'] for key in resultados]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=indices, y=ingresos, mode='lines+markers', name=texts['income_trace'], yaxis='y1'))
    fig.add_trace(go.Scatter(x=indices, y=accuracy_values, mode='lines+markers', name=texts['accuracy_trace'], yaxis='y2'))

    fig.update_layout(
        title=texts['comparison_title'],
        xaxis=dict(title=texts['threshold_axis']),
        yaxis=dict(title=texts['income_axis'], side='left'),
        yaxis2=dict(title=texts['accuracy_axis'], overlaying='y', side='right', tickformat='.0%', range=[0.25, 1]),
    )

    ingreso_esperado = resultados[umbral]['Ingreso esperado']
    accuracy = resultados[umbral]['Accuracy']

    table_fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[texts['threshold_axis'], texts['income_axis'], texts['accuracy_axis']],
                    fill_color='#7a0177',
                    align='center',
                    font=dict(size=14, color='white'),
                ),
                cells=dict(values=[[umbral], [ingreso_esperado], [accuracy]], fill_color='#fde0dd', align='center'),
            )
        ]
    )

    table_fig.update_layout(title=texts['table_title'])
    return fig, table_fig


@app.callback(
    Output('prices-graph', 'figure'),
    Input('umbral', 'value'),
    Input('language-selector', 'value'),
    Input('browser-language', 'data'),
)
def update_prices(_umbral, language, browser_language):
    texts = get_texts(resolve_language(language, browser_language))
    datos_precios = {
        'TP_precio': 106.68880783994922,
        'FP_precio': -0.1238527007957243,
        'TN_precio': 0.1238527007957243,
        'FN_precio': -106.81266054074494,
    }

    price_matrix = np.array([
        [datos_precios['TP_precio'], datos_precios['FP_precio']],
        [datos_precios['TN_precio'], datos_precios['FN_precio']],
    ])

    fig = go.Figure(
        data=go.Heatmap(
            z=price_matrix,
            x=[texts['predicted_positive'], texts['predicted_negative']],
            y=[texts['actual_positive'], texts['actual_negative']],
            colorscale='RdPu',
            colorbar=dict(title='Price Value'),
        )
    )

    for i in range(price_matrix.shape[0]):
        for j in range(price_matrix.shape[1]):
            fig.add_annotation(
                x=j,
                y=i,
                text=f'{price_matrix[i, j]:.2f}',
                showarrow=False,
                font=dict(color='white' if price_matrix[i, j] > 100 else 'black'),
            )

    fig.update_layout(
        title=texts['price_matrix_title'],
        xaxis_title=texts['predicted_price_axis'],
        yaxis_title=texts['actual_price_axis'],
    )
    return fig


@app.callback(
    Output('output-coincidenceE', 'children'),
    Output('output-coincidenceY', 'children'),
    Output('output-coincidenceN', 'children'),
    Output('output-coincidence', 'children'),
    Output('pie-chart-coincidence', 'figure'),
    Output('pie-chart-figLH', 'figure'),
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
    Input('poutcome', 'value'),
    Input('language-selector', 'value'),
    Input('browser-language', 'data'),
)
def update_output_div(age, job, marital, education, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome, language, browser_language):
    texts = get_texts(resolve_language(language, browser_language))
    if engine is None:
        pie_df = pd.DataFrame(
            {
                texts['outcome']: [texts['yes'], texts['no']],
                'Coincidences': [0, 0],
            }
        )
        fig1 = px.pie(
            pie_df,
            names=texts['outcome'],
            values='Coincidences',
            title=f"{texts['coincidence_pie_title']} (offline)",
            color_discrete_sequence=['#7a0177', '#f768a1'],
        )

        pie_df_loan_housing = pd.DataFrame(
            {
                texts['outcome']: [texts['yes'], texts['no']],
                'LoansHousing': [0, 0],
            }
        )
        figLH = px.pie(
            pie_df_loan_housing,
            names=texts['outcome'],
            values='LoansHousing',
            title=f"{texts['loan_housing_pie_title']} (offline)",
            color_discrete_sequence=['#7a0177', '#f768a1'],
        )

        return 0, 0, 0, 0, fig1, figLH

    cursor = engine.cursor()

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
    coincidenceE = cursor.fetchone()[0] or 0

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
    coincidenceY = cursor.fetchone()[0] or 0

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
    coincidenceN = cursor.fetchone()[0] or 0

    query_loan_housingY = f"""
    SELECT COUNT(*)
    FROM bank_client_data
    WHERE housing = '{housing}' AND
        loan = '{loan}' AND
        y = 'yes';
    """
    cursor.execute(query_loan_housingY)
    loand_housign_Y = cursor.fetchone()[0] or 0

    query_loan_housingN = f"""
    SELECT COUNT(*)
    FROM bank_client_data
    WHERE housing = '{housing}' AND
        loan = '{loan}' AND
        y = 'no';
    """
    cursor.execute(query_loan_housingN)
    loand_housign_N = cursor.fetchone()[0] or 0

    coincidence = coincidenceY + coincidenceN
    custom_colors = ['#7a0177', '#f768a1']

    pie_df = pd.DataFrame(
        {
            texts['outcome']: [texts['yes'], texts['no']],
            'Coincidences': [coincidenceY, coincidenceN],
        }
    )
    fig1 = px.pie(
        pie_df,
        names=texts['outcome'],
        values='Coincidences',
        title=texts['coincidence_pie_title'],
        color_discrete_sequence=custom_colors,
    )

    pie_df_loan_housing = pd.DataFrame(
        {
            texts['outcome']: [texts['yes'], texts['no']],
            'LoansHousing': [loand_housign_Y, loand_housign_N],
        }
    )
    figLH = px.pie(
        pie_df_loan_housing,
        names=texts['outcome'],
        values='LoansHousing',
        title=texts['loan_housing_pie_title'],
        color_discrete_sequence=custom_colors,
    )

    return coincidenceE, coincidenceY, coincidenceN, coincidence, fig1, figLH


@app.callback(
    Output('output-prediction-pie', 'figure'),
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
    Input('poutcome', 'value'),
    Input('language-selector', 'value'),
    Input('browser-language', 'data'),
)
def update_prediccion_cliente(umbral, age, job, marital, education, balance, contact, day, month, duration, campaign, pdays, previous, poutcome, language, browser_language):
    texts = get_texts(resolve_language(language, browser_language))
    umbral = float(umbral) if umbral is not None else 0.3

    age = int(age) if age is not None else 18
    job = job if job is not None else 'unknown'
    marital = marital if marital is not None else 'single'
    education = education if education is not None else 'unknown'
    balance = balance if balance is not None else 0
    contact = contact if contact is not None else 'unknown'
    day = day if day is not None else 1
    month = month if month is not None else 'jan'
    duration = duration if duration is not None else 0
    campaign = campaign if campaign is not None else 0
    pdays = pdays if pdays is not None else -1
    previous = previous if previous is not None else 0
    poutcome = poutcome if poutcome is not None else 'unknown'

    encoder = {
        'contact': {'cellular': 0, 'telephone': 1, 'unknown': 2},
        'education': {'primary': 0, 'secondary': 1, 'tertiary': 2, 'unknown': 3},
        'poutcome': {'failure': 0, 'other': 1, 'success': 2, 'unknown': 3},
        'marital': {'divorced': 0, 'married': 1, 'single': 2},
        'job': {
            'admin.': 0,
            'blue-collar': 1,
            'entrepreneur': 2,
            'housemaid': 3,
            'management': 4,
            'retired': 5,
            'self-employed': 6,
            'services': 7,
            'student': 8,
            'technician': 9,
            'unemployed': 10,
            'unknown': 11,
        },
        'month': {'apr': 0, 'aug': 1, 'dec': 2, 'feb': 3, 'jan': 4, 'jul': 5, 'jun': 6, 'mar': 7, 'may': 8, 'nov': 9, 'oct': 10, 'sep': 11},
    }

    job = encoder['job'][job]
    marital = encoder['marital'][marital]
    education = encoder['education'][education]
    contact = encoder['contact'][contact]
    month = encoder['month'][month]
    poutcome = encoder['poutcome'][poutcome]

    model_input = [
        np.array([contact]),
        np.array([education]),
        np.array([poutcome]),
        np.array([marital]),
        np.array([job]),
        np.array([month]),
        np.array([age]),
        np.array([balance]),
        np.array([campaign]),
        np.array([pdays]),
        np.array([previous]),
        np.array([duration]),
        np.array([day]),
    ]

    selected_model = model3
    if umbral == 0.1:
        selected_model = model1
    elif umbral == 0.2:
        selected_model = model2
    elif umbral == 0.4:
        selected_model = model4
    elif umbral == 0.5:
        selected_model = model5
    elif umbral == 0.6:
        selected_model = model6
    elif umbral == 0.7:
        selected_model = model7
    elif umbral == 0.8:
        selected_model = model8
    elif umbral == 0.9:
        selected_model = model9

    if selected_model is None:
        positive_prob = 0.5
        negative_prob = 0.5
    else:
        ypred = selected_model.predict(model_input)
        positive_prob = float(ypred[0][0])
        negative_prob = 1 - positive_prob

    pie_data = {
        texts['outcome']: [texts['yes'], texts['no']],
        texts['probability']: [positive_prob, negative_prob],
    }

    fig = px.pie(
        pie_data,
        names=texts['outcome'],
        values=texts['probability'],
        title=texts['prediction_pie_title'],
        color_discrete_sequence=['#7a0177', '#f768a1'],
    )

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
