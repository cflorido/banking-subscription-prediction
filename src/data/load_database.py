import psycopg2 
engine = psycopg2.connect( 
    dbname="postgres", 
    user="postgres", 
    password="manolo21", 
    host="database-2.cw2fwyuh7xdb.us-east-1.rds.amazonaws.com", 
    port='5432' 
)


#### UBUNTU
# Conectar a la base de datos
engine = psycopg2.connect(
    dbname="prod", 
    user="postgres", 
    password="manolo21", 
    host="database-2.cw2fwyuh7xdb.us-east-1.rds.amazonaws.com", 
    port="5432"
)