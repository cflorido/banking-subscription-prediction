import psycopg2 
engine = psycopg2.connect( 
    dbname="proyect2", 
    user="proyect2", 
    password="manolo21", 
    host="database-proyect2.cw2fwyuh7xdb.us-east-1.rds.amazonaws.com", 
    port='5432' 
)