import psycopg2 
engine = psycopg2.connect( 
    dbname="postgres", 
    user="postgres", 
    password="manolo21", 
    host="database-2.cw2fwyuh7xdb.us-east-1.rds.amazonaws.com", 
    port='5432' 
)