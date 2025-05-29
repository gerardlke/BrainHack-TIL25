import mysql.connector
from mysql.connector import Error
import os
import subprocess
import argparse
from pathlib import Path


# Configs for local development
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = ""
DB_NAME = "RL"
DB_TABLE = "checkpoints"

# --- Database Connection Functions ---

def create_server_connection(host_name, user_name, user_password):
    """Establishes a connection to the MySQL server (without specifying a database)."""
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password
        )
        print("MySQL Server connection successful")
    except Error as err:
        print(f"Error: '{err}'")
    return connection

def create_db_connection(host_name, user_name, user_password, db_name):
    """Establishes a connection to a specific MySQL database."""
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name
        )
        print(f"Connection to database '{db_name}' successful")
    except Error as err:
        print(f"Error: '{err}'")
    return connection

def execute_query(connection, query):
    """Executes a SQL query."""
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query successful")
    except Error as err:
        print(f"Error: '{err}'")

def read_query(connection, query):
    """Executes a SQL query and returns the results."""
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except Error as err:
        print(f"Error: '{err}'")
        return None

# --- Database Management ---

def create_database(connection, db_name):
    """Creates a new database if it doesn't exist."""
    create_db_query = f"CREATE DATABASE IF NOT EXISTS {db_name}"
    execute_query(connection, create_db_query)

def create_table(connection, table_name):
    """Creates a table in the specified database."""
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        email VARCHAR(255) UNIQUE NOT NULL
    );
    """
    execute_query(connection, create_table_query)

# --- CRUD Operations (Skeleton) ---

def add_user(connection, name, email):
    """Adds a new user to the users table."""
    insert_user_query = f"""
    INSERT INTO {DB_TABLE} (name, email)
    VALUES ('{name}', '{email}');
    """
    execute_query(connection, insert_user_query)
    print(f"Added user: {name} ({email})")

def delete_user(connection, user_id):
    """Deletes a user by ID from the users table."""
    delete_user_query = f"DELETE FROM {DB_TABLE} WHERE id = {user_id};"
    execute_query(connection, delete_user_query)
    print(f"Deleted user with ID: {user_id}")

def get_all_users(connection):
    """Retrieves all users from the users table."""
    select_users_query = f"SELECT * FROM {DB_TABLE};"
    users = read_query(connection, select_users_query)
    if users:
        print("\n--- Current Users ---")
        for user in users:
            print(f"ID: {user[0]}, Name: {user[1]}, Email: {user[2]}")
        print("---------------------\n")
    else:
        print("\nNo users found in the database.\n")
    return users

# --- Database Export/Import (for "moving" the DB) ---

def export_database(db_name, export_file_path, user, password, host="localhost"):
    """
    Exports a MySQL database to a .sql dump file using mysqldump.
    Requires mysqldump to be in your system's PATH.
    """
    try:
        command = [
            "mysqldump",
            f"--host={host}",
            f"--user={user}",
            f"--password={password}",
            db_name
        ]
        with open(export_file_path, 'w') as f:
            subprocess.run(command, stdout=f, check=True)
        print(f"Database '{db_name}' exported successfully to '{export_file_path}'")
    except subprocess.CalledProcessError as e:
        print(f"Error exporting database: {e}")
        print("Please ensure 'mysqldump' is installed and in your system's PATH, and credentials are correct.")
    except FileNotFoundError:
        print("Error: 'mysqldump' command not found. Please ensure MySQL client tools are installed and in your PATH.")
    except Exception as e:
        print(f"An unexpected error occurred during export: {e}")

def import_database(db_name, import_file_path, user, password, host="localhost"):
    """
    Imports a MySQL database from a .sql dump file using the mysql client.
    Requires the mysql client to be in your system's PATH.
    Note: This will overwrite existing data in the target database.
    """
    try:
        # You might need to add --force if there are errors in the dump file
        command = [
            "mysql",
            f"--host={host}",
            f"--user={user}",
            f"--password={password}",
            db_name
        ]
        with open(import_file_path, 'r') as f:
            subprocess.run(command, stdin=f, check=True)
        print(f"Database '{db_name}' imported successfully from '{import_file_path}'")
    except subprocess.CalledProcessError as e:
        print(f"Error importing database: {e}")
        print("Please ensure 'mysql' client is installed and in your system's PATH, and credentials are correct.")
    except FileNotFoundError:
        print("Error: 'mysql' command not found. Please ensure MySQL client tools are installed and in your PATH.")
    except Exception as e:
        print(f"An unexpected error occurred during import: {e}")

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script with config file option")
    parser.add_argument(
        "-f", "--file",
        type=Path,
        required=False,
        help="Path to the database file"
    )
    args = parser.parse_args()
    db_file = args.file

    if not db_file:
        print(f'No database file inputted. Creating new database {DB_NAME} and table {DB_TABLE}.')
    else:
        print(f'Importing database file from {db_file}.')

    server = create_server_connection(DB_HOST, DB_USER, DB_PASSWORD)
    if server is None:
        print("Failed to connect to MySQL server. Exiting.")
        exit()

    print(f"\nAttempting to create database '{DB_NAME}'...")
    create_database(server_connection, DB_NAME)
    server.close() # Close server connection as we'll connect to specific DB now

    db = create_db_connection(DB_HOST, DB_USER, DB_PASSWORD, DB_NAME)
    if db is None:
        print("Failed to connect to the database. Exiting.")
        exit()

    print(f"\nAttempting to create table '{DB_TABLE}' in '{DB_NAME}'...")
    create_table(db, DB_TABLE)

    # print("\n--- Adding Users ---")
    # add_user(db, "Alice Smith", "alice@example.com")
    # add_user(db, "Bob Johnson", "bob@example.com")
    # add_user(db, "Charlie Brown", "charlie@example.com")

    # get_all_users(db_connection)

    # # --- Demonstrate Export/Import ---
    # export_file = f"{DB_NAME}_backup.sql"

    # print(f"\n--- Exporting Database to {export_file} ---")
    # export_database(DB_NAME, export_file, DB_USER, DB_PASSWORD, DB_HOST)

    # # Optional: Clean up the database before import to simulate moving to a new empty server
    # # WARNING: This will delete all data in your 'my_local_db' database!
    # confirm_delete = input(f"\nDo you want to drop database '{DB_NAME}' before importing? (y/N): ").lower()
    # if confirm_delete == 'y':
    #     print(f"Dropping database '{DB_NAME}'...")
    #     drop_db_conn = create_server_connection(DB_HOST, DB_USER, DB_PASSWORD)
    #     if drop_db_conn:
    #         execute_query(drop_db_conn, f"DROP DATABASE IF EXISTS {DB_NAME};")
    #         drop_db_conn.close()
    #         print(f"Database '{DB_NAME}' dropped.")
    #     else:
    #         print("Could not establish server connection to drop database.")

    # # 10. Import the database from the SQL file
    # print(f"\n--- Importing Database from {export_file} ---")
    # # You might need to create the database again if you dropped it,
    # # or ensure your .sql dump file contains the CREATE DATABASE statement.
    # # mysqldump usually includes CREATE DATABASE.
    # # If not, uncomment the following:
    # # server_conn_for_import = create_server_connection(DB_HOST, DB_USER, DB_PASSWORD)
    # # create_database(server_conn_for_import, DB_NAME)
    # # server_conn_for_import.close()

    # import_database(DB_NAME, export_file, DB_USER, DB_PASSWORD, DB_HOST)

    # # 11. Reconnect and verify imported data
    # print("\n--- Verifying Imported Data ---")
    # re_db_connection = create_db_connection(DB_HOST, DB_USER, DB_PASSWORD, DB_NAME)
    # if re_db_connection:
    #     get_all_users(re_db_connection)
    #     re_db_connection.close()
    # else:
    #     print("Failed to reconnect to database after import.")

    # # Close the final database connection
    # if db_connection:
    #     db_connection.close()
    #     print("\nDatabase connection closed.")

    # print("\nScript finished.")
