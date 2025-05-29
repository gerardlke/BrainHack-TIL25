"""
The sole purpose of this script is to assist in Ben's self-play,
where many variables wil be hard-coded to our specific use case.
"""

import sqlite3
from sqlite3 import Error
import os
import argparse
from pathlib import Path

# Default configurations
DEFAULT_DB_FILE = "RL.db"
DB_TABLE = "checkpoints"

class RL_DB:
    def __init__(self, db_file=DEFAULT_DB_FILE, table_name=DB_TABLE):
        self.connection = None
        self.db_file = db_file
        self.table_name = table_name 

    def set_up_db(self):
        """Sets up the database instance within class attributes."""
        self.create_connection()
        self.create_table()

    def shut_down_db(self):
        """Shuts down the database instance."""
        if self.connection:
            self.connection.close()

    # Database connections

    def create_connection(self):
        """Establishes a connection to a SQLite database file and creates the file if it doesn't exist."""
        try:
            self.connection = sqlite3.connect(self.db_file)
            self.connection.row_factory = sqlite3.Row  # Set row_factory to sqlite3.Row for dictionary-like access to rows
            print(f"Connection to SQLite database '{self.db_file}' successful")
        except Error as err:
            print(f"Error connecting to SQLite: '{err}'")

    def execute_query(self, query, params=None):
        """Executes a SQL query."""
        cursor = self.connection.cursor()
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            self.connection.commit()
            return cursor.lastrowid # Return the ID of the last inserted row
        except Error as e:
            print(f"Error executing query: '{e}'")
            return None

    def execute_query_and_return(self, connection, query, params=None):
        """Executes a SQL query and returns the results."""
        cursor = self.connection.cursor()
        result = None
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            result = cursor.fetchall()
        except Error as e:
            print(f"Error reading query: '{e}'")
        return result

    # --- Database Operations ---

    def create_table(self):
        """Creates the checkpoint table in the database if the table doesn't already exist."""
        query = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filepath TEXT NOT NULL,
            policy INT NOT NULL,
            datetime NOT NULL,
            score NOT NULL,
            hyperparameters,
            best_opponents
        );
        """
        execute_query(query)

    def add_checkpoints(checkpoints):
        """Adds multiple checkpoints into checkpoints table."""  # TODO: Can choose to optimise to instantaneous batch insertion 
        for checkpoint in checkpoints:
            self.add_checkpoint(checkpoint)

    def add_checkpoint(self, name, email):  # TODO: Add in all fields once table fields are finalised
        """Adds a new checkpoint to the checkpoints table."""
        query = f"""
        INSERT INTO {DB_TABLE} (name, email)
        VALUES (?, ?);
        """
        last_id = execute_query(query, (name, email))
        if last_id is not None:
            print(f"Added user: {name} ({email}) with ID: {last_id}")
        return last_id
        
    def delete_all_checkpoints(self):
        """Deletes all checkpoints from the table."""
        query = f"DELETE FROM {self.table_name};"
        execute_query(query)
        print(f"Deleted all checkpoints.")

    def delete_checkpoint(self, id)
        """Deletes a checkpoint by id??? from the checkpoints table."""
        query = f"DELETE FROM {self.table_name} WHERE id = ?;"
        execute_query(query, (user_id,))
        print(f"Deleted checkpoint with ID: {user_id}")

    def get_all_checkpoints(self):
        """Retrieves all checkpoints from the checkpoints table."""
        query = f"SELECT * FROM {self.table_name};"
        users = execute_query_and_return(query)
        if users:
            return users
        print("\nNo checkpoints found in the database.\n")

    def get_checkpoint_by_policy(self, policy, shuffle=False):
        """Retrieves checkpoints from the checkpoints table by index."""
        query = f"""
        SELECT * FROM {self.table_name}
        WHERE policy = ?;
        """
        checkpoints = execute_query_and_return(query, (policy,))
        if checkpoints:
            return checkpoints
        print(f"\nNo checkpoints found in the database for policy index {policy}.\n")

    # Database saving

    def export_database_to_sql_dump(self, db_file_path, export_sql_file_path):
        """
        Exports the entire SQLite database schema and data to a .sql dump file.
        This is useful for moving the database to a different SQL system or for backup.
        """
        try:
            conn = sqlite3.connect(db_file_path)
            with open(export_sql_file_path, 'w') as f:
                for line in conn.iterdump():
                    f.write(f'{line}\n')
            conn.close()
            print(f"Database '{db_file_path}' exported successfully to '{export_sql_file_path}'")
        except Error as e:
            print(f"Error exporting database to SQL dump: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during SQL dump export: {e}")


# Sample code
if __name__ == "__main__":
    db = RL_DB(db_file_path=DEFAULT_DB_FILE)

    db.set_up_db(db_file_path)

    sample_data = {}
    db.add_checkpoints(sample_data)

    db.get_all_checkpoints(db_connection)
        
    db.export_database_to_sql_dump(db_file_path, args.export_sql)

    db.shut_down_db()
