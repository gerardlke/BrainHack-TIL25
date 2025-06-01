"""
The sole purpose of this script is to assist in Ben's self-play,
where many variables wil be hard-coded to our specific use case.
"""

import json
import random
import sqlite3
from sqlite3 import Error
from datetime import datetime


# Default configurations
DEFAULT_DB_FILE = "RL.db"
DB_TABLE = "checkpoints"

class RL_DB:
    def __init__(self, db_file=DEFAULT_DB_FILE, table_name=DB_TABLE):
        self.connection = None
        self.db_file = db_file
        self.table_name = table_name 

    def set_up_db(self, timeout=100):
        """Sets up the database instance within class attributes."""
        self.create_connection(timeout=timeout)
        self.create_table()

    def shut_down_db(self):
        """Shuts down the database instance."""
        try:
            if self.connection:
                self.connection.close()
        except Error as e:
            print(f'Error shutting down connection to db: {e}')

    def drop_table(self):
        """
        Drops (deletes) a specified table from the database, including its structure and all data.
        """
        drop_table_query = f"DROP TABLE IF EXISTS {self.table_name};"
        try:
            cursor = self.connection.cursor()
            cursor.execute(drop_table_query)
            self.connection.commit()
            print(f"Table '{self.table_name}' dropped successfully.")
        except Error as err:
            print(f"Error dropping table: '{err}'")

    # Database connections

    def create_connection(self, timeout=100):
        """Establishes a connection to a SQLite database file and creates the file if it doesn't exist."""
        print('Attempting to connect to sqlite with timeout', timeout)
        try:
            self.connection = sqlite3.connect(self.db_file, timeout=timeout)
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
            raise e
            print(f"Error executing query: '{e}'")
            return None

    def execute_query_and_return(self, query, params=None):
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
            filepath TEXT NOT NULL UNIQUE,
            timestamp DATETIME NOT NULL,
            policy_id INTEGER NOT NULL,
            hyperparameters JSON,
            score REAL NOT NULL
        );
        """
        res = self.execute_query(query)
        if res is not None:
            print(f'Table {self.table_name} set up successfully.')
        else:
            print('Failure to create table. Please debug.')

    def add_checkpoints(self, checkpoints):
        """Adds multiple checkpoints into checkpoints table."""  # TODO: Can choose to optimise to instantaneous batch insertion 
        for checkpoint in checkpoints:
            self.add_checkpoint(
                filepath=checkpoint.get('filepath'),
                policy_id=checkpoint.get('policy_id'),
                hyperparameters=checkpoint.get('hyperparameters'),
                score=checkpoint.get('score'),
                # best_opponents=checkpoint.get('best_opponents')
            )

    def add_checkpoint(self, filepath='', policy_id=0, hyperparameters={}, score=0.0):  # TODO: Add in all fields once table fields are finalised
        """Adds a new checkpoint to the checkpoints table."""
        query = f"""
        INSERT INTO {DB_TABLE} (
            filepath, timestamp, policy_id, hyperparameters, score
        )
        VALUES (?, ?, ?, ?, ?);
        """
        last_id = self.execute_query(query, (filepath, datetime.now().isoformat(), policy_id, json.dumps(hyperparameters), score))
        if last_id is not None:
            print(f"Added checkpoint: ({filepath}, {policy_id}, {hyperparameters}, {score})")
        
    def delete_all_checkpoints(self):
        """Deletes all checkpoints from the table."""
        query = f"DELETE FROM {self.table_name};"
        self.execute_query(query)
        print("Deleted all checkpoints.")

    def delete_checkpoint(self, id):
        """Deletes a checkpoint by id??? from the checkpoints table."""
        query = f"DELETE FROM {self.table_name} WHERE id = ?;"
        self.execute_query(query, (id,))
        print(f"Deleted checkpoint with ID: {id}")

    def get_all_checkpoints(self):
        """Retrieves all checkpoints from the checkpoints table."""
        query = f"SELECT * FROM {self.table_name};"
        users = self.execute_query_and_return(query)
        if users:
            return users
        print("\nNo checkpoints found in the database.")

    def get_checkpoint_by_policy(self, policy, shuffle=False):
        """Retrieves single checkpoint from the checkpoints table by index, either best index or ."""
        query = f"""
        SELECT * FROM {self.table_name}
        WHERE policy_id = ?
        ORDER BY score DESC;
        """
        checkpoints = self.execute_query_and_return(query, (policy,))
        if checkpoints:
            if shuffle:
                random.shuffle(checkpoints)
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
    db = RL_DB(db_file=DEFAULT_DB_FILE, table_name=DB_TABLE)

    db.set_up_db()
    db.drop_table()

    db.set_up_db()

    sample_data = [
        {
            'filepath': 'file1.pth',
            'policy_id': 0, 
            'hyperparameters': {'a':2, 'b':1},
            'score': 0.5,
            # 'best_opponents': 'ship'
        },
        {
            'filepath': 'file2.pth',
            'policy_id': 0, 
            'hyperparameters': {'a':1, 'b': 2},
            'score': 0.6,
            # 'best_opponents': 'idk'
        },
        {
            'filepath': 'file3.pth',
            'policy_id': 1, 
            'hyperparameters': {'a':1, 'b': 3},
            'score': 0.5,
            # 'best_opponents': 'um'
        },
    ]
    db.add_checkpoints(sample_data)

    checkpoints = db.get_all_checkpoints()
    for checkpoint in checkpoints:
        print('1 checkpoint', checkpoint['filepath'], checkpoint['policy_id'], checkpoint['hyperparameters'], checkpoint['score'])

    checkpoints = db.get_checkpoint_by_policy(0, shuffle=False)
    for checkpoint in checkpoints:
        print('2 checkpoint', checkpoint['filepath'], checkpoint['policy_id'], checkpoint['hyperparameters'], checkpoint['score'])

    checkpoints = db.get_checkpoint_by_policy(0, shuffle=True)
    for checkpoint in checkpoints:
        print('3 checkpoint', checkpoint['filepath'], checkpoint['policy_id'], checkpoint['hyperparameters'], checkpoint['score'])
        
    # db.export_database_to_sql_dump(db_file_path)

    db.shut_down_db()
