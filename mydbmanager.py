import mysql.connector
from mysql.connector import Error
from tabulate import tabulate

class DatabaseManager:

    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        self.cursor = None

    def switch_database(self, new_database):
        try:
            if self.connection and self.connection.is_connected():
                self.cursor.execute(f"USE {new_database}")
                self.connection.database = new_database
                print(f"Switched to database: {new_database}")
            else:
                print("No active connection. Connect first before switching database.")
        except Error as e:
            print(f"Error switching database: {e}")

    def get_table_count(self):
        query = "SELECT lastno FROM `syst_counters` WHERE descp='tables'"
        result = self.getResults(query, "lastno")
        if result is not None and result:
            return result[0]  # Return the first (and only) result
        else:
            print("Error retrieving table count or no count found")
            return None
        
    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            if self.connection.is_connected():
                self.cursor = self.connection.cursor()
                print("Connected to MySQL database")
            else:
                raise Error("Failed to connect to the database")
        except Error as e:
            print(f"Error: {e}")
            raise

    def create_table(self, table_structure):
        try:
            if not self.connection or not self.connection.is_connected():
                self.connect()

            parts = table_structure.split('#')
            table_name = parts[0]
            fields = parts[1:-1]

            # Check if table already exists
            self.cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
            result = self.cursor.fetchone()

            if result:
                print(f"Table {table_name} already exists.")
                return

            query = f"CREATE TABLE {table_name} ("
            for field in fields:
                name, data_type = field.split('|')
                query += f"{name} {data_type}, "
            query = query[:-2] + ")"

            self.cursor.execute(query)
            self.connection.commit()
            print(f"Table {table_name} created successfully")
        except Error as e:
            print(f"Error creating table: {e}")

    def get_last_insert_id(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT LAST_INSERT_ID()")
        last_id = cursor.fetchone()[0]
        cursor.close()
        return last_id
    
    
    def insert_valuesXXX(self, table_name, values,fldlist):
        try:
            values_list = values.split('#')
            placeholders = ', '.join(['%s'] * len(values_list))
            query = f"INSERT INTO {table_name} ({fldlist}) VALUES ({placeholders})"
            self.cursor.execute(query, values_list)
            self.connection.commit()
            print("Values inserted successfully")
        except Error as e:
            print(f"Error inserting values: {e}")

    def insert_valuesYYY(self, table_name, values, fieldlist):
        fields = fieldlist.split(', ')
        value_list = values.split('#')

        # Create placeholders for the SQL query
        placeholders = ', '.join(['%s'] * len(fields))

        # Construct the SQL query
        query = f"INSERT INTO {table_name} ({fieldlist}) VALUES ({placeholders})"

        try:
            self.cursor.execute(query, value_list)
            self.connection.commit()
            print("Values inserted successfully")
        except mysql.connector.Error as err:
            print(f"Error inserting values: {err}")

    def insert_values(self, table_name, values, fieldlist):
        fields = fieldlist.split(',')
        value_list = values.split('#')

        # Ensure the number of values matches the number of fields
        if len(fields) != len(value_list):
            print(f"Warning: Number of fields ({len(fields)}) does not match number of values ({len(value_list)})")
            # Trim or pad the value_list to match the number of fields
            value_list = value_list[:len(fields)] if len(value_list) > len(fields) else value_list + [''] * (
                        len(fields) - len(value_list))

        # Create placeholders for the SQL query
        placeholders = ', '.join(['%s'] * len(fields))

        # Construct the SQL query
        query = f"INSERT INTO {table_name} ({fieldlist}) VALUES ({placeholders})"

        try:
            self.cursor.execute(query, value_list)
            self.connection.commit()
            print("Values inserted successfully")
        except mysql.connector.Error as err:
            print(f"Error inserting values: {err}")
            print(f"Query: {query}")
            print(f"Values: {value_list}")
    

    def update_table(self, table_name, set_fields, condition):
        try:
            if not self.connection or not self.connection.is_connected():
                self.connect()

            set_parts = []
            for field in set_fields.split(','):
                name, value = field.split('|')
                if value.replace('.', '').isdigit() or value.startswith('-') and value[1:].replace('.', '').isdigit():
                    # This is a numeric value
                    set_parts.append(f"{name} = {value}")
                elif '+' in value or '-' in value or '*' in value or '/' in value:
                    # This is an arithmetic operation
                    set_parts.append(f"{name} = {value}")
                else:
                    # This is a string value
                    set_parts.append(f"{name} = '{value}'")

            set_clause = ', '.join(set_parts)
            query = f"UPDATE {table_name} SET {set_clause} WHERE {condition}"

            self.cursor.execute(query)
            self.connection.commit()
            return self.cursor.rowcount
        except Error as e:
            print(f"Error updating table: {e}")
            return 0

        
    def delete_rows(self, table_name, condition):
        try:
            query = f"DELETE FROM {table_name} WHERE {condition}"
            self.cursor.execute(query)
            self.connection.commit()
            return self.cursor.rowcount
        except Error as e:
            print(f"Error deleting rows: {e}")
            return 0

    def display_delimited(self, table_name, fields, condition):
        try:
            if not self.connection or not self.connection.is_connected():
                self.connect()

            query = f"SELECT {fields} FROM {table_name} WHERE {condition}"
            print(f"Executing query: {query}")  # Debug print

            self.cursor.execute(query)
            results = self.cursor.fetchall()

            print(f"Number of rows returned: {len(results)}")  # Debug print

            if not results:
                print("No data found for the given query.")
                return None

            # Get column names
            column_names = [desc[0] for desc in self.cursor.description]

            # Prepare the header
            header = '|'.join(column_names)

            # Prepare the rows
            rows = []
            for row in results:
                row_str = '|'.join(str(value) for value in row)
                rows.append(row_str)

            # Combine header and rows
            output = header + '#r#' + '#r#'.join(rows)

            return output

        except Error as e:
            print(f"Error displaying data: {e}")
            return None

    def display_data(self, table_name, fields, condition):
        try:
            query = f"SELECT {fields} FROM {table_name} WHERE {condition}"
            self.cursor.execute(query)
            result = self.cursor.fetchall()
            headers = [i[0] for i in self.cursor.description]
            print(tabulate(result, headers=headers, tablefmt="grid"))
        except Error as e:
            print(f"Error displaying data: {e}")

    def execute_query(self, query, fetch=True):
        try:
            if not self.connection or not self.connection.is_connected():
                self.connect()

            print(f"Executing query: {query}")  # Debug print

            self.cursor.execute(query)

            if fetch and self.cursor.description:
                # This is a SELECT query, fetch the results
                results = self.cursor.fetchall()
                print(f"Number of rows returned: {len(results)}")  # Debug print

                if not results:
                    print("No data found for the given query.")
                    return None

                # Get column names
                column_names = [desc[0] for desc in self.cursor.description]

                # Prepare the header
                header = '|'.join(column_names)

                # Prepare the rows
                rows = []
                for row in results:
                    row_str = '|'.join(str(value) for value in row)
                    rows.append(row_str)

                # Combine header and rows
                output = header + '#r#' + '#r#'.join(rows)

                return output
            else:
                # This is an INSERT, UPDATE, or DELETE query
                self.connection.commit()
                print(f"Query executed successfully. Rows affected: {self.cursor.rowcount}")
                return self.cursor.rowcount

        except Error as e:
            print(f"Error executing query: {e}")
            return None

    def execute_sql(self, query):
        cursor = self.connection.cursor()
        try:
            cursor.execute(query)
            self.connection.commit()
        except mysql.connector.Error as err:
            self.connection.rollback()
            raise err
        finally:
            cursor.close()

    def getResults(self, query, fields):
        try:
            if not self.connection or not self.connection.is_connected():
                self.connect()

            self.cursor.execute(query)
            results = self.cursor.fetchall()

            if '|' in fields:
                field_list = fields.split('|')
                return [dict(zip(field_list, row)) for row in results]
            else:
                return [row[0] if row else None for row in results]

        except Error as e:
            print(f"Error executing query: {e}")
            return None

    def close_connection(self):
        if self.connection and self.connection.is_connected():
            self.cursor.close()
            self.connection.close()
            print("MySQL connection closed")