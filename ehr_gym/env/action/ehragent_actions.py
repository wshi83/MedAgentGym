"""
The original action space of EHRAgent: https://github.com/wshi83/EhrAgent/tree/main
"""
import os
import pandas as pd
import jsonlines
import json
import re
import sqlite3
import sys
import Levenshtein
from operator import pow, truedic, mul, add, sub
import wolframalpha

def data_filter(data, argument):
    """
    Filter the data based on the argument.

    Example:
        data_filter(data, 'HADM_ID=1076555")
        data_filter(data, "SPEC_TYPE_DESC=peripheral blood lymphocytes")
    """
    backup_data = data
    commands = argument.split('||')
    for i in range(len(commands)):
        try:
            if '>=' in commands[i]:
                command = commands[i].split('>=')
                column_name = command[0]
                value = command[1]
                try:
                    value = type(data[column_name][0])(value)
                except:
                    value = value
                data = data[data[column_name] >= value]
            elif '<=' in commands[i]:
                command = commands[i].split('<=')
                column_name = command[0]
                value = command[1]
                try:
                    value = type(data[column_name][0])(value)
                except:
                    value = value
                data = data[data[column_name] <= value]
            elif '>' in commands[i]:
                command = commands[i].split('>')
                column_name = command[0]
                value = command[1]
                try:
                    value = type(data[column_name][0])(value)
                except:
                    value = value
                data = data[data[column_name] > value]
            elif '<' in commands[i]:
                command = commands[i].split('<')
                column_name = command[0]
                value = command[1]
                if value[0] == "'" or value[0] == '"':
                    value = value[1:-1]
                try:
                    value = type(data[column_name][0])(value)
                except:
                    value = value
                data = data[data[column_name] < value]
            elif '=' in commands[i]:
                command = commands[i].split('=')
                column_name = command[0]
                value = command[1]
                if value[0] == "'" or value[0] == '"':
                    value = value[1:-1]
                try:
                    examplar = backup_data[column_name].tolist()[0]
                    value = type(examplar)(value)
                except:
                    value = value
                data = data[data[column_name] == value]
            elif ' in ' in commands[i]:
                command = commands[i].split(' in ')
                column_name = command[0]
                value = command[1]
                value_list = [s.strip() for s in value.strip("[]").split(',')]
                value_list = [s.strip("'").strip('"') for s in value_list]
                value_list = list(map(type(data[column_name][0]), value_list))
                data = data[data[column_name].isin(value_list)]
            elif 'max' in commands[i]:
                command = commands[i].split('max(')
                column_name = command[1].split(')')[0]
                data = data[data[column_name] == data[column_name].max()]
            elif 'min' in commands[i]:
                command = commands[i].split('min(')
                column_name = command[1].split(')')[0]
                data = data[data[column_name] == data[column_name].min()]
        except:
            if column_name not in data.columns.tolist():
                columns = ', '.join(data.columns.tolist())
                raise Exception("The filtering query {} is incorrect. Please modify the column name or use LoadDB to read another table. The column names in the current DB are {}.".format(commands[i], columns))
            if column_name == '' or value == '':
                raise Exception("The filtering query {} is incorrect. There is syntax error in the command. Please modify the condition or use LoadDB to read another table.".format(commands[i]))
        if len(data) == 0:
            # get 5 examples from the backup data what is in the same column
            column_values = list(set(backup_data[column_name].tolist()))
            if ('=' in commands[i]) and (not value in column_values) and (not '>=' in commands[i]) and (not '<=' in commands[i]):
                levenshtein_dist = {}
                for cv in column_values:
                    levenshtein_dist[cv] = Levenshtein.distance(str(cv), str(value))
                levenshtein_dist = sorted(levenshtein_dist.items(), key=lambda x: x[1], reverse=False)
                column_values = [i[0] for i in levenshtein_dist[:5]]
                column_values = ', '.join([str(i) for i in column_values])
                raise Exception("The filtering query {} is incorrect. There is no {} value in the column. Five example values in the column are {}. Please check if you get the correct {} value.".format(commands[i], value, column_values, column_name))
            else:
                return data
    return data

def get_value(data, argument):
    """
    Extract the specific features of the selected elements in the tabular data.

    Example:
        get_value(data, "CHARTTIME")
    """
    try:
        commands = argument.split(', ')
        if len(commands) == 1:
            column = argument
            while column[0] == '[' or column[0] == "'":
                column = column[1:]
            while column[-1] == ']' or column[-1] == "'":
                column = column[:-1]
            if len(data) == 1:
                return str(data.iloc[0][column])
            else:
                answer_list = list(set(data[column].tolist()))
                answer_list = [str(i) for i in answer_list]
                return ', '.join(answer_list)
        else:
            column = commands[0]
            if 'mean' in commands[-1]:
                res_list = data[column].tolist()
                res_list = [float(i) for i in res_list]
                return sum(res_list)/len(res_list)
            elif 'max' in commands[-1]:
                res_list = data[column].tolist()
                try:
                    res_list = [float(i) for i in res_list]
                except:
                    res_list = [str(i) for i in res_list]
                return max(res_list)
            elif 'min' in commands[-1]:
                res_list = data[column].tolist()
                try:
                    res_list = [float(i) for i in res_list]
                except:
                    res_list = [str(i) for i in res_list]
                return min(res_list)
            elif 'sum' in commands[-1]:
                res_list = data[column].tolist()
                res_list = [float(i) for i in res_list]
                return sum(res_list)
            elif 'list' in commands[-1]:
                res_list = data[column].tolist()
                res_list = [str(i) for i in res_list]
                return list(res_list)
            else:
                raise Exception("The operation {} contains syntax errors. Please check the arguments.".format(commands[-1]))
    except:
        column_values = ', '.join(data.columns.tolist())
        raise Exception("The column name {} is incorrect. Please check the column name and make necessary changes. The columns in this table include {}.".format(column, column_values))

def sql_interpreter(database, command):
    """
    Execute the SQL command in the database.

    Example:
        sql_interpreter(database, "select max(t1.c1) from ( select sum(cost.cost) as c1 from cost where cost.hadm_id in ( select diagnoses_icd.hadm_id from diagnoses_icd where diagnoses_icd.icd9_code = ( select d_icd_diagnoses.icd9_code from d_icd_diagnoses where d_icd_diagnoses.short_title = 'comp-oth vasc dev/graft' ) ) and datetime(cost.chargetime) >= datetime(current_time,'-1 year') group by cost.hadm_id ) as t1")
    """
    con = sqlite3.connect(database)
    cur = con.cursor()
    results = cur.execute(command).fetchall()
    return results

def date_calculator(database, argument):
    """
    Calculate the date in the database based on the argument.

    Example:
        date_calculator("-1 year")
    """
    try:
        con = sqlite3.connect(database)
        cur = con.cursor()
        command = "select datetime(current_time, '{}')".format(argument)
        results = cur.execute(command).fetchall()[0][0]
    except:
        raise Exception("The date calculator {} is incorrect. Please check the syntax and make necessary changes. For the current date and time, please call Calendar('0 year').".format(argument))
    return results

def calculator(query: str):
    """
    Basic calculator to calculate the arithmetic operations.

    Example:
        calculator("1+2")
        calculator("3*4")
        calculator("5/6")
        calculator("7-8")
    """
    operators = {
        '+': add,
        '-': sub,
        '*': mul,
        '/': truediv,
    }
    query = re.sub(r'\s+', '', query)
    if query.isdigit():
        return float(query)
    for c in operators.keys():
        left, operator, right = query.partition(c)
        if operator in operators:
            return round(operators[operator](calculator(left), calculator(right)),2)

def WolframAlphaCalculator(input_query: str):
    """
    Use Wolfram Alpha API to calculate the advanced input query.

    Example:
        WolframAlphaCalculator("max(37.97,76.1)")
    """
    try:
        wolfram_client = wolframalpha.Client(os.environ['WOLFRAM_ALPHA_APPID'])
        res = wolfram_client.query(input_query)
        assumption = next(res.pods).text
        answer = next(res.results).text
    except:
        raise Exception("Invalid input query for Calculator. Please check the input query or use other functions to do the computation.")
    return answer

def run_code(cell):
    """
    Returns the path to the python interpreter.
    """
    try:
        global_var = {"answer": 0}
        exec(cell, global_var)
        cell = "\n".join([line for line in cell.split("\n") if line.strip() and not line.strip().startswith("#")])
        if not 'answer' in cell.split('\n')[-1]:
            return "Please save the answer to the question in the variable 'answer'."
        return str(global_var['answer'])
    except Exception as e:
        error_info = traceback.format_exc()
        code = cell
        if "SyntaxError" in str(repr(e)):
            error_line = str(repr(e))
            
            error_type = error_line.split('(')[0]
            # then parse out the error message
            error_message = error_line.split(',')[0].split('(')[1]
            # then parse out the error line
            error_line = error_line.split('"')[1]
        elif "KeyError" in str(repr(e)):
            code = code.split('\n')
            key = str(repr(e)).split("'")[1]
            error_type = str(repr(e)).split('(')[0]
            for i in range(len(code)):
                if key in code[i]:
                    error_line = code[i]
            error_message = str(repr(e))
        elif "TypeError" in str(repr(e)):
            error_type = str(repr(e)).split('(')[0]
            error_message = str(e)
            function_mapping_dict = {"get_value": "GetValue", "data_filter": "FilterDB", "db_loader": "LoadDB", "sql_interpreter": "SQLInterpreter", "date_calculator": "Calendar"}
            error_key = ""
            for key in function_mapping_dict.keys():
                if key in error_message:
                    error_message = error_message.replace(key, function_mapping_dict[key])
                    error_key = function_mapping_dict[key]
            code = code.split('\n')
            error_line = ""
            for i in range(len(code)):
                if error_key in code[i]:
                    error_line = code[i]
        else:
            error_type = ""
            error_message = str(repr(e)).split("('")[-1].split("')")[0]
            error_line = ""
        # use one sentence to introduce the previous parsed error information
        if error_type != "" and error_line != "":
            error_info = f'{error_type}: {error_message}. The error messages occur in the code line "{error_line}".'
        else:
            error_info = f'Error: {error_message}.'
        error_info += '\nPlease make modifications accordingly and make sure the rest code works well with the modification.'

        return error_info