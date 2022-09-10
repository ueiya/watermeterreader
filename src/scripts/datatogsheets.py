# datatogsheets.py

# THIS SCRIPT IS INCOMPLETE

# Description

# Opens “data_digested.csv”
# Filters for the last 60 min of data
# Send Liters used to Google sheets

import csv
import datetime

import gspread

# Open CSV file digested_data.csv

# # Filters for the last 60 min of data

# Access Google Sheets with gspread

# Send Liters used to Google sheets - gspread setup but me done manually for this step to work

gc = gspread.service_account(filename="credentials.json")

# Open spreadsheet by key
sh = gc.open_by_key("1SpFfW5fRbZJ-Acs3yePcDEsvWOo88p22K5Y_NP_c26U")

# Open worksheet
wks = sh.worksheet("raw-data")

# Search for a table in the worksheet and append a row to it
wks.append_rows(
    [outputList],
    value_input_option="USER-ENTERED",
    insert_data_option=None,
    table_range=None,
)
# wks.append_rows([outputList]) # Simple append, no extra options
