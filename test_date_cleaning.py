#!/usr/bin/env python
"""Test date cleaning to identify the corruption issue."""

import pandas as pd
import re

# Simulate the original data
df = pd.DataFrame({
    'InvoiceDate': [
        '12/1/2010 8:26',
        '12/1/2010 8:26',
        '12/1/2010 8:26',
        '12/1/2010 8:26',
        '12/1/2010 8:26'
    ]
})

print("ORIGINAL DATA:")
print(df)
print("\nData types:")
print(df.dtypes)
print()

# Apply the clean_date_string function
def clean_date_string(date_str):
    """Clean malformed date strings before parsing."""
    if pd.isna(date_str) or not isinstance(date_str, str):
        return date_str
    
    # Remove common malformed patterns
    date_str = re.sub(r'(/00)+$', '', date_str)  # Remove trailing /00/00/00
    date_str = re.sub(r'(/0)+$', '', date_str)   # Remove trailing /0/0
    
    # Remove extra slashes
    date_str = re.sub(r'/+', '/', date_str)  # Multiple slashes -> single slash
    
    # Remove trailing/leading slashes
    date_str = date_str.strip('/')
    
    return date_str

# Clean the dates
df['InvoiceDate'] = df['InvoiceDate'].apply(clean_date_string)

print("AFTER CLEANING:")
print(df)
print()

# Parse as datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

print("AFTER PARSING TO DATETIME:")
print(df)
print("\nData types:")
print(df.dtypes)
print()

# Convert back to string (simulating what might happen in display)
df['InvoiceDate_str'] = df['InvoiceDate'].astype(str)

print("AFTER CONVERTING BACK TO STRING:")
print(df)
print()

# Test if the issue is with object dtype conversion
df_test = df.copy()
df_test['InvoiceDate'] = df_test['InvoiceDate'].astype('object')

print("AFTER CONVERTING TO OBJECT DTYPE:")
print(df_test)
