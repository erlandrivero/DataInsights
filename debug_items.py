import requests
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

# Load data
url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/groceries.csv"
response = requests.get(url)

# Parse transactions
transactions = []
for line in response.text.splitlines():
    transactions.append(line.split(','))

print(f"Total transactions: {len(transactions)}")

# Get unique items from raw data
all_items_raw = set()
for trans in transactions:
    all_items_raw.update(trans)
print(f"Unique items (raw): {len(all_items_raw)}")

# Encode with TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

print(f"Unique items (encoded): {len(df_encoded.columns)}")
print(f"\nItems lost: {len(all_items_raw) - len(df_encoded.columns)}")

# Find missing items
missing = all_items_raw - set(df_encoded.columns)
if missing:
    print(f"\nMissing items ({len(missing)}):")
    for item in sorted(missing)[:10]:
        print(f"  - '{item}'")
