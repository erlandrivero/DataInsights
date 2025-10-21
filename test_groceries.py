import requests

url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/groceries.csv"
response = requests.get(url)

lines = response.text.splitlines()
print(f"Total transactions: {len(lines)}")
print(f"\nFirst 5 transactions:")
for i, line in enumerate(lines[:5]):
    items = line.split(',')
    print(f"{i+1}. {items} ({len(items)} items)")

# Calculate avg basket size
total_items = sum(len(line.split(',')) for line in lines)
avg_basket = total_items / len(lines)
print(f"\nAverage basket size: {avg_basket:.1f}")

# Get unique items
all_items = set()
for line in lines:
    all_items.update(line.split(','))
print(f"Unique items: {len(all_items)}")
