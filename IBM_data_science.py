import pandas as pd  # import pandas library
import numpy as np  # import numpy library
import re  # import regular expression library

# Read the CSV file directly using pandas
recipes = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0103EN-SkillsNetwork/labs/Module%202/recipes.csv")

# Display all columns
pd.set_option('display.max_columns', None)

# Show that data is read
print("Data read into dataframe!")

ingredients = list(recipes.columns.values)

print([match.group(0) for ingredient in ingredients for match in [(re.compile(".*(rice).*")).search(ingredient)] if match])
print([match.group(0) for ingredient in ingredients for match in [(re.compile(".*(wasabi).*")).search(ingredient)] if match])
print([match.group(0) for ingredient in ingredients for match in [(re.compile(".*(soy).*")).search(ingredient)] if match])

# Check if 'country' column exists before creating a frequency table
if "country" in recipes.columns:
    print(recipes["country"].value_counts())
else:
    print("'country' column not found!")

# Check if 'cuisine' column exists before proceeding
if "cuisine" in recipes.columns:
    # Convert 'cuisine' column to lowercase
    recipes["cuisine"] = recipes["cuisine"].str.lower()

    # Perform cuisine replacements
    replacements = {
        "austria": "austrian", "belgium": "belgian", "china": "chinese",
        "canada": "canadian", "netherlands": "dutch", "france": "french",
        "germany": "german", "india": "indian", "indonesia": "indonesian",
        "iran": "iranian", "italy": "italian", "japan": "japanese", "israel": "israeli",
        "korea": "korean", "lebanon": "lebanese", "malaysia": "malaysian",
        "mexico": "mexican", "pakistan": "pakistani", "philippines": "philippine",
        "scandinavia": "scandinavian", "spain": "spanish_portuguese", "portugal": "spanish_portuguese",
        "switzerland": "swiss", "thailand": "thai", "turkey": "turkish", "vietnam": "vietnamese",
        "uk-and-ireland": "uk-and-irish", "irish": "uk-and-irish"
    }

    # Update the cuisine column based on the replacements dictionary
    recipes["cuisine"].replace(replacements, inplace=True)

    # Get a list of cuisines to keep based on value counts greater than 50
    recipes_counts = recipes["cuisine"].value_counts()
    cuisines_to_keep = recipes_counts[recipes_counts > 50].index.tolist()

    # Print number of rows before filtering
    rows_before = recipes.shape[0]
    print(f"Number of rows of original dataframe is {rows_before}.")

    # Filter rows based on cuisines to keep
    recipes = recipes.loc[recipes['cuisine'].isin(cuisines_to_keep)]

    # Print number of rows after filtering
    rows_after = recipes.shape[0]
    print(f"Number of rows of processed dataframe is {rows_after}.")

    # Print the number of rows removed
    print(f"{rows_before - rows_after} rows removed!")

    # Replace "Yes" and "No" with 1 and 0, respectively
    recipes = recipes.replace(to_replace="Yes", value=1)
    recipes = recipes.replace(to_replace="No", value=0)

else:
    print("'cuisine' column not found!")

# Output the resulting dataframe
print(recipes)

# Replace "Yes" with 1 and "No" with 0 if necessary
recipes = recipes.replace(to_replace="Yes", value=1)
recipes = recipes.replace(to_replace="No", value=0)

# Group by cuisine and sum the ingredient columns (assuming binary data for ingredients)
ingredient_columns = recipes.columns.drop('cuisine')  # Exclude the 'cuisine' column
cuisines = recipes.groupby("cuisine")[ingredient_columns].sum()

# Display the first few rows of the cuisines dataframe
cuisines.head()

# Number of top ingredients to print
num_ingredients = 4

# Define a function that prints the top ingredients for each cuisine
def print_top_ingredients(row):
    print(row.name.upper())  # Print cuisine name in uppercase
    
    # Sort the ingredients in descending order and convert to percentages
    row_sorted = row.sort_values(ascending=False) * 100
    
    # Get the top N ingredients and their percentages
    top_ingredients = row_sorted.index[:num_ingredients]
    row_sorted_values = row_sorted.values[:num_ingredients]

    # Print each top ingredient and its percentage
    for ind, ingredient in enumerate(top_ingredients):
        print(f"{ingredient} ({int(row_sorted_values[ind])}%)", end=' ')
    print("\n")

# Apply the function to each row of the cuisines dataframe
create_cuisines_profiles = cuisines.apply(print_top_ingredients, axis=1)