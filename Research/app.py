from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the recipe data (assuming CSV file is available)
data = pd.read_csv("13k-recipes.csv")
data = data.head(1000)  # First 1000 rows
data = data.drop(columns=['Unnamed: 0', 'Ingredients', 'Image_Name'])  # Remove unwanted columns

# Clean the "Cleaned_Ingredients" column
cleanedIngredients = []
for row in data['Cleaned_Ingredients']:
    ingredientsList = eval(row)  # Convert string representation of list to list
    cleanedList = [ingredient.replace("'", "").strip() for ingredient in ingredientsList]
    cleanedIngredients.append(", ".join(cleanedList))
data['Cleaned_Ingredients'] = cleanedIngredients

# Measurements and fractions to filter out
measurements = ["g", "kg", "lb", "oz", "cup", "cups", "tsp", "tbsp", "tablespoon", "teaspoon", "ml", "l", "liter", "liters", "dash", "pinch", "pound", "ounce"]
fractions = ["½", "¼", "¾", "⅓", "⅔"]

# Clean ingredients again by removing numbers, fractions, and measurements
cleanedIngredientsFinal = []
for row in data['Cleaned_Ingredients']:
    ingredientsList = row.split(", ")  # Split ingredients
    cleanedList = []
    
    for ingredient in ingredientsList:
        words = ingredient.split()  # Split into individual words
        filteredWords = [
            word for word in words
            if not (word.isdigit() or word in fractions or word.lower() in measurements)
        ]
        cleanedList.append(" ".join(filteredWords).strip())
    
    cleanedIngredientsFinal.append(", ".join(cleanedList))

data['Cleaned_Ingredients'] = cleanedIngredientsFinal

# TF-IDF vectorizer for ingredients
vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(data['Cleaned_Ingredients'].tolist())

# Function for meal recommendation
def recommend_meal(userIngredients, data, matrix, vectorizer):
    userVec = vectorizer.transform([userIngredients])  # Vectorize user input
    scores = cosine_similarity(userVec, matrix).flatten()  # Cosine similarity
    topIndices = scores.argsort()[-10:][::-1]  # Top 10 results
    return data.iloc[topIndices][['Title', 'Instructions']]

# Route for the homepage (main page)
@app.route('/')
def home():
    return render_template('main_page.html')

# Route to handle the food selection and generate the food list based on user ingredients
@app.route('/generate_recipe', methods=['POST'])
def generate_recipe():
    user_ingredients = request.form['ingredients']  # Get user input ingredients
    recommendations = recommend_meal(user_ingredients, data, matrix, vectorizer)  # Get recipe recommendations
    return render_template('food_list.html', recipes=recommendations)

# Route to display the recipe for a selected food
@app.route('/recipe')
def recipe():
    food_item = request.args.get('food', 'Unknown Food')  # Get the selected food
    # Render the recipe page (here we assume we have detailed recipe instructions)
    selected_recipe = data[data['Title'] == food_item].iloc[0] if food_item != 'Unknown Food' else None
    return render_template('recipe_page.html', food=selected_recipe)

if __name__ == '__main__':
    app.run(debug=True)
