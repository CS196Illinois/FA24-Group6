from flask import Flask, render_template, request
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load and preprocess the dataset
script_dir = os.path.dirname(os.path.abspath("Research/13k-recipes.csv"))
filepath = os.path.join(script_dir, "13k-recipes.csv")
data = pd.read_csv(filepath).head(1000)
data = data.drop(columns=['Unnamed: 0', 'Ingredients', 'Image_Name'])

# Clean the "Cleaned_Ingredients" column
measurements = ["g", "kg", "lb", "oz", "cup", "cups", "tsp", "tbsp", "tablespoon", "teaspoon", "ml", "l", "liter", "liters", "dash", "pinch", "pound", "ounce"]
fractions = ["½", "¼", "¾", "⅓", "⅔"]

cleanedIngredientsFinal = []
for row in data['Cleaned_Ingredients']:
    ingredientsList = eval(row)
    cleanedList = [
        " ".join(
            word for word in ingredient.split()
            if not (word.isdigit() or word in fractions or word.lower() in measurements)
        ).strip()
        for ingredient in ingredientsList
    ]
    cleanedIngredientsFinal.append(", ".join(cleanedList))
data['Cleaned_Ingredients'] = cleanedIngredientsFinal

# Prepare TF-IDF vectorizer
vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(data['Cleaned_Ingredients'])

# Function to recommend meals based on user input
def recommend_meal(userIngredients):
    if not userIngredients.strip():
        # Return random recommendations if no ingredients are provided
        return data.sample(15)[['Title', 'Instructions']]
    
    userVec = vectorizer.transform([userIngredients])
    scores = cosine_similarity(userVec, matrix).flatten()
    topIndices = scores.argsort()[-15:][::-1]
    return data.iloc[topIndices][['Title', 'Instructions']]

# Flask Routes
@app.route('/')
def home():
    return render_template('main_page.html')

@app.route('/generate_recipe', methods=['POST'])
def generate_recipe():
    # Collect selected ingredients from the form
    selected_ingredients = request.form.getlist('ingredients')  # Collect hidden inputs

    user_input = ", ".join(selected_ingredients)  # Combine into a single string

    # Get recommended meals
    recommendations = recommend_meal(user_input)

    # Pass recommendations to the food list page
    return render_template(
        'food_list.html',
        recommendations=recommendations.to_dict(orient='records')
    )

@app.route('/recipe')
def recipe():
    food_item = request.args.get('food', 'Unknown Food')
    
    # Find the selected recipe from the dataset
    recipe_data = data[data['Title'] == food_item].iloc[0]
    
    # Convert the row to a dictionary to pass to the template
    recipe_dict = recipe_data.to_dict()
    
    return render_template('recipe_page.html', food=recipe_dict)

if __name__ == '__main__':
    app.run(debug=True)