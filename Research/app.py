from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load and preprocess the dataset
filepath = '/Users/seungwookyoon/FA24-Group6/Research/13k-recipes.csv'
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
    # Get selected ingredients from the cart
    selected_ingredients = request.form.getlist('ingredients')
    user_input = ", ".join(selected_ingredients)
    
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
    return render_template('recipe_page.html', food=food_item)

if __name__ == '__main__':
    app.run(debug=True)
