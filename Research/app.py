from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Route for the homepage (main page)
@app.route('/')
def home():
    return render_template('main_page.html')

# Route to handle the food selection and generate the food list
@app.route('/generate_recipe', methods=['POST'])
def generate_recipe():
    # Logic for handling the generation (e.g., based on ingredients selected)
    return render_template('food_list.html')

# Route to display the recipe for a selected food
@app.route('/recipe')
def recipe():
    # Get the selected food from the query parameter, default to 'Unknown Food' if not provided
    food_item = request.args.get('food', 'Unknown Food')
    
    # Render the recipe page with the selected food item
    return render_template('recipe_page.html', food=food_item)

if __name__ == '__main__':
    app.run(debug=True)
