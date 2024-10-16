from flask import Flask, render_template

app = Flask(__name__)

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle the recipe generation
@app.route('/generate_recipe', methods=['POST'])
def generate_recipe():
    # You can add logic here if needed to process ingredients
    return render_template('food_list.html')

if __name__ == '__main__':
    app.run(debug=True)
