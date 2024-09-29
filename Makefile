# Makefile for setting up and running the Dash app

# Step 1: Install dependencies using pip
install:
	@echo "Installing dependencies from requirements.txt..."
	pip install -r requirements.txt

# Step 2: Run the web application on localhost:3000
run:
	@echo "Running the Dash app on http://localhost:3000..."
	python app.py
