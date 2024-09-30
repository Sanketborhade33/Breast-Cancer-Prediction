

import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import gradio as gr

# Load the breast cancer dataset
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

# Convert to DataFrame
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
data_frame['label'] = breast_cancer_dataset.target

# Split the data into features (X) and labels (Y)
X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, Y_train)

# Define the prediction function for Gradio
def predict_breast_cancer(input_data):
    try:
        # Convert the comma-separated input data into a list of floats
        input_data_as_list = [float(i) for i in input_data.split(",")]
        
        # Ensure the correct number of features are provided
        if len(input_data_as_list) != 30:
            return "⚠️ Please provide exactly 30 comma-separated values."
        
        # Convert the list to a numpy array and reshape it
        input_data_as_numpy_array = np.asarray(input_data_as_list).reshape(1, -1)
        
        # Make a prediction
        prediction = model.predict(input_data_as_numpy_array)
        
        # Return the result based on prediction
        if prediction[0] == 0:
            return "🛑 The Breast Cancer is Malignant"
        else:
            return "✅ The Breast Cancer is Benign"
    except ValueError:
        return "⚠️ Error: Please enter valid numerical values separated by commas."

# Create Gradio interface
with gr.Blocks(css=".block-container { max-width: 800px; margin: auto; padding: 20px; background-color: #f9f9f9; border-radius: 10px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); }") as interface:
    # Title and Description
    gr.Markdown("""
    <h1 style="text-align: center; color: #2F4F4F;">Breast Cancer Prediction System 🩺</h1>
    <p style="text-align: center; color: #555;">
        Predict if a breast cancer case is benign or malignant based on 30 features.
    </p>
    <hr style="margin: 20px 0;">
    """)
    
    # Input box with description
    input_box = gr.Textbox(
        label="Enter 30 Features",
        placeholder="Example: 17.99,10.38,122.8,...",
        info="Enter 30 comma-separated values (features) to make a prediction.",
        lines=2,
        max_lines=2,
        interactive=True,
    )
    
    # Predict Button
    with gr.Row():
        predict_button = gr.Button("🔮 Predict", elem_classes="primary-button")

    # Output area
    output = gr.Textbox(label="Prediction", placeholder="Your result will appear here...", lines=2, max_lines=2)
    
    # Click action for the button
    predict_button.click(fn=predict_breast_cancer, inputs=input_box, outputs=output)
    
    # Custom CSS to style buttons and boxes
    gr.Markdown("""
    <style>
        .primary-button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s ease;
            border-radius: 5px;
            display: block;
            margin: auto;
        }
        .primary-button:hover {
            background-color: #45a049;
        }
        .gr-input, .gr-output {
            margin-top: 10px;
            padding: 10px;
            background-color: #f0f0f0;
            border-radius: 5px;
            font-size: 16px;
        }
    </style>
    """)

# Launch the interface
interface.launch()
