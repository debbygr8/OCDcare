from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.cluster import Birch

app = Flask(__name__)

# Load trained Birch model
def load_model():
    data = pd.read_csv("cleaned_ocd_dataset.csv")
    features = data[[
        'Age',
        'Family History of OCD',
        'Duration of Symptoms (months)',
        'Depression Diagnosis',
        'Anxiety Diagnosis'
    ]].dropna()

    for col in features.columns:
        if features[col].dtype == 'object':
            features[col] = features[col].astype('category').cat.codes

    model = Birch(n_clusters=3)
    model.fit(features.values)
    return model

model = load_model()

def map_severity(label):
    return {0: "Mild", 1: "Moderate", 2: "Severe"}.get(label, "Unknown")

@app.route('/')
def home():
    return render_template('landing.html', image_url="static/images/ocdcare_banner.jpg")

@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        # Extract user input
        age = int(request.form['age'])
        history = 1 if request.form['history'] == 'Yes' else 0
        duration = int(request.form['duration'])
        depression = 1 if request.form['depression'] == 'Yes' else 0
        anxiety = 1 if request.form['anxiety'] == 'Yes' else 0

        selected_subtypes = request.form.getlist('subtypes')

        # Combine data for prediction
        input_data = np.array([[age, history, duration, depression, anxiety]])
        label = model.predict(input_data)[0]
        severity = map_severity(label)
        subtype_result = ",".join(selected_subtypes)

        return redirect(url_for('result', severity=severity, subtype=subtype_result))

    return render_template('form.html')

@app.route('/result')
def result():
    severity = request.args.get('severity', 'Unknown')
    return render_template(
        'result.html',
        severity=severity,
        subtype=request.args.get('subtype', 'None')
    )

@app.route('/recommendation')
def recommendation():
    return render_template('recommendation.html')

@app.route('/about')
def about():
    return render_template('about.html')

# Quiz Page with 30 questions
@app.route('/quiz', methods=['GET', 'POST'])
def quiz():
    questions = [
        "Do you ever get thoughts, images, or urges that keep coming back adainst your will?",
        "Do these thoughts seem strange or unreasonable to you?",
        "can you push them out of your mind,or do they come back even if you try to ignore them?",
        "Do you feel you have to do certain things over and over again-like washing, checking, or repeating words in your head?",
        "Do these actions help you reduce anxiety or discomfort, even if only for a while?",
        "Do they interfere with your work, studies, relationships, or daily routines?",
        "Have you ever felt life was not worth living?",
        "Have you been treated by a psychiatrist or psychologist before?",
        "Have you taken any medications or therapy for this in the past?",
        "Any major illnesses, head injuries, or operations?",
        "Has anyone in your family had mental health problems, anxiety, or OCD-like symptoms?",
        "Do you drink alcohol, smoke, or use drugs like cannabis or stimulants?",
        "Do you re-read things multiple times?",
        "Do you repeat tasks until it feels ‘just right’?",
        "Do you feel guilty for thoughts you can’t control?",
        "Do you avoid certain numbers due to fear?",
        "Do you collect useless items compulsively?",
        "Do you need constant reassurance?",
        "Do you spend hours organizing things?",
        "Do you have difficulty discarding old things or scraps?",
        "Do you tap or touch objects repeatedly?",
        "Do you feel compelled to confess intrusive thoughts?",
        "Do you excessively groom or shower?",
        "Do you have rituals before sleeping?",
        "Do you avoid public places due to contamination fears?",
        "Do you repeatedly pray to neutralize thoughts?",
        "Do you redo simple actions many times?",
        "Do you constantly check health symptoms?",
        "Do you feel extreme discomfort when routines are interrupted?",
        "Do you avoid social activities because of your rituals?"
    ]

    if request.method == 'POST':
        answers = [request.form.get(f'q{i+1}', 'no') for i in range(30)]
        yes_count = answers.count("yes")

        # Map Yes count to severity
        if yes_count <= 10:
            severity = "Mild"
        elif yes_count <= 20:
            severity = "Moderate"
        else:
            severity = "Severe"

        return render_template(
            'quiz_result.html',
            yes=yes_count,
            no=30 - yes_count,
            severity=severity
        )

    return render_template('quiz.html', questions=questions)


if __name__ == "__main__":
    app.run(debug=True)
@app.route('/faq')
def faq():
    return render_template('faq.html')
