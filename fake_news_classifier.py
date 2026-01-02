# fake_news_classifier.py

import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Ensure NLTK resources are available
try:
    stopwords.words('english')
except LookupError:
    import nltk
    nltk.download('stopwords')

def generate_dummy_dataset():
    data = {
        'title': [
            "Real News: Scientists Discover New Planet",
            "Fake News: Aliens Invade Earth, White House Confirms",
            "Real News: Stock Market Shows Steady Growth",
            "Fake News: Drinking Bleach Cures All Diseases, Doctors Shocked",
            "Real News: Local Elections Conclude Peacefully",
            "Fake News: Secret Society Controls World Governments",
            "Real News: New Study Links Exercise to Longevity",
            "Fake News: Time Travel Invented, But It's a Secret",
            "Real News: Company X Releases Quarterly Earnings Report",
            "Fake News: Bigfoot Sighted in Central Park",
            "Real News: Government Passes New Education Bill",
            "Fake News: Elvis Presley Alive and Well, Living in Hawaii",
            "Real News: Major Breakthrough in Cancer Research",
            "Fake News: Vaccines Cause Zombie Apocalypse",
            "Real News: International Summit Discusses Climate Change",
            "Real News: Global Economy Shows Signs of Recovery",
            "Fake News: Ancient Artifact Discovered on Moon's Surface",
            "Real News: New Study Finds Coffee Benefits Health",
            "Fake News: Famous Celebrity Replaced by Robot Clone",
            "Real News: Tech Giant Unveils Revolutionary New Smartphone",
            "Fake News: Hidden City Found Under Antarctic Ice",
            "Real News: Scientists Warn of Impending Climate Crisis",
            "Fake News: Cure for Baldness Found, Pharmaceutical Companies Suppress It",
            "Real News: Sports Team Wins Championship Title",
            "Fake News: Your Pet Can Now Talk to You Telepathically",
            "Real News: New Law Aims to Reduce Plastic Waste",
            "Fake News: World Leaders Meeting in Secret Bunker",
            "Real News: Researchers Develop Advanced AI System",
            "Fake News: Government Mandates Mind-Reading Devices",
            "Real News: Art Exhibition Opens to Critical Acclaim",
            "Fake News: Zombie Virus Spreads Through Online Game",
            "Real News: Local Community Rallies for Charity Event",
            "Fake News: All Birds Are Government Surveillance Drones",
            "Real News: Breakthrough in Renewable Energy Technology",
            "Fake News: Scientists Prove Earth is Flat",
            "Real News: Major Bank Reports Record Profits",
            "Fake News: Celebrity Endorses Dangerous New Diet Pill",
            "Real News: Education System Undergoes Significant Reforms",
            "Fake News: Global Conspiracy to Control Weather Patterns",
            "Real News: Space Agency Launches New Mars Rover",
            "Fake News: Ancient Aliens Built Pyramids, Evidence Uncovered",
            "Real News: New Restaurant Receives Michelin Star",
            "Fake News: Time Traveler Reveals Future Lottery Numbers",
            "Real News: Medical Breakthrough in Gene Therapy",
            "Fake News: Secret Underground Bases House Extraterrestrials",
            "Real News: Political Leader Delivers Unifying Speech",
            "Fake News: Drinking Ocean Water Cures All Illnesses",
            "Real News: Environmental Activists Protest Deforestation",
            "Fake News: Universal Basic Income to Be Implemented Globally Next Month",
            "Real News: Historic Landmark Restored After Years of Neglect"
        ],
        'text': [
            "Astronomers have announced the discovery of a new exoplanet, Proxima Centauri e, orbiting our nearest star system. The planet is believed to be rocky and potentially habitable, according to a report published today.",
            "BREAKING: Multiple credible sources report that extraterrestrial forces have landed in major cities worldwide. The White House has issued a statement confirming the invasion, urging citizens to remain indoors. Panic spreads.",
            "The stock market continued its upward trend this week, with major indices closing higher for the third consecutive day. Analysts attribute the growth to strong corporate earnings and positive economic indicators.",
            "A groundbreaking new study claims that consuming household bleach can eliminate all known diseases. Medical professionals are reportedly stunned by these findings, which contradict decades of scientific research. (Disclaimer: Do NOT drink bleach.)",
            "Local elections across the state concluded without incident, with preliminary results indicating a high voter turnout. Candidates are awaiting final counts, expected to be released tomorrow morning.",
            "Evidence suggests a shadowy organization, known as 'The Illuminati Reborn,' has been manipulating global events for centuries. Their agenda includes controlling financial markets and political outcomes, sources claim.",
            "A comprehensive study published in the Journal of Medicine provides compelling evidence that regular physical activity significantly extends lifespan. Researchers tracked thousands of participants over 20 years.",
            "Sources deep within a classified government project reveal that time travel technology has been perfected. The invention is being kept under wraps to prevent paradoxes and widespread societal disruption.",
            "Company X announced robust quarterly earnings, exceeding analyst expectations. Revenue increased by 15% year-over-year, driven by strong sales in its software division. Investors are optimistic.",
            "Eyewitnesses in New York City's Central Park reported seeing a large, hairy bipedal creature matching descriptions of Bigfoot. Police are investigating, but no definitive evidence has been found yet.",
            "A bipartisan education bill aimed at improving school funding and teacher salaries has successfully passed both houses of Congress. The bill is expected to be signed into law next week.",
            "Shocking new photos and testimonials confirm that Elvis Presley faked his death and has been living a quiet life in Hawaii for decades. Fans are demanding answers.",
            "Scientists at the Institute for Advanced Research have made a significant discovery in the fight against cancer. Early trials show promising results for a new gene-editing therapy.",
            "URGENT WARNING: Newly leaked documents suggest that common vaccinations are secretly designed to turn recipients into docile, flesh-eating zombies. Protect yourself and your family!",
            "Leaders from over 100 nations gathered today for an international summit on climate change. Discussions focused on new strategies to reduce carbon emissions and promote sustainable practices.",
            "The global economy has shown remarkable resilience, with key indicators pointing towards a steady recovery. International trade has increased, and unemployment rates are declining in several major nations.",
            "An astounding discovery has been made on the lunar surface. Astronauts on a recent mission uncovered what appears to be an ancient, intricately carved artifact, defying all known archaeological records.",
            "New research published in the American Journal of Clinical Nutrition indicates that moderate coffee consumption is associated with a reduced risk of several chronic diseases, including heart disease and type 2 diabetes.",
            "Reports circulating on social media claim that a world-renowned pop star has been replaced by an advanced robot clone. Fans are citing subtle behavioral changes and inconsistencies in recent public appearances.",
            "Tech giant InnovateCorp today unveiled its latest flagship smartphone, featuring a revolutionary holographic display and an AI-powered personal assistant that can anticipate user needs.",
            "A team of explorers using advanced sonar technology has reportedly found evidence of a vast, ancient city hidden beneath the thick ice sheets of Antarctica, sparking widespread scientific interest.",
            "Leading climate scientists have issued a dire warning, stating that without immediate and drastic action, the planet faces irreversible damage from global warming, including extreme weather events and rising sea levels.",
            "Whispers from inside the pharmaceutical industry suggest a miraculous cure for baldness has been developed, but major companies are reportedly suppressing its release to protect their lucrative hair-loss treatment market.",
            "The local Lions football team secured a historic victory in the national championship, ending a decades-long drought. Thousands of jubilant fans celebrated in the streets late into the night.",
            "A new app, 'TelePawth', claims to translate your pet's thoughts directly to your smartphone. Early users report profound conversations with their cats and dogs, though skeptics remain.",
            "A landmark environmental bill has been passed, mandating significant reductions in single-use plastics and promoting sustainable alternatives across industries, effective next year.",
            "Multiple anonymous sources confirm that leaders from the world's most powerful nations are currently holding a top-secret meeting in an undisclosed underground bunker, discussing an urgent global threat.",
            "Researchers at the Artificial Intelligence Institute have successfully developed a new AI system capable of complex problem-solving and creative tasks, setting new benchmarks in the field.",
            "A leaked government document reveals plans for mandatory microchip implants that will allow authorities to read citizens' thoughts, raising widespread privacy concerns and public outcry.",
            "The 'Visions of Tomorrow' art exhibition, featuring immersive digital installations and thought-provoking sculptures, has opened to rave reviews from critics and art enthusiasts alike.",
            "A terrifying new strain of computer virus, dubbed 'ZombieNet', is reportedly spreading through popular online multiplayer games, turning players' characters into uncontrollable, aggressive zombies.",
            "The community of Willow Creek successfully raised over $50,000 for the local children's hospital through a series of events, including a charity run and a bake sale, exceeding their goal.",
            "A viral documentary purports to offer irrefutable proof that all birds are, in fact, sophisticated government surveillance drones, meticulously observing human activity.",
            "A significant breakthrough in fusion energy technology promises to deliver clean, virtually limitless power, potentially solving the global energy crisis within the next decade.",
            "Controversial new 'scientific' findings claim to definitively prove that the Earth is not a sphere, but rather a flat plane, challenging centuries of astronomical understanding.",
            "Global Bank PLC announced unprecedented profits for the last fiscal year, largely due to successful investments in emerging markets and strategic mergers. Shareholders are pleased.",
            "Famed social media influencer 'WellnessGuru' is under fire for promoting a radical new diet pill that promises instant weight loss but has been linked to severe health complications by medical experts.",
            "The national education system is set to undergo sweeping reforms, including a revised curriculum focusing on critical thinking and digital literacy, and increased funding for teacher training.",
            "An elaborate theory gaining traction suggests a covert alliance of powerful individuals is actively manipulating global weather patterns for undisclosed strategic advantages, causing extreme climate events.",
            "NASA successfully launched its most advanced Mars rover to date, equipped with cutting-edge instruments designed to search for signs of ancient microbial life and prepare for human missions.",
            "New archaeological findings near ancient sites globally are being interpreted by some as definitive proof that highly advanced extraterrestrial civilizations visited Earth in antiquity and assisted in building structures like the pyramids.",
            "Chef Antoine Dubois's new Parisian restaurant, 'L'Étoile Cachée,' has been awarded a prestigious Michelin Star within months of its opening, praising its innovative cuisine and impeccable service.",
            "A self-proclaimed time traveler has published a blog post allegedly revealing the winning lottery numbers for the next five years, causing a frenzy among online followers and lottery enthusiasts.",
            "Groundbreaking research in gene editing has led to the successful correction of a genetic defect responsible for a debilitating inherited disease, offering hope for future therapeutic applications.",
            "Whistleblowers from deep within government agencies claim that vast, secret underground facilities exist worldwide, housing advanced technology and even extraterrestrial beings, kept hidden from the public.",
            "In a powerful address to the nation, President Anya Sharma called for unity and cooperation, outlining a vision for a more inclusive society and urging citizens to work together for common goals.",
            "A viral post advocates for consuming pure ocean water as a miraculous cure-all for every ailment, from common colds to chronic diseases, despite strong warnings from health organizations.",
            "Thousands of environmental activists staged a peaceful protest in front of the national parliament, demanding stronger legislation to protect endangered forests from logging and agricultural expansion.",
            "A leaked draft proposal suggests that a universal basic income program will be rolled out globally next month, providing every citizen with a regular income regardless of employment status, aiming to eradicate poverty.",
            "After decades of painstaking restoration work, the ancient Colossus of Rhodes statue has been fully unveiled to the public, drawing tourists and historians from around the globe to witness its grandeur."
        ],
        'label': [
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0
        ]
    }
    df = pd.DataFrame(data)
    return df

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Lowercasing
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)  # Remove punctuation
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text

MODEL_PATH = 'models/'

def train_and_save_models():
    print("Starting model training and saving...")
    df = generate_dummy_dataset()
    df['cleaned_text'] = df['text'].apply(preprocess_text)

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Create models directory if it doesn't exist
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    # Train and save Multinomial Naive Bayes model
    mnb_model = MultinomialNB()
    mnb_model.fit(X_train, y_train)
    joblib.dump(mnb_model, os.path.join(MODEL_PATH, 'mnb_model.joblib'))
    print("Multinomial Naive Bayes model trained and saved.")

    # Train and save Logistic Regression model
    lr_model = LogisticRegression(max_iter=200)
    lr_model.fit(X_train, y_train)
    joblib.dump(lr_model, os.path.join(MODEL_PATH, 'lr_model.joblib'))
    print("Logistic Regression model trained and saved.")

    # Save Vectorizer
    joblib.dump(vectorizer, os.path.join(MODEL_PATH, 'vectorizer.joblib'))
    print("Vectorizer trained and saved.")
    print("Model training and saving complete.")

    # Evaluation (optional, for verification during training)
    mnb_predictions = mnb_model.predict(X_test)
    lr_predictions = lr_model.predict(X_test)

    print("\n--- Training Performance (for verification) ---")
    print("Multinomial Naive Bayes Accuracy:", accuracy_score(y_test, mnb_predictions))
    print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_predictions))


def load_models():
    print("Loading models and vectorizer...")
    try:
        vectorizer = joblib.load(os.path.join(MODEL_PATH, 'vectorizer.joblib'))
        mnb_model = joblib.load(os.path.join(MODEL_PATH, 'mnb_model.joblib'))
        lr_model = joblib.load(os.path.join(MODEL_PATH, 'lr_model.joblib'))
        print("Models and vectorizer loaded successfully.")
        return vectorizer, mnb_model, lr_model
    except FileNotFoundError:
        print("Models not found. Please run train_and_save_models() first.")
        return None, None, None

def predict_news(news_text, vectorizer, mnb_model, lr_model):
    cleaned_text = preprocess_text(news_text)
    vectorized_text = vectorizer.transform([cleaned_text])

    mnb_prediction = mnb_model.predict(vectorized_text)[0]
    lr_prediction = lr_model.predict(vectorized_text)[0]

    return {
        'Multinomial Naive Bayes': 'Fake' if mnb_prediction == 1 else 'Real',
        'Logistic Regression': 'Fake' if lr_prediction == 1 else 'Real'
    }

def main():
    # This main function is primarily for initial setup/testing the training process
    # In a real application, you might run train_and_save_models once to create the models
    # and then use load_models and predict_news in your Streamlit app.
    
    train_and_save_models()

    # Example of loading and predicting
    vectorizer, mnb_model, lr_model = load_models()
    if vectorizer and mnb_model and lr_model:
        sample_news = "BREAKING: Scientists have discovered a cure for all diseases. Local doctors are amazed."
        predictions = predict_news(sample_news, vectorizer, mnb_model, lr_model)
        print(f"\nSample News: '{sample_news}'")
        print(f"Predictions: {predictions}")

if __name__ == "__main__":
    main()