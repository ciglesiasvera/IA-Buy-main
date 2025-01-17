import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", "app.log")),
        logging.StreamHandler()
    ]
)

# Step 1: Data Collection and Preprocessing
def load_and_preprocess_data():
    try:
        # Define file paths
        buyers_path = os.path.join("data", "buyers.csv")
        offers_path = os.path.join("data", "offers.csv")

        # Load buyer and offer data (example datasets)
        buyers = pd.read_csv(buyers_path)  # Columns: buyer_id, preferences, purchase_history, demographics
        offers = pd.read_csv(offers_path)  # Columns: offer_id, price, category, availability, location

        # Preprocess buyer data
        buyers['preferences'] = buyers['preferences'].fillna('')  # Handle missing preferences
        buyers['demographics'] = buyers['demographics'].fillna('')  # Handle missing demographics

        # Preprocess offer data
        offers['category'] = offers['category'].fillna('')  # Handle missing categories
        offers['location'] = offers['location'].fillna('')  # Handle missing locations

        logging.info("Data loaded and preprocessed successfully.")
        return buyers, offers
    except Exception as e:
        logging.error(f"Error loading or preprocessing data: {e}")
        raise

# Step 2: Feature Engineering
def create_features(buyers, offers):
    try:
        # Combine buyer preferences and demographics into a single feature
        buyers['combined_features'] = buyers['preferences'] + " " + buyers['demographics']

        # Combine offer attributes into a single feature
        offers['combined_features'] = offers['category'] + " " + offers['location']

        # Use TF-IDF to vectorize text features
        tfidf = TfidfVectorizer(stop_words='english')
        buyer_features = tfidf.fit_transform(buyers['combined_features'])
        offer_features = tfidf.transform(offers['combined_features'])

        logging.info("Features created successfully.")
        return buyer_features, offer_features, tfidf
    except Exception as e:
        logging.error(f"Error creating features: {e}")
        raise

# Step 3: Collaborative Filtering (Example: User-Item Matrix)
def collaborative_filtering(buyers, offers):
    try:
        # Create a user-item interaction matrix (example: purchase history)
        interaction_matrix = pd.crosstab(buyers['buyer_id'], offers['offer_id'])

        # Normalize the interaction matrix
        scaler = MinMaxScaler()
        interaction_matrix = scaler.fit_transform(interaction_matrix)

        logging.info("Collaborative filtering completed successfully.")
        return interaction_matrix, scaler
    except Exception as e:
        logging.error(f"Error in collaborative filtering: {e}")
        raise

# Step 4: Hybrid Recommendation System
def hybrid_recommendation(buyer_features, offer_features, interaction_matrix, buyer_id, top_n=5):
    try:
        # Content-based similarity (cosine similarity between buyer and offer features)
        content_similarity = cosine_similarity(buyer_features[buyer_id], offer_features)

        # Collaborative filtering similarity (user-item interactions)
        collaborative_similarity = interaction_matrix[buyer_id]

        # Combine both similarities (weighted average)
        hybrid_similarity = 0.7 * content_similarity + 0.3 * collaborative_similarity

        # Get top N recommendations
        top_offers = np.argsort(hybrid_similarity[0])[-top_n:][::-1]

        logging.info(f"Top {top_n} recommendations generated for buyer {buyer_id}.")
        return top_offers
    except Exception as e:
        logging.error(f"Error generating recommendations: {e}")
        raise

# Step 5: Model Evaluation
def evaluate_model(buyer_features, offer_features, interaction_matrix):
    try:
        # Split data into training and testing sets
        train_data, test_data = train_test_split(interaction_matrix, test_size=0.2, random_state=42)

        # Train a Nearest Neighbors model for evaluation
        model = NearestNeighbors(n_neighbors=5, metric='cosine')
        model.fit(train_data)

        # Evaluate on test data
        precision = 0
        for i in range(test_data.shape[0]):
            recommendations = model.kneighbors(test_data[i].reshape(1, -1), return_distance=False)
            true_positives = len(set(recommendations[0]) & set(np.where(test_data[i] > 0)[0]))
            precision += true_positives / len(recommendations[0])

        precision /= test_data.shape[0]
        logging.info(f"Model evaluation completed. Precision: {precision:.2f}")
        return precision
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        raise

# Main Function
def main():
    try:
        # Step 1: Load and preprocess data
        buyers, offers = load_and_preprocess_data()

        # Step 2: Create features
        buyer_features, offer_features, tfidf = create_features(buyers, offers)

        # Step 3: Collaborative filtering
        interaction_matrix, scaler = collaborative_filtering(buyers, offers)

        # Step 4: Hybrid recommendation for a specific buyer
        buyer_id = 0  # Example buyer ID
        top_offers = hybrid_recommendation(buyer_features, offer_features, interaction_matrix, buyer_id)
        print(f"Top recommendations for buyer {buyer_id}: {top_offers}")

        # Step 5: Evaluate the model
        precision = evaluate_model(buyer_features, offer_features, interaction_matrix)
        print(f"Model Precision: {precision:.2f}")

    except Exception as e:
        logging.error(f"Error in main function: {e}")

if __name__ == "__main__":
    # Ensure the logs directory exists
    os.makedirs("logs", exist_ok=True)
    main()