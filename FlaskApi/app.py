from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# 🔹 Load Trained Models
model_paths = {
    "user_based": r"C:/Users/sivay/CSC 577 RECOMENDED SYSTEMS/project/trained models/user_based_cf.pkl",
    "item_based": r"C:/Users/sivay/CSC 577 RECOMENDED SYSTEMS/project/trained models/item_based_cf.pkl",
    "cbf": r"C:/Users/sivay/CSC 577 RECOMENDED SYSTEMS/project/trained models/cbf_tfidf_model.pkl",
    "svd": r"C:/Users/sivay/CSC 577 RECOMENDED SYSTEMS/project/trained models/svd_model.pkl",
}

models = {}

# Load all models
for model_name, path in model_paths.items():
    with open(path, "rb") as file:
        models[model_name] = pickle.load(file)

user_sim_df = models["user_based"]  # User-Based CF model
item_sim_df = models["item_based"]  # Item-Based CF model

# 🔹 Fix CBF Model Loading (Handle `csr_matrix` & Compute Cosine Similarity)
cbf_data = models["cbf"]
if isinstance(cbf_data, tuple) and len(cbf_data) == 2:
    tfidf_matrix, product_ids = cbf_data  # Unpack sparse matrix & product IDs
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=product_ids)  # Convert to DataFrame
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)  # Compute cosine similarity
    print("✅ Successfully computed cosine similarity for CBF model!")
else:
    raise ValueError("The loaded CBF model does not contain both matrix and index mapping.")

svd_model = models["svd"]  # SVD-based recommendation model

# 🔹 Print available IDs for debugging
print("✅ Available User IDs in User-Based CF:", user_sim_df.index[:10].tolist())
print("✅ Available Item IDs in Item-Based CF:", item_sim_df.index[:10].tolist())
print("✅ Available Product IDs in CBF Model:", tfidf_df.index[:10].tolist())  # Fixed issue
print("✅ Available User IDs in SVD Model:", svd_model.index[:10].tolist())

# 🔹 Function to Generate User-Based CF Recommendations
def recommend_user_based(user_id, num_recommendations=5):
    try:
        if user_id not in user_sim_df.index:
            return {"error": f"User ID '{user_id}' not found in User-Based CF model"}
        similar_users = user_sim_df.loc[user_id].sort_values(ascending=False).index[:num_recommendations]
        return similar_users.tolist()
    except Exception as e:
        return {"error": str(e)}

# 🔹 Function to Generate Item-Based CF Recommendations
def recommend_item_based(item_id, num_recommendations=5):
    try:
        if item_id not in item_sim_df.index:
            return {"error": f"Item ID '{item_id}' not found in Item-Based CF model"}
        similar_items = item_sim_df.loc[item_id].sort_values(ascending=False).index[:num_recommendations]
        return similar_items.tolist()
    except Exception as e:
        return {"error": str(e)}

# 🔹 Function to Generate Content-Based Filtering (CBF) Recommendations
def recommend_cbf(product_id, num_recommendations=5):
    try:
        # ✅ Convert Index to List to Avoid Ambiguity
        product_ids_list = tfidf_df.index.tolist()

        if product_id not in product_ids_list:
            return {"error": f"Product ID '{product_id}' not found in Content-Based model"}

        # Get similarity scores
        product_index = product_ids_list.index(product_id)
        similarity_scores = cosine_sim[product_index]  # Extract row of similarity scores

        # Sort and get top similar items
        top_indices = similarity_scores.argsort()[-(num_recommendations+1):-1][::-1]
        recommended_products = [product_ids_list[i] for i in top_indices]

        return recommended_products
    except Exception as e:
        return {"error": str(e)}


# 🔹 Function to Generate SVD-Based Recommendations
def recommend_svd(user_id, num_recommendations=5):
    try:
        if user_id not in svd_model.index:
            return {"error": f"User ID '{user_id}' not found in SVD model"}
        recommended_items = svd_model.loc[user_id].sort_values(ascending=False).index[:num_recommendations]
        return recommended_items.tolist()
    except Exception as e:
        return {"error": str(e)}

# 🔹 API Endpoint for Recommendations
@app.route('/recommend', methods=['GET'])
def get_recommendations():
    user_id = request.args.get('user_id')
    method = request.args.get('method', 'user_based')  # Default to user-based

    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    print(f"📌 Received User ID: {user_id}, Method: {method}")

    if method == "user_based":
        recommendations = recommend_user_based(user_id)
    elif method == "svd":
        recommendations = recommend_svd(user_id)
    else:
        return jsonify({"error": "Invalid recommendation method"}), 400

    return jsonify({"user_id": user_id, "method": method, "recommendations": recommendations})

# 🔹 API Endpoint for Item-Based CF Recommendations
@app.route('/recommend_item', methods=['GET'])
def get_item_recommendations():
    item_id = request.args.get('item_id')

    if not item_id:
        return jsonify({"error": "Item ID is required"}), 400

    print(f"📌 Received Item ID: {item_id}")

    recommendations = recommend_item_based(item_id)
    return jsonify({"item_id": item_id, "recommendations": recommendations})

# 🔹 API Endpoint for Content-Based Filtering (CBF) Recommendations
@app.route('/recommend_cbf', methods=['GET'])
def get_cbf_recommendations():
    product_id = request.args.get('product_id')  # Extract product_id from request

    if not product_id:
        return jsonify({"error": "Product ID is required"}), 400  # Return error if no product_id

    print(f"📌 Received Product ID: {product_id}")  # Debugging

    recommendations = recommend_cbf(product_id)  # Call recommend_cbf with correct argument
    return jsonify({"product_id": product_id, "recommendations": recommendations})



# 🔹 API Endpoint for Popular Items (Static Example)
@app.route('/popular', methods=['GET'])
def get_popular_items():
    popular_items = ["B001E4KFG0", "B002QYW8LW", "B003LSTD38"]  # Replace with real data
    return jsonify({"popular_items": popular_items})

# 🔹 Run Flask App
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)
