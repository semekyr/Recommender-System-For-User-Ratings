# %%
# Step 1: Load training and testing data
import numpy as np
import csv
def load_data_withoutcsv(training_path, testing_path):
    """"
    Loads the data from the training and testing CSV files
    Parameters:
    training_path: path to the training data file
    testing_path: path to the testing data file

    Returns:
    train_data: a list of lists, where each inner list represents a training instance (user_id, item_id, rating, timestamp) (timestamp is not used)
    test_data: a list of lists, where each inner list represents a testing instance (user_id, item_id, timestamp) (timestamp is not used)
    """
    linestraining = [line.rstrip('\n') for line in open(training_path)]
    train_data = []
    for line in linestraining: 
        words = line.split(',')
        userid = int(words[0])
        itemid = int(words[1])
        rating = float(words[2])
        timestamp = int(words[3])
        train_data.append([userid, itemid, rating, timestamp])
    
    test_data = []
    linestesting = [line.rstrip('\n') for line in open(testing_path)]
    for line in linestesting:
        words = line.split(',')
        userid = int(words[0])
        itemid = int(words[1])
        timestamp = int(words[2])
        test_data.append([userid, itemid, timestamp])

    
    return train_data, test_data




# %%
# Step 2: Build the User-Item Rating Matrix
def build_user_item_matrix(train_data):
    """"
    Builds the user-item rating matrix from the training data.
    Parameters:
    train_data: the processed training data from Step 1

    Returns:
    matrix: a 2D numpy array representing the user-item rating matrix
    user_map: a dictionary that maps user_id to the row index in the matrix
    item_map: a dictionary that maps item_id to the column index in the matrix
    """

    # Create a list of unique users and items and sort them: 
    users = sorted(set([d[0] for d in train_data]))
    items = sorted(set([d[1] for d in train_data]))

    # Create a mapping from user_id and item_id to the row and column index: 
    user_map = {u: i for i, u in enumerate(users)}
    item_map = {i: j for j, i in enumerate(items)}

    # Create and fill the user-item rating matrix:
    matrix = np.zeros((len(users), len(items)))
    for user, item, rating, timestamp in train_data:
        matrix[user_map[user], item_map[item]] = rating
    

    return matrix, user_map, item_map

# %%
# Step 3: Calculate baseline estimates
def calc_baselines(matrix):
    '''
    Calculates the user and item baseline estimates.
    This implementation is based on the formula from the paper: Collaborative Filtering Recommender Systems, Rahul Makhijani, Saleh Samaneh, Megh Mehta
    Formula:
    b_{xi} = μ + b_x + b_i
    where:
    - μ: overall mean rating
    - b_x: rating deviation of user x (user bias) = avg.rating of user x - μ
    - b_i: rating deviation of item i (item bias)
    Parameters:
    matrix: the user-item rating matrix from Step 2

    Returns:
    overall_mean: the overall mean rating
    user_deviations: a numpy array of user deviations from the overall mean
    item_deviations: a numpy array of item deviations from the overall mean
    
    '''
    # Calculate the overall mean rating:
    overall_mean = np.mean(matrix[np.where(matrix != 0)])

    # Calculate user deviations:
    user_means = np.true_divide(matrix.sum(1), (matrix != 0).sum(1))
    user_means[np.isnan(user_means)] = 0
    user_deviations = user_means - overall_mean

    # Calculate item deviations:
    item_means = np.true_divide(matrix.sum(0), (matrix != 0).sum(0))
    item_means[np.isnan(item_means)] = 0
    item_deviations = item_means - overall_mean

    return overall_mean, user_deviations, item_deviations



# %%
# Step 4: Compute Adjusted Cosine Similarity using Baseline Esimtates
# Used shrinkage factor (shrinkage_param) to prevent overfitting
def adjacent_cosine_similarity(matrix, shrinkage_param=15):
    """
    Compute item-item cosine similarity with shrinkage.
    This function calculates the adjusted cosine similarity between items using the user-item rating matrix, incorporating a shrinkage factor to reduce 
    the importance of the similarity of items with few co-ratings.

    Parameters:
    matrix: the user-item rating matrix.
    shrinkage_param: the shrinkage parameter (default=10).
    Returns:
    item_sim: a 2D numpy array representing the item-item cosine similarity matrix.
    """
    num_items = matrix.shape[1]
    item_sim = np.zeros((num_items, num_items))

    # Calculate the mean rating for each user, ignoring missing ratings:
    user_means = np.true_divide(matrix.sum(1), (matrix != 0).sum(1))
    user_means[np.isnan(user_means)] = 0

    # Compute adjusted cosine similarity with shrinkage:
    for item1 in range(num_items):
        for item2 in range(num_items):
            if item1 != item2:
                # Find users who rated both items:
                U_rated = np.where((matrix[:, item1] != 0) & (matrix[:, item2] != 0))[0]
        

                if len(U_rated) > 0:
                    # Center ratings by subtracting user means, to adjust user-specific biases:
                    item1_ratings = matrix[U_rated, item1] - user_means[U_rated]
                    item2_ratings = matrix[U_rated, item2] - user_means[U_rated]

                    # Compute cosine similarity:
                    numerator = np.sum(item1_ratings * item2_ratings)
                    denominator = np.sqrt(np.sum(item1_ratings ** 2)) * np.sqrt(np.sum(item2_ratings ** 2))
                    sim = numerator / denominator if denominator != 0 else 0

                    # Apply shrinkage as similarity based on few ratings is less reliable:
                    # When the number of co-ratings is small, the similarity is shrunk towards 0:
                    # When the number of co-ratings is large, the similarity is not adjusted:
                    shrinkage = len(U_rated) / (len(U_rated) + shrinkage_param)
                    sim *= shrinkage

                    item_sim[item1, item2] = sim

    return item_sim



# %%
# Step 5: Predict Ratings using Baseline Estimates, Item-Item Collaborative Filtering and Confidence Formula
def predict_ratings_with_confidence(userid, itemid, matrix, similarity, user_map, item_map, means, user_means, item_means, k, similarity_threshold):
    """
    Predicts the rating for a given user and item using item-item collaborative filtering with baseline estimates.
    This implementation is based on the formula from the paper: Collaborative Filtering Recommender Systems, Rahul Makhijani, Saleh Samaneh, Megh Mehta

    Formula:
    r_{xi} = b_{xi} + (sum_{j in N(i;x)} s_{ij} * (r_{xj} - b_{xj})) / (sum_{j in N(i;x)} s_{ij})
    where:
    - r_{xi}: predicted rating for user x and item i
    - b_{xi} = μ + b_x + b_i (baseline estimate for user x and item i)
    - μ: overall mean rating
    - b_x: rating deviation of user x
    - b_i: rating deviation of item i
    - s_{ij}: similarity between items i and j
    - N(i; x): set of items rated by user x that are similar to item i

    Parameters:
    - userid: ID of the target user
    - itemid: ID of the target item
    - matrix: user-item rating matrix
    - similarity: item-item similarity matrix
    - user_map: dictionary mapping user IDs to matrix indices
    - item_map: dictionary mapping item IDs to matrix indices
    - means: overall mean rating
    - user_means: user deviations from the overall mean
    - item_means: item deviations from the overall mean
    - k: number of neighbors to consider
    - similarity_threshold: minimum similarity score for an item to be considered a neighbor

    Returns:
    - Predicted rating for the user-item pair, rounded to the nearest integer
    """

    # If the user or item is not in the training data, fallback to the baseline estimate:
    if userid not in user_map or itemid not in item_map:
        return means  


    user_index = user_map[userid]
    item_index = item_map[itemid]

    # If the item has not been rated by any user, fallback to baseline estimate:
    if item_index >= similarity.shape[0]:
        return means 

    item_similarity_scores = similarity[item_index]
    rated_items = np.where(matrix[user_index, :] > 0)[0]

    # If the user has not rated any items, fallback to baseline estimate:
    if len(rated_items) == 0:
        return means + user_means[user_index] + item_means[item_index] 

    valid_rated_items = rated_items[rated_items < similarity.shape[0]]

    # If the user has not rated any items that are in the similarity matrix, fallback to baseline estimate:
    if len(valid_rated_items) == 0:
        return means + user_means[user_index] + item_means[item_index]  

    # Select the top-k most similar items to the target item:
    sorted_items = valid_rated_items[np.argsort(item_similarity_scores[valid_rated_items])[::-1]]
    top_k_items = [item for item in sorted_items if item_similarity_scores[item] > similarity_threshold][:k]

    if len(top_k_items) == 0:
        return means + user_means[user_index] + item_means[item_index]  # Fallback to baseline estimate

    # Weighted average with confidence, adjusted by baseline estimates:
    numerator = 0
    denominator = 0
    for item in top_k_items:
        # Compute the confidence as the square root of the number of ratings for the item:
        # This is done to give more weight to items with more ratings:
        confidence = np.sqrt(np.sum(matrix[:, item] > 0))
        numerator += similarity[item_index, item] * (matrix[user_index, item] - (means + user_means[user_index] + item_means[item])) * confidence
        denominator += similarity[item_index, item] * confidence

    rating = (means + user_means[user_index] + item_means[item_index]) + (numerator / denominator if denominator != 0 else 0)
    # Round the rating to the nearest integer:
    return round(rating)


# %%
# Step 6: Generate Predictions for the Test Data
def gen_preds(test_dataset, matrix, similarity, user_map, item_map, mean, user_means, item_means, output):
    '''
    Generates predictions for the test data and writes them to an output file.
    Parameters:
    test_dataset: the processed testing data from Step 1
    matrix: the user-item rating matrix from Step 2
    similarity: the item-item cosine similarity matrix from Step 3
    user_map: a dictionary that maps user_id to the row index in the matrix
    item_map: a dictionary that maps item_id to the column index in the matrix
    mean: the overall mean rating
    user_means: user deviations from the overall mean
    item_means: item deviations from the overall mean
    output: the path to the output file (results.csv)
    '''
    # Generate predictions for the test data:
    preds = []
    for user, item, timestamp in test_dataset:
        pred = predict_ratings_with_confidence(user, item, matrix, similarity, user_map, item_map, mean, user_means, item_means, k=35, similarity_threshold=0.01)
        pred = min(max(pred, 0.5), 5.0)
        preds.append([user, item, pred, timestamp])
    
    # Write the predictions to the output file:
    with open(output, 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(['userid', 'itemid', 'rating', 'timestamp'])
        writer.writerows(preds)


# %%
# Helper Functions for Error measures (as mentioned in lecture slides)
# 1. Mean Absolute Error (MAE)
def MAE(y_pred, y_true):
    return np.mean(np.abs(np.array(y_pred) - np.array(y_true)))

# 2. Root Mean Squared Error (RMSE)
def RMSE(y_pred, y_true):
    return np.sqrt(np.mean((np.array(y_pred) - np.array(y_true))**2))

# %%
''' After grid search, the best hyperparameters were found to be:
- Shrinkage parameter = 15
- Number of neighbors = 35
- Similarity threshold = 0.01'''
# %%
# Step 7: Once the model is validated, train on the entire dataset and generate predictions
train_dataset =  'train_100k_withratings.csv'
test_dataset =  'test_100k_withoutratings.csv'
# Create the output file:
output = 'results.csv'
train_data, test_data = load_data_withoutcsv(train_dataset, test_dataset)
print('Datasets found, data loaded')
matrix, user_map, item_map = build_user_item_matrix(train_data)
print ('User-item matrix created')
item_similarity = adjacent_cosine_similarity(matrix, shrinkage_param=15)
print('Similarity matrix created')
global_avg, user_bias, item_bias = calc_baselines(matrix)
print('User and item baselines calculated')
gen_preds(test_data, matrix, item_similarity, user_map, item_map, global_avg, user_bias, item_bias, output)
print ('Predictions generated to output file') 



