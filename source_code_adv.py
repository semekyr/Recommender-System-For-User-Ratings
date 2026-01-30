# %%
import sqlite3
import numpy as np 
import csv
import random


# %%
# Connect to the SQLite database (or create if it does not exist) and create a cursor for executing SQL commands:
con = sqlite3.connect('predictions.db')
cursor = con.cursor()

# Drop any existing tables from previous runs to ensure a clean database state: 
cursor.execute("DROP TABLE IF EXISTS ratings")
cursor.execute("DROP TABLE IF EXISTS testing")
con.commit()
print ("Existing tables dropped successfully.") 

# Create new tables for training ratings and testing data: 
cursor.execute("""
               CREATE TABLE ratings (
                   user_id INTEGER,
                   item_id INTEGER,
                   rating REAL, 
                   fold INTEGER,
                   PRIMARY KEY (user_id, item_id)
                 
               )
               """)

cursor.execute("""
               CREATE TABLE testing (
                   user_id INTEGER,
                   item_id INTEGER,
                   timestamp INTEGER, 
                   PRIMARY KEY (user_id, item_id)
               )
               """)

# Ensure tables are empty by deleting rows (in case they arleady existed and had content): 
cursor.execute("DELETE FROM ratings")
cursor.execute("DELETE FROM testing")
print ("Tables cleared successfully.")

# Commit the changes to the database:
con.commit()

# Load and insert training and testing data from CSV files into the database:
training_data = 'train_20M_withratings.csv'
testing_data = 'test_20M_withoutratings.csv'
k_folds = 5

# Read training CSV line by line and assign a random fold (for k-fold cross-validation):
with open(training_data, 'r') as f:
    my_reader = csv.reader(f)
    for row in my_reader:
        user_id = int(row[0])
        item_id = int(row[1])
        rating = float(row[2])
        fold = random.randint(0, k_folds -1) # Random fold assignments for cross-validation (not used in the final code)
        cursor.execute("INSERT INTO ratings (user_id, item_id, rating, fold) VALUES (?, ?, ?, ?)", (user_id, item_id, rating, fold))
print ("Training data loaded successfully.")

# Read testing data for which predictions are needed to be generated:
with open(testing_data, 'r') as f:
    my_reader = csv.reader(f)
    for row in my_reader:
        user_id = int(row[0])
        item_id = int(row[1])
        timestamp = int(row[2])
        cursor.execute("INSERT INTO testing (user_id, item_id, timestamp) VALUES (?, ?, ?)", (user_id, item_id, timestamp))
print ("Testing data loaded successfully.")

# Commit the changes to the database:
con.commit()
print ("Changes committed successfully.")



# %%
# INPUT: 
# - latent_factors(int): number of dimensions in the latent factor space
# - initial_learning_rate(float): initial learning rate for gradient descent updates
# - decay_rate(float): factor that reduces the learning rate each epoch to allow finer convergence
# - regularization(float): regularization term to prevent overfitting
# - num_epochs(int): number of training epochs
# OUTPUT:
# - Initialized matrix factorization model with internal parameters for training
class MatrixFactorization: 
    # Initialize the model parameters:
    def __init__(self, latent_factors, initial_learning_rate, decay_rate, regularization, num_epochs):

        self.latent_factors = latent_factors
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate
        self.regularization = regularization
        self.num_epochs = num_epochs

        # Latent feature matrix for users:
        self.user_latent_factors = None
        # Latent feature matrix for items:
        self.item_latent_factors = None
        # Bias term for each user:
        self.user_biases = None
        # Bias term for each item:
        self.item_biases = None
        # Global average rating across all training data: 
        self.global_bias = None


    # INPUT: 
    # - db(sqlite3.Connection): SQLite database connection to the ratings table
    # - batch_size(int): number of training samples processed per batch to manage memory
    # - fold(int or None): fold index to exclude from training for cross-validation, or None to use all data
    def fit(self, db, batch_size=1000000, fold = None):
        cursor = db.cursor()
        
        # Global bias initialization:
        if fold is not None:
            cursor.execute("SELECT AVG(rating) FROM ratings WHERE fold != ?", (fold,))
        else:
            cursor.execute("SELECT AVG(rating) FROM ratings")
        self.global_bias = cursor.fetchone()[0]

        # Get the number of users and items:
        cursor.execute("SELECT MAX(user_id) FROM ratings")
        num_users = cursor.fetchone()[0] + 1
        cursor.execute("SELECT MAX(item_id) FROM ratings")
        num_items = cursor.fetchone()[0] + 1

        # Initialize the latent factors and biases: 
        self.user_latent_factors = np.random.normal(loc=0.0, scale=0.01, size=(num_users, self.latent_factors))
        self.item_latent_factors = np.random.normal(loc=0.0, scale=0.01, size=(num_items, self.latent_factors))
        self.user_biases = np.zeros(num_users)
        self.item_biases = np.zeros(num_items)

        # Load training data into memory:
        if fold is not None:
            cursor.execute("SELECT user_id, item_id, rating FROM ratings WHERE fold != ?", (fold,))
        else:
            cursor.execute("SELECT user_id, item_id, rating FROM ratings")
        
        training_set = np.array(cursor.fetchall())
        total_rows = len(training_set)
     


        
        # Start the training loop for a specified number of epochs:
        for i in range (self.num_epochs):

            # Apply learning rate decay to progressively reduce step size:
            # Step decay formula: lr = lr_initial * decay_rate^epoch
            # from: https://www.geeksforgeeks.org/learning-rate-decay/
            current_lr = self.initial_learning_rate * (self.decay_rate ** i)

            print(f"Epoch {i+1}/{self.num_epochs}")

            # Suffle training set to improve SGD stability: 
            np.random.shuffle(training_set)
            
            # Store the mean absolute error for each batch to calculate an avarage for each epoch when all batches are done:
            batch_maes = []

            # Loop through training data in small batches to manage memory:
            for start in range(0, total_rows, batch_size):

                batch = training_set[start:start+batch_size]
                errors = []


                # Compute prediction with biases: 
                # For each (uid, iid, rating) triplet in the current batch:
                for uid, iid, actual_rating in batch: 
                    uid = int(uid)
                    iid = int(iid)
                    try: 
                        # Compute the prediction using: 
                        # r'_ui = μ + b_u + b_i + qᵢ · pᵤ
                        # where: 
                        # μ: global bias 
                        # b_u: user bias 
                        # b_i: item bias
                        # q_i: item latent factor space
                        # p_u: user latent factor space
                        q_item = self.item_latent_factors[iid]
                        p_user = self.user_latent_factors[uid]
                        prediction = (self.global_bias + 
                                 self.user_biases[uid] + 
                                 self.item_biases[iid] + 
                                 np.dot(p_user, q_item))
                        
                        # Compute the prediction error (error = actual_rating - predicted_rating):
                        error = actual_rating - prediction
                    except IndexError:
                        continue
                    
                 
                    # Update global bias per-sample, using gradient descent (same as user & item bias updates):
                    self.global_bias += current_lr * (error - self.regularization * self.global_bias)

                    # Update biases using gradient descent:
                    # To update the bias rules, I applied SGD to a regularised square loss function. 
                    # The biases user_biases (b_u) and item_biases (b_i) account for the tendencies of users and items to deviate from the global average rating. 
                    # Each update adjusts the bias based on the prediction error while also penalizing large values using regularization. 
                    # The resulting update rules are: 
                    # b_u ← b_u + γ(e_ui - λ * b_u)
                    # b_i ← b_i + γ(e_ui - λ * b_i)
                    # where:
                    # γ: the learning rate 
                    # λ: the regularization term
                    # These updates follow the suggestion in slides to include bias terms and the approach for it described in: 
                    # Koren, Y. (2008). "Factorization Meets the Neighborhood: A Multifaceted Collaborative Filtering Model." KDD 2008.
                    self.user_biases[uid] += current_lr * (error - self.regularization * self.user_biases[uid])
                    self.item_biases[iid] += current_lr * (error - self.regularization * self.item_biases[iid])

                    # Gradient computation for user/item latent vectors (as shown in the slides):
                    # p(u, *) += γ * (error * q(i,*) - λ * p(u,*))
                    # q(i, *) += γ * (error * p(u,*) - λ * q(i,*))
                    # where:
                    # γ: the learning rate 
                    # λ: the regularization term
                    user_grad = error * q_item - self.regularization* p_user
                    item_grad = error * p_user - self.regularization * q_item
                    
                    # Clip gradients to ensure stability:
                    clip_val = 0.5
                    user_grad = np.clip(user_grad, -clip_val, clip_val)
                    item_grad = np.clip(item_grad, -clip_val, clip_val)

                    # Apply the updates to the user and item latent vectors:
                    self.user_latent_factors[uid] += current_lr * user_grad
                    self.item_latent_factors[iid] += current_lr * item_grad

                    errors.append(abs(error))

                if errors:
                    # Compute MAE for the current batch and append it to the list for averaging over the entire epoch:
                    batch_mean_error = np.mean(errors)
                    batch_maes.append(batch_mean_error)

            if batch_maes:
                # Print the average MAE for each epoch after completing all batches:
                print(f"\nEpoch {i+1} Avg MAE: {np.mean(batch_maes):.4f}")
    
    # INPUT: 
    # - user_id(int): the ID of the user for whom the rating prediction is being generated
    # - item_id(int): the ID of the item to be rated by the user 
    # OUTPUT: 
    # -predicted_rating(float): the estimated rating value, rounded to the nearest 0.5 and clipped between 1.0 and 5.0
    def predict (self, user_id, item_id):
        try:
            # Predict rating using r'_ui = μ + b_u + b_i + qᵢ · pᵤ (from the slides):
            # where: 
            # μ: global bias 
            # b_u: user bias 
            # b_i: item bias
            # q_i: item latent factor space
            # p_u: user latent factor space
            prediction = (self.global_bias + self.user_biases[user_id] + self.item_biases[item_id] + 
                         np.dot(self.user_latent_factors[user_id], self.item_latent_factors[item_id]))
            
        # If user/item is not in the training data, fall back to the global average:
        except IndexError:
            prediction = round(self.global_bias * 2) / 2
        # Round the predictions and clip the predictions to a valid rating range from the 1.0 to 5.0:
        predicted_rating =  max(1, min(5.0, (round(prediction * 2)/ 2)))
        return predicted_rating
    
    # INPUT: 
    # - db(sqlite3.Connection): database connection used to fetch ratings for vallidation
    # - fold(int or None): if specified, only ratings in this fold are used for validation 
    # OUTPUT:
    # - mae(float): the average absolute difference between predicted and true ratings for the fold
    def validate(self, db, fold=None):
        cursor = db.cursor()
        # Use only the current fold for validation IF specified (when doing cross-validation):
        if fold is not None:
            cursor.execute("SELECT user_id, item_id, rating FROM ratings WHERE fold = ?", (fold,))
        else:
            cursor.execute("SELECT user_id, item_id, rating FROM ratings")
        
        mae = 0
        count = 0
        for user_id, item_id, rating in cursor.fetchall():
            prediction = self.predict(user_id, item_id)
            # Compute the Mean Absolue Error (MAE):
            mae += abs(rating - prediction)
            count += 1
        if count == 0:
            print("No data to validate.")
            return 0
        # Measures the average prediction deviation: 
        mae /= count
        return mae
    
    # INPUT: 
    # - db(sqlite3.Connection)
    # OUTPUT: 
    # - predictions: List[Tuple[int, int, float, int]]: each tuple contains the user ID, item ID, predicted rating and the original timestamps
    def generate_predictions(self, db):
        cursor = db.cursor()
        cursor.execute("SELECT user_id, item_id, timestamp FROM testing")
        test_data = cursor.fetchall()

        predictions = []
        for user_id, item_id, timestamp in test_data:
            # Predict rating for each user-item pairing in the testing data:
            predictions.append((user_id, item_id, self.predict(user_id, item_id), timestamp))
        print (f"Generated {len(predictions)} predictions successfully.")
        return predictions
    

# %%
# Train the final model with the best hyperparameters chosen from a previous grid search(not used currently):
print("Training the final model with best parameters...")
model = MatrixFactorization(latent_factors=100, initial_learning_rate=0.03, decay_rate=0.95, regularization=0.005, num_epochs=4)
model.fit(con, batch_size=1000000, fold=None)
print ("Final model trained successfully.")

# Generate predictions for the testing data:
print("Generating predictions for the testing data...")
predictions = model.generate_predictions(con)
 



# Save predictions to a CSV file named 'results_csv':
# INPUT: 
# - predictions: List[Tuple[int, int, float, int]]: each tuple contains the user ID, item ID, predicted rating and the original timestamps
# OUTPUT: 
# - None: Writes to a CSV file named 'results'
def save_predictions_to_csv(predictions, output_file):
    with open(output_file, 'w', newline='') as f:
        my_writer = csv.writer(f)
        for user_id, item_id, prediction, timestamp in predictions:
            my_writer.writerow([user_id, item_id, prediction, timestamp])

    print(f"Predictions saved to {output_file} successfully.")

output_file = 'results.csv'
save_predictions_to_csv(predictions, output_file)

# Close the database connection:
print ("Closing the database connection...")
con.close()




