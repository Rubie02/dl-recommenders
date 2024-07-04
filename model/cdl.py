from keras.models import Model
from keras.layers import Input
from keras.layers import Embedding
from keras.optimizers import Adam
from keras.layers import Concatenate, Dense, Dropout, Lambda, Flatten
import numpy as np

def CustomModel(min_rating, max_rating, n_users, n_products):
    user = Input(shape=(1,))
    u = Embedding(n_users, 50)(user)
    u = Flatten()(u)

    product = Input(shape=(1,))
    m = Embedding(n_products, 50, name="products")(product)
    m = Flatten()(m)

    x = Concatenate()([u, m])
    x = Dropout(0.05)(x)

    x = Dense(128, kernel_initializer='he_normal', activation="relu")(x)

    x = Dense(10, kernel_initializer='he_normal', activation="relu")(x)
    x = Dropout(0.5)(x)

    x = Dense(1, kernel_initializer='he_normal', activation="sigmoid")(x)

    x = Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(x)

    model = Model(inputs=[user, product], outputs=x)
    opt = Adam(learning_rate=0.001)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
    return model

def recommend_products(user_id, num_recommendations, model, df_rating, le_user, le_product):
    # Get all product IDs
    all_product_ids = df_rating['product'].unique()

    # Get the product IDs that the user has already rated
    rated_product_ids = df_rating[df_rating['user'] == user_id]['product'].unique()

    # Get the product IDs that the user has not rated yet
    unrated_product_ids = np.setdiff1d(all_product_ids, rated_product_ids)

    # Create an array of the user ID repeated for the number of unrated products
    user_ids = np.array([user_id] * len(unrated_product_ids))

    # Use the model to predict the ratings for the unrated products for the given user
    predicted_ratings = model.predict([user_ids, unrated_product_ids])

    # Sort the predicted ratings in descending order and get the indices of the top ratings
    top_ratings_indices = predicted_ratings.flatten().argsort()[-num_recommendations:][::-1]

    # Use these indices to get the corresponding product IDs
    recommended_product_ids = unrated_product_ids[top_ratings_indices]

    # Convert the product IDs and user ID back to their original form
    recommended_product_ids = le_product.inverse_transform(recommended_product_ids)
    user_id = le_user.inverse_transform([user_id])

    return user_id[0], recommended_product_ids