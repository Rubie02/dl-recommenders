import json
from pymongo import MongoClient
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from random import randrange
import torch
from torch import optim
from torch import nn
from model.ncf import NeutralColabFilteringNet, DatasetBatchIterator, precision_recall_at_k, accuracy_f1_at_k
from model.cdl import CustomModel, recommend_products
import time
import copy
from sklearn.metrics import accuracy_score, f1_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Embedding, Flatten, Dot, Dense
from keras.models import Model, load_model
from keras.saving import save_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, \
                            precision_score, recall_score, accuracy_score, f1_score

def get_evaluate(path):
    with open(path, 'r') as file:
        result = json.load(file)
    return result

def training_process(algorithm_type):
    connection_string = "mongodb+srv://root:GiaMinh0802@cluster0.hrfrhsi.mongodb.net/HMart_v2?retryWrites=true&w=majority"
    client = MongoClient(connection_string)
    db = client.HMart_v2

    list_rating = []
    ratings_col = db["ratings"]
    ratings = ratings_col.find()

    for rating in ratings:
        rating_format = {
            "user": str(rating["user"]),
            "product": str(rating["product"]),
            "star": str(rating["star"])
        }
        list_rating.append(rating_format)
    
    df_rating = pd.DataFrame(list_rating)

    le_user = LabelEncoder()
    le_product = LabelEncoder()

    # Convert the user_id and product_id columns into integers
    df_rating['user'] = le_user.fit_transform(df_rating['user'])
    df_rating['product'] = le_product.fit_transform(df_rating['product'])

    X = df_rating[['user', 'product']]
    Y = df_rating['star'].astype(np.float32)

    random_state = 8
    test_size = 0.2

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    datasets = {'train': (X_train, Y_train), 'test': (X_test, Y_test)}

    if algorithm_type == "ncf":
        ncf(df_rating, datasets, le_user, le_product)
    elif algorithm_type == "dmf":
        dmf(df_rating, X_train, Y_train, X_test, Y_test, le_user, le_product)
    elif algorithm_type == "cdl":
        cdl(df_rating, X_train, Y_train, X_test, Y_test, le_user, le_product)

def ncf(df_rating, datasets, le_user, le_product):
    user_count = df_rating["user"].nunique()
    product_count = df_rating["product"].nunique()
    rating_count = df_rating["star"].count()

    ncf = NeutralColabFilteringNet(user_count, product_count, rating_count)

    ncf.eval()

    ratings_row = randrange(0, df_rating.shape[0]-1)
    test_user = int(df_rating.iloc[ratings_row]["user"])
    test_product = int(df_rating.iloc[ratings_row]["product"])
    actual_rating = int(df_rating.iloc[ratings_row]["star"])

    ncf.to('cpu')
    predicted_rating = ncf(torch.tensor([test_user]), torch.tensor([test_product]), torch.tensor([actual_rating]))

    ncf._init_params()
    ncf.train()

    lr = 1e-3
    wd = 1e-4
    batch_size = 2046
    max_epochs = 50
    early_stop_epoch_threshold = 3

    no_loss_reduction_epoch_counter = 0
    min_loss = np.inf
    min_loss_model_weights = None
    history = []

    min_epoch_number = 1
    epoch_start_time = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ncf.to(device)
    print(f'Device configured: {device}')

    loss_criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(ncf.parameters(), lr=lr, weight_decay=wd)

    training_start_time = time.perf_counter()
    for epoch in range(max_epochs):
        stats = {'epoch': epoch+1, 'total': max_epochs}
        epoch_start_time = time.perf_counter()

        for phase in ('train', 'test'):
            is_training = phase == 'train'
            ncf.train(is_training)
            running_loss = 0.0
            n_batches = 0

            rating_value = int(df_rating['star'].unique()[0])

            # Maybe shuffle can be False (now is True)
            for x_batch, y_batch in DatasetBatchIterator(datasets[phase][0], datasets[phase][1], batch_size=batch_size, shuffle=False):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(is_training):
                    rating_tensor = torch.full((x_batch.size(0),), rating_value, dtype=torch.long)
                    y_pred = ncf(x_batch[:, 0], x_batch[:, 1], rating_tensor)
                    loss = loss_criterion(y_pred, y_batch)
                    if is_training:
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(datasets[phase][0])
            stats[phase] = epoch_loss
            history.append(stats)

            if (phase == 'test'):
                stats['time'] = time.perf_counter() - epoch_start_time
                print(f'Epoch {epoch+1}/{max_epochs} [{phase}] loss: {epoch_loss:.4f} - {stats["time"]:.2f}s')
                if (epoch_loss < min_loss):
                    min_loss = epoch_loss
                    min_loss_model_weights = copy.deepcopy(ncf.state_dict())
                    min_epoch_number = epoch+1
                    no_loss_reduction_epoch_counter = 0
                else:
                    no_loss_reduction_epoch_counter += 1
        if no_loss_reduction_epoch_counter >= early_stop_epoch_threshold:
            print(f'Early stopping at epoch {min_epoch_number}')
            break

    MODEL_PATH = 'model_state/ncf_model.pth'
    torch.save(ncf.state_dict(), MODEL_PATH)

    ncf.load_state_dict(min_loss_model_weights)
    ncf.eval()
    groud_truth, predictions = [], []
    rating_value = int(df_rating['star'].unique()[0])

    with torch.no_grad():
        for x_batch, y_batch in DatasetBatchIterator(datasets['test'][0], datasets['test'][1], batch_size=batch_size, shuffle=False):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            rating_tensor = torch.full((x_batch.size(0),), rating_value, dtype=torch.long)
            outputs = ncf(x_batch[:, 0], x_batch[:, 1], rating_tensor)
            groud_truth.extend(y_batch.tolist())
            predictions.extend(outputs.tolist())

    groud_truth = np.asarray(groud_truth).ravel()
    predictions = np.asarray(predictions).ravel()

    MSE = np.mean((predictions - groud_truth)**2)
    RMSE = np.sqrt(MSE)
    MAE = mean_absolute_error(groud_truth, predictions)
    R2 = r2_score(groud_truth, predictions)
    precision, recall = precision_recall_at_k(groud_truth, predictions, k=10, threshold=3.5)
    accuracy, f1 = accuracy_f1_at_k(groud_truth, predictions, threshold=3.5)

    data = {
        "RMSE": round(float(RMSE), 3),
        "MAE": round(float(MAE), 3),
        "R2": round(float(R2), 3),
        "Precision": round(float(precision), 3),
        "Recall": round(float(recall), 3),
        "Accuracy": round(float(accuracy), 3),
        "F1": round(float(f1), 3)
    }
    with open('evaluate/ncf.json', 'w') as f:
        json.dump({"Neutral Collaborative Filtering": data}, f)

    result = []

    for user_id in range(0, user_count-1):
        all_products = df_rating['product'].unique()

        rated_products = df_rating[df_rating['user'] == user_id]['product'].unique()

        unrated_products = np.setdiff1d(all_products, rated_products)

        product_tensor = torch.tensor(unrated_products)

        user_tensor = torch.tensor([user_id]*len(unrated_products))

        predicted_ratings = ncf(user_tensor, product_tensor, torch.tensor([0]*len(unrated_products)))

        predicted_ratings = predicted_ratings.detach().numpy().tolist()

        df_predicted_ratings = pd.DataFrame({
            'product': unrated_products,
            'predicted_rating': predicted_ratings
        })

        top_10_recommended_products = df_predicted_ratings.sort_values(by='predicted_rating', ascending=False).head(8)

        top_10_recommended_products['product'] = le_product.inverse_transform(top_10_recommended_products['product'])
        original_user_id = le_user.inverse_transform([user_id])[0]

        data = {
            "user_id": original_user_id,
            "products":  top_10_recommended_products['product'].tolist()
        }
        result.append(data)

    # Save the data into a JSON file
    with open('model_state/ncf_predicted.json', 'w') as f:
        json.dump({"data": result}, f)

def dmf(df_rating, X_train, Y_train, X_test, Y_test, le_user, le_product):
    n_users = len(df_rating["user"].unique())
    n_products = len(df_rating["product"].unique())

    embedding_size = 100

    user_input = Input(shape=(1,))
    user_embedding = Embedding(n_users, embedding_size)(user_input)
    user_flat = Flatten()(user_embedding)

    product_input = Input(shape=(1,))
    product_embedding = Embedding(n_products, embedding_size)(product_input)
    product_flat = Flatten()(product_embedding)

    dot_product = Dot(axes=1)([user_flat, product_flat])

    hidden1 = Dense(64, activation='relu')(dot_product)
    hidden2 = Dense(32, activation='relu')(hidden1)

    output = Dense(1)(hidden2)

    model = Model(inputs=[user_input, product_input], outputs=output)

    model.compile(loss='mean_squared_error', optimizer='adam')

    early_stopping = EarlyStopping(patience=4, restore_best_weights=True)
    model.fit([X_train["user"], X_train["product"]], Y_train[0:],
            validation_data=([X_test["user"], X_test["product"]], Y_test[0:]),
            epochs=10, batch_size=1288, callbacks=[early_stopping])
    
    y_pred = model.predict([X_test["user"], X_test["product"]])

    # Flatten the arrays
    y_true = Y_test[0:].values.flatten()
    y_pred = y_pred.flatten()

    # Calculate the metrics
    MSE = mean_squared_error(y_true, y_pred)
    RMSE = np.sqrt(MSE)
    MAE = mean_absolute_error(y_true, y_pred)
    R2 = r2_score(y_true, y_pred)

    # Convert the true and predicted ratings to binary labels
    threshold = 3.5
    y_true_binary = (y_true >= threshold).astype(int)
    y_pred_binary = (y_pred >= threshold).astype(int)

    # Calculate precision, recall, accuracy, and F1 score
    precision = precision_score(y_true_binary, y_pred_binary)
    recall = recall_score(y_true_binary, y_pred_binary)
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    f1 = f1_score(y_true_binary, y_pred_binary)

    data = {
        "RMSE": round(float(RMSE), 3),
        "MAE": round(float(MAE), 3),
        "R2": round(float(R2), 3),
        "Precision": round(float(precision), 3),
        "Recall": round(float(recall), 3),
        "Accuracy": round(float(accuracy), 3),
        "F1": round(float(f1), 3)
    }
    with open('evaluate/dmf.json', 'w') as f:
        json.dump({"Deep Matrix Factorization": data}, f)

    MODEL_PATH = 'model_state/dmf_model.h5'
    save_model(model, MODEL_PATH)

    model = load_model(MODEL_PATH)
    result = []
    # Get all product IDs
    all_product_ids = df_rating['product'].unique()
    for user_id in range(0, n_users-1):
        
        # Get the product IDs that the user has already rated
        rated_product_ids = df_rating[df_rating['user'] == user_id]['product'].unique()

        # Get the product IDs that the user has not rated yet
        unrated_product_ids = np.setdiff1d(all_product_ids, rated_product_ids)

        # Create an array of the user ID repeated for the number of unrated products
        user_ids = np.array([user_id] * len(unrated_product_ids))

        # Use the model to predict the ratings for the unrated products for the given user
        predicted_ratings = model.predict([user_ids, unrated_product_ids])

        # Sort the predicted ratings in descending order and get the indices of the top ratings
        top_ratings_indices = predicted_ratings.flatten().argsort()[-8:][::-1]

        # Use these indices to get the corresponding product IDs
        recommended_product_ids = unrated_product_ids[top_ratings_indices]

        # Convert the product IDs and user ID back to their original form
        recommended_product_ids = le_product.inverse_transform(recommended_product_ids)
        original_user_id = le_user.inverse_transform([user_id])[0]

        data = {
            "user_id": original_user_id,
            "products":  recommended_product_ids.tolist()
        }
        result.append(data)

    with open('model_state/dmf_predicted.json', 'w') as f:
        json.dump({"data": result}, f)

def cdl(df_rating, X_train, Y_train, X_test, Y_test, le_user, le_product):
    user_enc = LabelEncoder()
    df_rating['user'] = user_enc.fit_transform(df_rating['user'].values)
    n_users = df_rating['user'].nunique() + 10

    product_enc = LabelEncoder()
    product_enc.fit(df_rating['product'])
    df_rating['product'] = product_enc.transform(df_rating['product'].values)
    n_products = df_rating['product'].nunique()

    df_rating['star'] = df_rating['star'].values.astype(np.float32)
    min_rating = min(df_rating['star'])
    max_rating = max(df_rating['star'])

    X_train_array = [X_train['user'], X_train.drop('user', axis=1)]
    X_test_array = [X_test['user'], X_test.drop('user', axis=1)]

    model = CustomModel(min_rating, max_rating, n_users, n_products)

    MODEL_PATH = 'model_state/cdl_model.weights.h5'

    callbacks = []
    callbacks.append(EarlyStopping(monitor='val_loss', patience=4))
    callbacks.append(ModelCheckpoint(MODEL_PATH,  monitor='val_loss', save_best_only=True, save_weights_only=True))

    history = model.fit(x=X_train_array, y=Y_train, batch_size=128, epochs=50,
                    verbose=1, validation_data=(X_test_array, Y_test),
                    callbacks=callbacks)

    model.load_weights(MODEL_PATH)
        # Predict the ratings
    y_pred = model.predict(X_test_array)

    # Flatten the arrays
    y_true = Y_test.values.flatten()
    y_pred = y_pred.flatten()

    # Calculate the metrics
    MSE = mean_squared_error(y_true, y_pred)
    RMSE = np.sqrt(MSE)
    MAE = mean_absolute_error(y_true, y_pred)
    R2 = r2_score(y_true, y_pred)

    # Convert the true and predicted ratings to binary labels
    threshold = 3.5
    y_true_binary = (y_true >= threshold).astype(int)
    y_pred_binary = (y_pred >= threshold).astype(int)

    # Calculate precision, recall, accuracy, and F1 score
    precision = precision_score(y_true_binary, y_pred_binary)
    recall = recall_score(y_true_binary, y_pred_binary)
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    f1 = f1_score(y_true_binary, y_pred_binary)
    data = {
        "RMSE": round(float(RMSE), 3),
        "MAE": round(float(MAE), 3),
        "R2": round(float(R2), 3),
        "Precision": round(float(precision), 3),
        "Recall": round(float(recall), 3),
        "Accuracy": round(float(accuracy), 3),
        "F1": round(float(f1), 3)
    }
    with open('evaluate/cdl.json', 'w') as f:
        json.dump({"Custom Deep Learning": data}, f)

    result = []
    for user_id in range(0, n_users-10):
        user_id, recommended_product_ids = recommend_products(user_id, 8, model, df_rating, le_user, le_product)

        data = {
            "user_id": user_id,
            "products":  recommended_product_ids.tolist()
        }
        result.append(data)

    with open('model_state/cdl_predicted.json', 'w') as f:
        json.dump({"data": result}, f)