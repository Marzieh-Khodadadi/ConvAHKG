# Import Libraries
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Dropout, Conv1D, MaxPooling1D,
                                     Flatten, Reshape, UpSampling1D, Concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc, brier_score_loss,
                             cohen_kappa_score, matthews_corrcoef, f1_score, recall_score,
                             accuracy_score, precision_score, roc_curve)
from sklearn.model_selection import StratifiedKFold

print(device_lib.list_local_devices())
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# GPU Configuration
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.05)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)




# Data loading
X_test_0 = np.load('./data/X_me.npy')
Y_test_0 = np.load('./data/Y_me.npy')
X_train_0 = np.load('./data/X.npy')
y_train_0 = np.load('./data/y.npy')

# Define BPR loss
def calculate_class_weights(y_true):
    pos_weight = tf.reduce_sum(1 - y_true) / tf.reduce_sum(y_true)
    neg_weight = 1.0
    return pos_weight, neg_weight

def weighted_binary_crossentropy_loss(y_true, y_pred, pos_weight=None, neg_weight=None):
    if pos_weight is None or neg_weight is None:
        pos_weight, neg_weight = calculate_class_weights(y_true)
    
    y_true = tf.cast(y_true, dtype=tf.float64)
    y_pred = tf.cast(y_pred, dtype=tf.float64)
    
    # Calculate the binary cross-entropy loss
    bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    
    # Apply weights to the loss
    weighted_loss = bce_loss * (y_true * pos_weight + (1 - y_true) * neg_weight)
    
    return tf.reduce_mean(weighted_loss)

def plot_roc_pr_curves(y_true, y_pred, file_prefix):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure(figsize=(12, 6))
    
    # ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='b', label='ROC curve (area = %0.2f)' % roc_auc_score(y_true, y_pred))
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='b', label='Precision-Recall curve (area = %0.2f)' % auc(recall, precision))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    
    plt.savefig(f'{file_prefix}_roc_pr_curve.png')
    plt.close()

# Function to plot ROC and Precision-Recall curves


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, auc

def plot_roc_pr_curves_all_folds(folds_roc, folds_pr, mean_fpr, mean_tpr, mean_recall, mean_precision, file_prefix):
    plt.figure(figsize=(18, 6))
    cmap = plt.get_cmap("tab10")  
    # Plot all ROC Curves
    plt.subplot(1, 2, 1)
    for fold, (fpr, tpr) in enumerate(folds_roc, 1):
        plt.plot(fpr, tpr, lw=1.2, color=cmap(fold % 10), alpha=0.6, label=f'Fold {fold} ROC (AUC = {auc(fpr, tpr):.2f})')
    plt.plot(mean_fpr, mean_tpr, color='black', linestyle='--', lw=1.5, label=f'Mean ROC (AUC = {auc(mean_fpr, mean_tpr):.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')

    # Plot all Precision-Recall Curves
    plt.subplot(1, 2, 2)
    for fold, (precision, recall) in enumerate(folds_pr, 1):
        plt.plot(recall, precision, color=cmap(fold % 10), lw=1.2, alpha=0.6, label=f'Fold {fold} PR (AUC = {auc(recall, precision):.2f})')
    plt.plot(mean_recall, mean_precision, color='black', linestyle='--', lw=1.5, label=f'Mean PR (AUC = {auc(mean_recall, mean_precision):.2f})')
    plt.xlim([-0.05, 1.0])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')

    # Save and close the figure
    plt.savefig(f'{file_prefix}_all_folds_roc_pr_curves.png', bbox_inches='tight')
    plt.close()


def model_history(histories, n):
    for i, history in enumerate(histories):
        plt.figure(figsize=(12, 4))
    
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(f'Model accuracy for fold {i+1}')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
    
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'Model loss for fold {i+1}')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
    
        plt.savefig(f'./output/{n}_fold_{i+1}_accuracy_loss.png')

# Model definition functions
def first_conv_layer(input_tensor):
    x = Reshape((650, 1))(input_tensor)
    x1 = Conv1D(32, 1, activation='relu', padding='same')(x)
    x2 = Conv1D(32, 3, activation='relu', padding='same')(x1)
    x3 = Conv1D(32, 5, activation='relu', padding='same')(x1)
    x4 = MaxPooling1D(pool_size=2)(x)
    x5 = Conv1D(32, 1, activation='relu', padding='same')(x4)
    x5 = UpSampling1D(size=2)(x5)
    x = Concatenate()([x1, x2, x3, x5])
    
    x = Conv1D(32, 5, activation='relu', padding='same')(x)
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = Conv1D(32, 1, activation='relu', padding='same')(x)
    x = Dropout(0.1)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    
    return x

def create_model():
    input1 = Input(shape=(650,))
    input2 = Input(shape=(650,))
    
    x1 = first_conv_layer(input1)
    x2 = first_conv_layer(input2)
    
    concatenated = Concatenate(axis=-1)([x1, x2])
    x = Reshape((int(concatenated.shape[1]), 1))(concatenated)
    x = Flatten()(x)
    
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=[input1, input2], outputs=output)
    return model

from sklearn.metrics import precision_recall_curve

# Precision-Recall plot function
def plot_fold_pr_curves(folds_pr, mean_pr, file_prefix):
    plt.figure(figsize=(10, 8))
    
    for fold_pr in folds_pr:
        plt.plot(fold_pr[1], fold_pr[0], lw=1, alpha=0.7)  # Plot precision-recall curve for each fold
    
    plt.plot(mean_pr[1], mean_pr[0], color='k', linestyle='--', lw=2, label='Mean Precision-Recall')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for All Folds and Mean PR')
    plt.legend(loc='lower left')
    
    plt.savefig(f'{file_prefix}_folds_pr_curves.png')
    plt.close()

def train_and_evaluate(X, y, X_test, Y_test, learning_rate, number, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    histories = []
    fold_roc_curves = []
    fold_pr_curves = []  # Store PR data for each fold
    mean_fpr = np.linspace(0, 1, 650)
    mean_tpr = 0.0
    mean_recall = np.linspace(0, 1, 650)
    mean_precision = 0.0
    y_test_preds = []  # List to store predictions for Y_test from each fold
    fold_metrics = []  # List to store fold metrics

    # Collect results for each fold on Y_test
    y_test_preds = []  # List to store predictions for Y_test from each fold

    for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
        print(f"Training fold {fold}...")
        
        x_train, x_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        l_train, r_train = x_train[:, :650], x_train[:, 650:]
        l_val, r_val = x_val[:, :650], x_val[:, 650:]
        
        model = create_model()
        optimizer = Adam(learning_rate=learning_rate)
        
        # Calculate weights for training data
        pos_weight, neg_weight = calculate_class_weights(y_train)
        
        # Compile with weighted binary cross-entropy loss
        model.compile(loss=lambda y_true, y_pred: weighted_binary_crossentropy_loss(y_true, y_pred, pos_weight, neg_weight),
                      optimizer=optimizer,
                      metrics=['accuracy'])
        
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=3, verbose=1),
            ModelCheckpoint('./output//best_weights.keras', monitor='val_accuracy', save_best_only=True)
        ]
        
        history = model.fit(
            [l_train, r_train], y_train,
            batch_size=128, epochs=100,
            validation_data=([l_val, r_val], y_val),
            callbacks=callbacks
        )
        
        histories.append(history)
                # Test set prediction for current fold
        l_test, r_test = X_test[:, :650], X_test[:, 650:]
        y_test_pred = model.predict([l_test, r_test])
        y_test_preds.append(y_test_pred)  # Collect predictions for Y_test from each fold


        # Compute ROC and PR curve data for Y_test based on the current fold's predictions
        fpr, tpr, _ = roc_curve(Y_test, y_test_pred)
        precision, recall, _ = precision_recall_curve(Y_test, y_test_pred)

        # Store curves for each fold
        fold_roc_curves.append((fpr, tpr))
        fold_pr_curves.append((precision, recall))

        # Accumulate mean values
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
        mean_precision += np.interp(mean_recall, recall[::-1], precision[::-1])  # Reverse for consistent interp

        # Calculate metrics for the fold
        auc_roc = roc_auc_score(Y_test, y_test_pred)
        precision_score_value = precision_score(Y_test, (y_test_pred >= 0.5).astype(int))
        recall_score_value = recall_score(Y_test, (y_test_pred >= 0.5).astype(int))
        accuracy = accuracy_score(Y_test, (y_test_pred >= 0.5).astype(int))
        f1 = f1_score(Y_test, (y_test_pred >= 0.5).astype(int))
        brier = brier_score_loss(Y_test, y_test_pred)
        kappa = cohen_kappa_score(Y_test, (y_test_pred >= 0.5).astype(int))
        mcc = matthews_corrcoef(Y_test, (y_test_pred >= 0.5).astype(int))

        # Collect fold metrics
        fold_metric = {
            'fold': fold,
            'auc-roc': auc_roc,
            'auc-pr': auc(recall, precision),
            'overall_recall': recall_score_value,
            'accuracy': accuracy,
            'precision': precision_score_value,
            'Brier score': brier,
            'Cohen Kappa': kappa,
            'MCC': mcc,
            'F1 score': f1
        }
        fold_metrics.append(fold_metric)

    # Convert the list of fold metrics to a DataFrame
    fold_results_df = pd.DataFrame(fold_metrics)
    
    # Save fold metrics to 'fold_results.csv'
    fold_results_df.to_csv(f'./output/fold_results.csv', index=False)
    print(f"Fold results saved to fold_results.csv")


    # Average the ROC and PR curves
    mean_tpr /= n_splits
    mean_tpr[-1] = 1.0
    mean_precision /= n_splits
    mean_precision[-1] = mean_precision[-2]
    model_history(histories, number)
    # Plot all ROC and PR curves across folds based on test data predictions
    plot_roc_pr_curves_all_folds(fold_roc_curves, fold_pr_curves, mean_fpr, mean_tpr, mean_recall, mean_precision,
                                 f'./output/{number}_test_set')

    # Evaluate on final test set and display/plot metrics
    plot_roc_pr_curves(Y_test, y_test_preds[-1], f'./output/{number}_roc_pr')
    
    auc_roc = roc_auc_score(Y_test, y_test_preds[-1])
    precision, recall, _ = precision_recall_curve(Y_test, y_test_preds[-1])
    auc_pr = auc(recall, precision)
    print(f"auc_roc: {auc_roc}")
    print(f"auc_pr: {auc_pr}")

    overall_recall = recall_score(Y_test, (y_test_preds[-1] >= 0.5).astype(int))
    accuracy = accuracy_score(Y_test, (y_test_preds[-1] >= 0.5).astype(int))
    precision = precision_score(Y_test, (y_test_preds[-1] >= 0.5).astype(int))

    print(f"Overall Recall: {overall_recall}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")

    brier = brier_score_loss(Y_test, y_test_preds[-1])
    print(f"Brier score: {brier}")
    
    y_hat_e = (y_test_preds[-1] >= 0.5).astype(int)
    
    kappa = cohen_kappa_score(Y_test, y_hat_e)
    print(f"Cohen Kappa score: {kappa}")
    
    mcc = matthews_corrcoef(Y_test, y_hat_e)
    print(f"MCC score: {mcc}")
    
    f1 = f1_score(Y_test, y_hat_e)
    print(f"F1 score: {f1}")
    
    return histories, auc_roc, auc_pr, overall_recall, accuracy, precision, brier, kappa, mcc, f1


# Main execution


# Main execution
start_time = time.time()

results_df = pd.DataFrame(columns=["number", 'learning_rate', 'auc-roc', 'auc-pr', 'overall_recall', 'accuracy', 'precision', 'Brier score', 'Cohen Kappa', 'MCC', 'F1 score'])
param_grid = {'learning_rate': [0.0001]}

for i, learning_rate in enumerate(param_grid['learning_rate']):
    histories, auc_roc, auc_pr, overall_recall, accuracy, precision, brier, kappa, mcc, f1 = train_and_evaluate(X_train_0, y_train_0, X_test_0, Y_test_0, learning_rate, i + 1)
    
    # Creating a new DataFrame with the current iteration's results
    new_row = pd.DataFrame({
        "number": [i + 1],
        "learning_rate": [learning_rate],
        'auc-roc': [auc_roc],
        'auc-pr': [auc_pr],
        'overall_recall': [overall_recall],
        'accuracy': [accuracy],
        'precision': [precision],
        'Brier score': [brier],
        'Cohen Kappa': [kappa],
        'MCC': [mcc],
        'F1 score': [f1]
    })
    print(new_row)

    results_df = pd.concat([results_df, new_row], ignore_index=True)

results_df.to_csv("./output/Results_df.csv", index=False)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
