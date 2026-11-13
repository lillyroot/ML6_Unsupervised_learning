import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import optuna
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score


def objective(trial, x_train, x_val, y_train, y_val, model_name):
    if model_name == 'Lasso':
        alpha = trial.suggest_float('alpha', 0.01, 1)
        max_iter = trial.suggest_categorical('max_iter', [100, 400, 700, 1000])
        tol = trial.suggest_float('tol', 10 ** -4, 10 ** -2)

        rgr = Lasso(
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            random_state=21
        )

    if model_name == 'Ridge':
        alpha = trial.suggest_float('alpha', 0.01, 1)
        tol = trial.suggest_float('tol', 10 ** -4, 10 ** -2)
        solver = trial.suggest_categorical('solver', ['sag', 'auto'])
        max_iter = trial.suggest_categorical('max_iter', [100, 400, 700, 1000])

        rgr = Ridge(
            alpha=alpha,
            tol=tol,
            solver=solver,
            max_iter=max_iter,
            random_state=21
        )

    if model_name == 'ElasticNet':
        alpha = trial.suggest_float('alpha', 0.01, 1)
        l1_ratio = trial.suggest_float('l1_ratio', 0.01, 1)
        max_iter = trial.suggest_categorical('max_iter', [1000, 1500, 2000])
        tol = trial.suggest_float('tol', 10 ** -4, 10 ** -1)

        rgr = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            tol=tol,
            random_state=21
        )

    if model_name == 'RandomForest':
        n_estimators = trial.suggest_categorical('n_estimators', [100, 300, 500])
        max_depth = trial.suggest_categorical('max_depth', [10, 50, 70])
        min_samples_split = trial.suggest_categorical('min_samples_split', [2, 6, 24])
        min_samples_leaf = trial.suggest_categorical('min_samples_leaf', [1, 4, 12])
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])

        rgr = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=-1,
            random_state=21
        )

    rgr.fit(x_train, y_train)

    preds = rgr.predict(x_val)

    rmse = root_mean_squared_error(y_val, preds)

    return rmse

# models_list: [Lasso, Ridge, ElasticNET, RandomForest]

def study(x_train, x_val, y_train, y_val, model_name, n_trials=10):
    sampler = optuna.samplers.TPESampler(seed=21)
    study = optuna.create_study(directions=['minimize'], sampler=sampler)
    study.optimize(lambda trial: objective(trial, x_train, x_val, y_train, y_val, model_name), n_trials=n_trials, show_progress_bar=True)
    best_trial = study.best_trials[0]
    return model_name, best_trial.value, best_trial.params
    # print(f'Best RF params:\n{best_trial.params}\n')
    # print(f'RMSE RF metric:\n{best_trial.values}')

def draw_scatterplot(data, target, method=None):
    data, target = np.asarray(data), np.asarray(target)
    trg_arg_sort = np.argsort(target)
    target = target[trg_arg_sort]
    data = data[trg_arg_sort]
    if method.lower() in ['umap', 'tsne']:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_title(f"{method}: 2 components")
        ax.scatter(x=data[:, 0], y=data[:, 1], c=target, cmap='viridis', s=2, edgecolors='none')
        ax.set_xlabel('1 comp')
        ax.set_ylabel('2 comp')
        ax.grid(True)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        fig.suptitle(f"{method}: 2 and 3 components")
        axes[0].scatter(x=data[:, 0], y=data[:, 1], c=target, cmap='viridis', s=2, edgecolors='none')
        axes[0].set_xlabel('1 comp')
        axes[0].set_ylabel('2 comp')
        axes[0].grid(True)
        axes[1].remove()
        axes[1] = fig.add_subplot(1, 2, 2, projection='3d')
        axes[1].set_xlabel('1 comp')
        axes[1].set_ylabel('2 comp')
        axes[1].set_zlabel('3 comp')
        axes[1].scatter(xs=data[:, 0], ys=data[:, 1], zs=data[:, 2], c=target, cmap='viridis', s=2, edgecolors='none')
        axes[1].set_box_aspect(None, zoom=0.95)
    fig.tight_layout(pad=-1)
    return

def draw_k_images(comp_img, k, rang_seq):
    if k != len(rang_seq):
        raise Exception(f"rang_seq len != k")
    if k < 1:
        raise Exception(f"k must be positive")
    
    rows = int(np.ceil(k / 3))
    cols = 3

    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    
    axes = axes.flatten()
    U = comp_img.U
    S = comp_img.S
    Vt = comp_img.Vh

    for i in range(k):
        axes[i].imshow(U[:, :rang_seq[i]] @ np.diag(S[:rang_seq[i]]) @ Vt[:rang_seq[i], :], cmap='gray')
        axes[i].set_title(f"k={rang_seq[i]} components")
    fig.tight_layout()
    return

def calculate_metrics(y_train, y_train_pred, y_test, y_test_pred):
    rmse_train = root_mean_squared_error(y_train, y_train_pred)
    rmse_test = root_mean_squared_error(y_test, y_test_pred)

    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    return rmse_train, rmse_test, r2_train, r2_test

def metrics_df_filler(model_n_method: str, timing, y_train, y_train_pred, y_test, y_test_pred, metrics: pd.DataFrame):
    if not metrics['model + method'].isin([model_n_method]).any():
        metrics.loc[len(metrics), :] = [model_n_method, timing, *calculate_metrics(y_train, y_train_pred, y_test, y_test_pred)]
    else:
        mask = metrics['model + method'] == model_n_method
        metrics[mask] = [model_n_method, timing, *calculate_metrics(y_train, y_train_pred, y_test, y_test_pred)]
    return metrics
