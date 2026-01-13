from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, StratifiedKFold

from .config import GRIDSEARCH_CV_SPLITS


def _binary_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float('nan')


def _get_model_scores(model: Any, X: np.ndarray):
    if hasattr(model, 'predict_proba'):
        try:
            p = model.predict_proba(X)
            if p is None:
                return None
            p = np.asarray(p)
            if p.ndim == 2 and p.shape[1] >= 2:
                return p[:, 1]
        except Exception:
            pass

    if hasattr(model, 'decision_function'):
        try:
            s = model.decision_function(X)
            if s is None:
                return None
            return np.asarray(s).reshape(-1)
        except Exception:
            pass

    return None


def _grid_search_cv_auc(estimator, param_grid, X, y, *, cv_splits=3, random_state=42, verbose=0, tag='GridSearch'):
    X = np.asarray(X)
    y = np.asarray(y).astype(int)

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    best_score = -np.inf
    best_params = {}
    best_cv_scores = None

    grid_list = list(ParameterGrid(param_grid))
    if verbose:
        print(f'[{tag}] combos={len(grid_list)} cv_splits={cv_splits}')

    for gi, params in enumerate(grid_list, start=1):
        if verbose:
            print(f'[{tag}] {gi}/{len(grid_list)} params={params}')
        scores = []
        for tr_idx, va_idx in cv.split(X, y):
            m = clone(estimator)
            m.set_params(**params)
            m.fit(X[tr_idx], y[tr_idx])
            y_score = _get_model_scores(m, X[va_idx])
            fold_auc = _binary_auc_score(y[va_idx], y_score) if y_score is not None else float('nan')
            scores.append(fold_auc)

            if verbose >= 2:
                print(f'[{tag}] fold_auc={fold_auc}')

        scores_arr = np.asarray(scores, dtype=float)
        score = float(np.nanmean(scores_arr))
        if score > best_score:
            best_score = score
            best_params = dict(params)
            best_cv_scores = scores_arr

            if verbose:
                print(f'[{tag}] best_auc_cv_mean={best_score} best_params={best_params}')

    best_est = clone(estimator)
    best_est.set_params(**best_params)
    best_est.fit(X, y)

    return best_est, {
        'best_auc_cv_mean': float(best_score),
        'best_params': best_params,
        'best_auc_cv_folds': None if best_cv_scores is None else [float(x) for x in best_cv_scores],
        'cv_splits': int(cv_splits),
    }


def train_models_doananhvu(X_train, y_train, X_test, y_test, random_state=42, cv_splits=None, verbose=None) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}

    cv_splits = GRIDSEARCH_CV_SPLITS if cv_splits is None else int(cv_splits)
    verbose = 0 if verbose is None else int(verbose)

    # ANN (MLP)
    base_mlp = MLPClassifier(
        activation='relu',
        solver='adam',
        early_stopping=True,
        random_state=random_state,
    )
    best_mlp, gs_mlp = _grid_search_cv_auc(
        base_mlp,
        {
            'hidden_layer_sizes': [(128,), (256, 128)],
            'alpha': [1e-4, 1e-3],
            'max_iter': [350, 500],
        },
        X_train,
        y_train,
        cv_splits=cv_splits,
        random_state=random_state,
        verbose=verbose,
        tag='DoanAnhVu/MLP',
    )

    y_pred = best_mlp.predict(X_test)
    try:
        y_prob = best_mlp.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob = None

    results['DoanAnhVu - ANN/MLP (Bandpower)'] = {
        'y_pred': y_pred,
        'y_prob': y_prob,
        'model_obj': best_mlp,
        'meta': {'type': 'sklearn', 'gridsearch': gs_mlp},
    }

    # XGBoost (optional)
    try:
        from xgboost import XGBClassifier

        base_xgb = XGBClassifier(
            random_state=random_state,
            eval_metric='logloss',
        )
        best_xgb, gs_xgb = _grid_search_cv_auc(
            base_xgb,
            {
                'n_estimators': [300, 500],
                'max_depth': [4, 6],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
            },
            X_train,
            y_train,
            cv_splits=cv_splits,
            random_state=random_state,
            verbose=verbose,
            tag='DoanAnhVu/XGB',
        )

        y_pred = best_xgb.predict(X_test)
        y_prob = best_xgb.predict_proba(X_test)[:, 1]

        results['DoanAnhVu - XGBoost (Bandpower)'] = {
            'y_pred': y_pred,
            'y_prob': y_prob,
            'model_obj': best_xgb,
            'meta': {'type': 'xgboost', 'gridsearch': gs_xgb},
        }

    except Exception:
        pass

    return results


def train_efficientnet_tl_grid(X_spec_train, y_train, X_spec_test, y_test, random_state=42, verbose=None) -> Optional[Dict[str, Any]]:
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        tf.random.set_seed(random_state)

        Xtr = X_spec_train.astype(np.float32)
        Xte = X_spec_test.astype(np.float32)

        vmin = float(Xtr.min())
        vmax = float(Xtr.max())
        Xtr_01 = (Xtr - vmin) / (vmax - vmin + 1e-6)
        Xte_01 = (Xte - vmin) / (vmax - vmin + 1e-6)

        Xtr_255 = Xtr_01 * 255.0
        Xte_255 = Xte_01 * 255.0

        def _prep(X_255):
            X = tf.convert_to_tensor(X_255, dtype=tf.float32)
            X = tf.image.resize(X, (224, 224), method='bilinear')
            X = tf.repeat(X, repeats=3, axis=-1)
            return X

        Xtr_img = _prep(Xtr_255)
        Xte_img = _prep(Xte_255)

        verbose = 0 if verbose is None else int(verbose)
        fit_verbose = 1 if verbose else 0

        grid = [
            {'lr_head': 1e-3, 'lr_ft': 1e-5, 'unfreeze_last': 30, 'dropout': 0.4},
            {'lr_head': 5e-4, 'lr_ft': 1e-5, 'unfreeze_last': 40, 'dropout': 0.5},
        ]

        best_auc = -np.inf
        best_cfg = None
        best_model = None

        for gi, cfg in enumerate(grid, start=1):
            if verbose:
                print(f'[DoanAnhVu/EfficientNet] {gi}/{len(grid)} cfg={cfg}')
            base = keras.applications.EfficientNetB0(
                include_top=False,
                weights='imagenet',
                input_shape=(224, 224, 3),
            )
            base.trainable = False

            inp = keras.Input(shape=(224, 224, 3))
            x = keras.applications.efficientnet.preprocess_input(inp)
            x = base(x, training=False)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dropout(cfg['dropout'])(x)
            out = layers.Dense(1, activation='sigmoid')(x)
            model = keras.Model(inp, out)

            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=cfg['lr_head']),
                loss='binary_crossentropy',
                metrics=[keras.metrics.AUC(name='auc'), keras.metrics.BinaryAccuracy(name='acc')],
            )

            cb = [keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=5, restore_best_weights=True)]

            model.fit(
                Xtr_img,
                y_train,
                validation_data=(Xte_img, y_test),
                epochs=15,
                batch_size=16,
                callbacks=cb,
                verbose=fit_verbose,
            )

            base.trainable = True
            if cfg['unfreeze_last'] is not None:
                for layer in base.layers[:-int(cfg['unfreeze_last'])]:
                    layer.trainable = False

            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=cfg['lr_ft']),
                loss='binary_crossentropy',
                metrics=[keras.metrics.AUC(name='auc'), keras.metrics.BinaryAccuracy(name='acc')],
            )

            model.fit(
                Xtr_img,
                y_train,
                validation_data=(Xte_img, y_test),
                epochs=10,
                batch_size=16,
                callbacks=cb,
                verbose=fit_verbose,
            )

            y_prob = model.predict(Xte_img, verbose=0).ravel()
            try:
                from sklearn.metrics import roc_auc_score

                auc = float(roc_auc_score(y_test, y_prob))
            except Exception:
                auc = float('nan')

            if auc > best_auc:
                best_auc = auc
                best_cfg = cfg
                best_model = model

                if verbose:
                    print(f'[DoanAnhVu/EfficientNet] best_auc_val={best_auc} best_cfg={best_cfg}')

        if best_model is None:
            return None

        y_prob = best_model.predict(Xte_img, verbose=0).ravel()
        y_pred = (y_prob >= 0.5).astype(int)

        return {
            'y_pred': y_pred,
            'y_prob': y_prob,
            'model_obj': best_model,
            'meta': {'type': 'keras', 'input': 'imagenet_preprocess', 'gridsearch': {'grid': grid, 'best_cfg': best_cfg, 'best_auc_val': float(best_auc)}},
        }

    except Exception:
        return None
