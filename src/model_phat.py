from typing import Dict, Any

import numpy as np

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


def train_classical_models(X_bp_train, y_train, X_bp_test, y_test, random_state=42):
    """Train a set of classical ML models on bandpower features.

    - Uses RobustScaler for models that are sensitive to feature scale (LR/KNN/MLP).
    - Returns a dict of model results (pred/ prob/ object) and the fitted scaler.
    """
    results: Dict[str, Any] = {}

    scaler_bp = RobustScaler()
    Xtr = scaler_bp.fit_transform(X_bp_train)
    Xte = scaler_bp.transform(X_bp_test)

    # Logistic Regression
    lr = LogisticRegression(random_state=random_state, max_iter=5000, solver='liblinear')
    lr.fit(Xtr, y_train)
    y_pred = lr.predict(Xte)
    y_prob = lr.predict_proba(Xte)[:, 1]
    results['Logistic Regression (Bandpower)'] = {
        'y_pred': y_pred,
        'y_prob': y_prob,
        'model_obj': lr,
        'meta': {'type': 'sklearn'},
    }

    # Random Forest
    rf = RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)
    rf.fit(X_bp_train, y_train)
    y_pred = rf.predict(X_bp_test)
    y_prob = rf.predict_proba(X_bp_test)[:, 1]
    results['Random Forest (Bandpower)'] = {
        'y_pred': y_pred,
        'y_prob': y_prob,
        'model_obj': rf,
        'meta': {'type': 'sklearn'},
    }

    # KNN
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(Xtr, y_train)
    y_pred = knn.predict(Xte)
    y_prob = knn.predict_proba(Xte)[:, 1]
    results['KNN (Bandpower)'] = {
        'y_pred': y_pred,
        'y_prob': y_prob,
        'model_obj': knn,
        'meta': {'type': 'sklearn'},
    }

    # ANN/MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        solver='adam',
        max_iter=350,
        early_stopping=True,
        random_state=random_state,
    )
    mlp.fit(Xtr, y_train)
    y_pred = mlp.predict(Xte)
    try:
        y_prob = mlp.predict_proba(Xte)[:, 1]
    except Exception:
        y_prob = None
    results['ANN/MLP (Bandpower)'] = {
        'y_pred': y_pred,
        'y_prob': y_prob,
        'model_obj': mlp,
        'meta': {'type': 'sklearn'},
    }

    # XGBoost (optional)
    try:
        from xgboost import XGBClassifier

        xgb = XGBClassifier(
            random_state=random_state,
            eval_metric='logloss',
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
        )
        xgb.fit(X_bp_train, y_train)
        y_pred = xgb.predict(X_bp_test)
        y_prob = xgb.predict_proba(X_bp_test)[:, 1]
        results['XGBoost (Bandpower)'] = {
            'y_pred': y_pred,
            'y_prob': y_prob,
            'model_obj': xgb,
            'meta': {'type': 'xgboost'},
        }
    except Exception:
        pass

    return results, scaler_bp


def train_small_cnn_spectrogram(X_spec_train, y_train, X_spec_test, y_test, random_state=42):
    """Train a small CNN on spectrogram images.

    Spectrograms are normalized by train mean/std (simple z-score) before training.
    Returns a result dict compatible with export_models, or None if TF is unavailable.
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        tf.random.set_seed(random_state)

        Xtr = X_spec_train.astype(np.float32)
        Xte = X_spec_test.astype(np.float32)

        mu = float(Xtr.mean())
        sd = float(Xtr.std())
        Xtr_n = (Xtr - mu) / (sd + 1e-6)
        Xte_n = (Xte - mu) / (sd + 1e-6)

        model = keras.Sequential([
            layers.Input(shape=Xtr_n.shape[1:]),
            layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
            layers.MaxPool2D((2, 2)),
            layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            layers.MaxPool2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid'),
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='binary_crossentropy',
            metrics=[keras.metrics.AUC(name='auc'), keras.metrics.BinaryAccuracy(name='acc')],
        )

        cb = [
            keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=6, restore_best_weights=True)
        ]

        model.fit(
            Xtr_n, y_train,
            validation_data=(Xte_n, y_test),
            epochs=60,
            batch_size=16,
            callbacks=cb,
            verbose=1,
        )

        y_prob = model.predict(Xte_n).ravel()
        y_pred = (y_prob >= 0.5).astype(int)

        return {
            'y_pred': y_pred,
            'y_prob': y_prob,
            'model_obj': model,
            'meta': {'type': 'keras', 'norm': 'zscore_train'},
        }

    except Exception:
        return None


def train_efficientnet_tl(X_spec_train, y_train, X_spec_test, y_test, random_state=42):
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
        x = layers.Dropout(0.4)(x)
        out = layers.Dense(1, activation='sigmoid')(x)
        model = keras.Model(inp, out)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='binary_crossentropy',
            metrics=[keras.metrics.AUC(name='auc'), keras.metrics.BinaryAccuracy(name='acc')],
        )

        cb = [
            keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=6, restore_best_weights=True)
        ]

        model.fit(
            Xtr_img, y_train,
            validation_data=(Xte_img, y_test),
            epochs=25,
            batch_size=16,
            callbacks=cb,
            verbose=1,
        )

        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss='binary_crossentropy',
            metrics=[keras.metrics.AUC(name='auc'), keras.metrics.BinaryAccuracy(name='acc')],
        )

        model.fit(
            Xtr_img, y_train,
            validation_data=(Xte_img, y_test),
            epochs=15,
            batch_size=16,
            callbacks=cb,
            verbose=1,
        )

        y_prob = model.predict(Xte_img).ravel()
        y_pred = (y_prob >= 0.5).astype(int)

        return {
            'y_pred': y_pred,
            'y_prob': y_prob,
            'model_obj': model,
            'meta': {'type': 'keras', 'input': 'imagenet_preprocess'},
        }

    except Exception:
        return None
