# kerasfm
An arbitrary order Factorization Machine based on TensorFlow2.0

It's a tensorflow2.0-style interpretation of [tffm](https://github.com/geffy/tffm).
Most of the core parts are rewritten with Keras API, which is highly encouraged in tensorflow2.0

# Dependencies
- scikit-learn
- numpy
- tqdm
- tensorflow2.0+

# Usage
Usage is a little different from the [original one](https://github.com/geffy/tffm#usage) because of the change in the optimizer of tf2.0. 
```python
from tffm import TFFMClassifier
model = TFFMClassifier(
    order=6,
    rank=10,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    n_epochs=100,
    batch_size=-1,
    init_std=0.001,
    input_type='dense'
)
model.fit(X_tr, y_tr, show_progress=True)
```

# Note
In tensorflow2.0, the `tf.summary` function cannot visit the variables computed by a graph function. So, if the `log_dir` parameter is assigned a path, the training loop will automatically switch to eager mode to enable the logging. It may cause a significant slowdown in training.
