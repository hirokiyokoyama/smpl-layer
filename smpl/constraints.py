import tensorflow as tf

class NonNegUnitSum(tf.keras.constraints.Constraint):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def __call__(self, w):
        w = w * tf.cast(w >= 0., w.dtype)
        w = w / tf.reduce_sum(w, axis=self.axis, keepdims=True)
        return w
