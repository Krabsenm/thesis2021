import tensorflow as tf

class EqualityOfOdds(tf.keras.metrics.Metric):
    def __init__(self, name, **kwargs):
        super(EqualityOfOdds, self).__init__(name=name, **kwargs)
        self.equal_odds = self.add_weight(name='EO', initializer='zeros')
    def update_state(self, y_true, y_pred, sample_weight=None):
        print(y_true)
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_weights(sample_weight, values)
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.equal_odds