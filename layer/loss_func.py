import tensorflow as tf


# orientation_loss here is not consist with loss in the paper
def orientation_loss(y_true, y_pred):
    # Find number of anchors
    anchors = tf.reduce_sum(tf.square(y_true), axis=2)
    anchors = tf.greater(anchors, tf.constant(0.5))
    anchors = tf.reduce_sum(tf.cast(anchors, tf.float32), 1)

    # Define the loss
    loss = -(y_true[:, :, 0] * y_pred[:, :, 0] + y_true[:, :, 1] * y_pred[:, :, 1])
    loss = tf.reduce_sum(loss, axis=1)
    loss = loss / anchors

    return tf.reduce_mean(loss)