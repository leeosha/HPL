import tensorflow as tf
def hierarchical_pairwise_loss(pctcvr_logits, prices, click_labels, pay_labels):
    logits = tf.reshape(logits, [-1, 1])
    labels = tf.reshape(labels, [-1, 1])
    levels = tf.reshape(levels, [-1, 1])
    pairwise_logits = logits - tf.transpose(logits)
    pairwise_logits = tf.sigmoid(pairwise_logits)
    pairwise_mask1 = tf.where(tf.greater(labels - tf.transpose(labels), 0), tf.ones_like(pairwise_logits), 
        tf.zeros_like(pairwise_logits))
    pairwise_mask2 = tf.where(tf.greater_equal(levels - tf.transpose(levels), 0), tf.ones_like(
        pairwise_logits),tf.zeros_like(pairwise_logits))
    pairwise_mask = pairwise_mask1 * pairwise_mask2
    pairwise_psudo_labels = tf.ones_like(pairwise_logits)
    rank_loss = tf.reduce_mean(tf.losses.log_loss(
        predictions=pairwise_logits,
        labels=pairwise_psudo_labels,
        weights=pairwise_mask
    ))
    
    h_rank_loss = tf.where(tf.is_nan(rank_loss), tf.zeros_like(rank_loss), rank_loss)
    return h_rank_loss
