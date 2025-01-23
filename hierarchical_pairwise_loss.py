import tensorflow as tf
def hierarchical_pairwise_loss(pctr_logits, pcvr_logits, click_labels, pay_labels, prices):
    pctr = tf.nn.sigmoid(pctr_logits)
    pcvr = tf.nn.sigmoid(pcvr_logits)
    pctcvr = pctr * pcvr
    ########################################
    # 1.pointwise loss
    ########################################
    ctcvr_loss_weight = 2
    ctcvr_pointwise_loss = tf.reduce_mean(tf.losses.log_loss(pay_labels, pctcvr))
    ctr_pointwise_loss = tf.reduce_mean(tf.losses.log_loss(click_labels, pctr))
    pointwise_loss = ctcvr_loss_weight * ctcvr_pointwise_loss + ctr_pointwise_loss


    ########################################
    # 2.conditional pairwise loss 
    ########################################
    pairwise_pctcvr = pctcvr - tf.transpose(pctcvr)
    price_mask = tf.where(tf.greater(prices - tf.transpose(prices), 0), tf.ones_like(pairwise_pctcvr), tf.zeros_like(pairwise_pctcvr))
    click_label_mask = tf.where(tf.greater_equal(click_labels - tf.transpose(click_labels), 0), tf.ones_like(pairwise_pctcvr),tf.zeros_like(pairwise_pctcvr))
    pay_label_mask = tf.where(tf.greater_equal(pay_labels - tf.transpose(pay_labels), 0), tf.ones_like(pairwise_pctcvr),tf.zeros_like(pairwise_pctcvr))
    mask = pay_label_mask * click_label_mask * price_mask
    
    conditional_loss = tf.reduce_mean(tf.losses.log_loss(
        predictions=pairwise_pctcvr,
        labels=tf.ones_like(pairwise_pctcvr),
        weights=mask
    ))
    conditional_loss = tf.where(tf.is_nan(conditional_loss), tf.zeros_like(conditional_loss), conditional_loss)

    hpl_loss = pointwise_loss + conditional_loss
    return hpl_loss
