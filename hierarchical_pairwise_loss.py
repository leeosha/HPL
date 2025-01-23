import tensorflow as tf
def hierarchical_pairwise_loss(pctr_logits, pcvr_logits, click_labels, pay_labels, prices):
    pctr = tf.nn.sigmoid(pctr_logits)
    pcvr = tf.nn.sigmoid(pcvr_logits)
    pctcvr = pctr * pcvr
    pctcvr_logit = tf.math.log(pctcvr/(1-pctcvr))
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
    pairwise_pctcvr = tf.nn.sigmoid(pctcvr_logit - tf.transpose(pctcvr_logit))
    price_gt_mask = tf.where(tf.greater(prices - tf.transpose(prices), 0), tf.ones_like(pairwise_pctcvr), tf.zeros_like(pairwise_pctcvr))
    click_label_ge_mask = tf.where(tf.greater_equal(click_labels - tf.transpose(click_labels), 0), tf.ones_like(pairwise_pctcvr),tf.zeros_like(pairwise_pctcvr))
    pay_label_ge_mask = tf.where(tf.greater_equal(pay_labels - tf.transpose(pay_labels), 0), tf.ones_like(pairwise_pctcvr),tf.zeros_like(pairwise_pctcvr))

    click_label_gt_mask = tf.where(tf.greater(click_labels - tf.transpose(click_labels), 0), tf.ones_like(pairwise_pctcvr),tf.zeros_like(pairwise_pctcvr))
    click_condition_on_pay_mask = pay_label_ge_mask * click_label_gt_mask
    click_condition_on_pay_loss = tf.reduce_mean(tf.losses.log_loss(
        predictions=pairwise_pctcvr,
        labels=tf.ones_like(pairwise_pctcvr),
        weights=click_condition_on_pay_mask
    ))
    click_condition_on_pay_loss = tf.where(tf.is_nan(click_condition_on_pay_loss), tf.zeros_like(click_condition_on_pay_loss), click_condition_on_pay_loss)

    
    price_condition_on_click_pay_mask = pay_label_ge_mask * click_label_ge_mask * price_gt_mask
    price_condition_on_click_pay_loss = tf.reduce_mean(tf.losses.log_loss(
        predictions=pairwise_pctcvr,
        labels=tf.ones_like(pairwise_pctcvr),
        weights=price_condition_on_click_pay_mask
    ))
    price_condition_on_click_pay_loss = tf.where(tf.is_nan(price_condition_on_click_pay_loss), tf.zeros_like(price_condition_on_click_pay_loss), price_condition_on_click_pay_loss)


    hpl_loss = pointwise_loss + click_condition_on_pay_loss + price_condition_on_click_pay_loss
    return hpl_loss
