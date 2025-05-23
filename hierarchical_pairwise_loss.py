import tensorflow as tf
def get_loss(pctr_logits, pcvr_logits, click_labels, pay_labels, prices):
    pctr = tf.nn.sigmoid(pctr_logits)
    pcvr = tf.nn.sigmoid(pcvr_logits)
    pctcvr = pctr * pcvr
    
    ######################################## 
    # 1.pointwise loss
    ########################################
    
    ctcvr_pointwise_loss = tf.reduce_mean(tf.losses.log_loss(pay_labels, pctcvr))
    ctr_pointwise_loss = tf.reduce_mean(tf.losses.log_loss(click_labels, pctr))
    pointwise_loss =  ctcvr_pointwise_loss + ctr_pointwise_loss

    
    ########################################
    # 2.ctcvr pariwise loss
    ########################################
    pairwise_pctcvr = tf.nn.sigmoid(pctcvr - tf.transpose(pctcvr))
    pairwise_mask = tf.greater(pay_labels - tf.transpose(pay_labels), 0)
    ctcvr_pairwise_loss = tf.reduce_mean(tf.losses.log_loss(
        predictions=pairwise_pctcvr,
        labels=tf.ones_like(pairwise_pctcvr),
        weights=pairwise_mask
    ))
    ctcvr_pairwise_loss = tf.where(tf.is_nan(ctcvr_pairwise_loss), tf.zeros_like(ctcvr_pairwise_loss), ctcvr_pairwise_loss)
    

    ########################################
    # 3.hierarchical pairwise loss 
    ########################################
    pairwise_pctcvr = tf.nn.sigmoid(pctcvr - tf.transpose(pctcvr))
    price_gt_mask = tf.where(tf.greater(prices - tf.transpose(prices), 0), tf.ones_like(pairwise_pctcvr), tf.zeros_like(pairwise_pctcvr))
    click_label_ge_mask = tf.where(tf.greater_equal(click_labels - tf.transpose(click_labels), 0), tf.ones_like(pairwise_pctcvr),tf.zeros_like(pairwise_pctcvr))
    pay_label_ge_mask = tf.where(tf.greater_equal(pay_labels - tf.transpose(pay_labels), 0), tf.ones_like(pairwise_pctcvr),tf.zeros_like(pairwise_pctcvr))

    click_label_gt_mask = tf.where(tf.greater(click_labels - tf.transpose(click_labels), 0), tf.ones_like(pairwise_pctcvr),tf.zeros_like(pairwise_pctcvr))
    ctr_hpl_mask = pay_label_ge_mask * click_label_gt_mask
    ctr_hpl_loss = tf.reduce_mean(tf.losses.log_loss(
        predictions=pairwise_pctcvr,
        labels=tf.ones_like(pairwise_pctcvr),
        weights=ctr_hpl_mask
    ))
    ctr_hpl_loss = tf.where(tf.is_nan(ctr_hpl_loss), tf.zeros_like(ctr_hpl_loss), ctr_hpl_loss)

    
    price_hpl_mask = pay_label_ge_mask * click_label_ge_mask * price_gt_mask
    price_hpl_loss = tf.reduce_mean(tf.losses.log_loss(
        predictions=pairwise_pctcvr,
        labels=tf.ones_like(pairwise_pctcvr),
        weights=price_hpl_mask
    ))
    price_hpl_loss = tf.where(tf.is_nan(price_hpl_loss), tf.zeros_like(price_hpl_loss), price_hpl_loss)

    loss = pointwise_loss + ctcvr_pairwise_loss + ctr_hpl_loss + price_hpl_loss
    return loss
