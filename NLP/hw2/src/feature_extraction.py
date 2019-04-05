# config {stack,buffer,label}
def get_features(config,sent_dict):
    features = []

    # TODO Improve Features
    import numpy as np 
    if len(config[0]) > 0:
        # Top of stack.
        top = config[0][-1]
        # Token
        top_stk_token_feature = 'TOP_STK_TOKEN_'+str(sent_dict['FORM'][top].lower())
        features.append(top_stk_token_feature)
        #1. Length of stack
        len_stk_token_feature = 'LEN_STK_TOKEN_'+str(len(config[0]))
        features.append(len_stk_token_feature)
        # 2. Distance between top of stack and top of buffer
        if len(config[1]) > 0:
            top2 = config[1][-1]
        else:
            top2 = 0
    
        dist_token_feature = 'DIST_TOKEN_'+str(top2-top)
        features.append(dist_token_feature)
        # 3. POS of top of stack
        top_stk_pos_feature = 'TOP_STK_POS_'+str(sent_dict['POSTAG'][top].lower())
        features.append(top_stk_pos_feature)
        # 4. POS of top of buffer
        top_buf_pos_feature = 'TOP_BUF_POS_'+str(sent_dict['POSTAG'][top2].lower())
        features.append(top_buf_pos_feature)
        # 5. Token Buffer
        top_buf_token_feature = 'TOP_BUF_TOKEN_'+str(sent_dict['FORM'][top2].lower())
        features.append(top_buf_token_feature)
        # 6. Length of buffer 
        len_buf_token_feature = 'LEN_BUF_TOKEN_'+str(len(config[1]))
        features.append(len_buf_token_feature)
        # 7. Lemma of top of stack token
        top_stk_lemma_feature = 'TOP_STK_LEMMA_'+str(sent_dict['LEMMA'][top].lower())
        features.append(top_stk_lemma_feature)
        # 8. Lemma of top of buffer token
        top_buf_lemma_feature = 'TOP_BUF_LEMMA_'+str(sent_dict['LEMMA'][top2].lower())
        features.append(top_buf_lemma_feature)
        # 9. Number of 'VB' in the buffer
        buf_vb = [sent_dict['POSTAG'][i] for i in config[1]]
        num_vb_buf_feature = 'NUM_VB_BUF_'+str(np.sum([j == 'VB' for j in buf_vb]))
        features.append(num_vb_buf_feature)
        # 10. Index of 'VB' 
        if 'VB' in sent_dict['POSTAG']:
            vb_index = sent_dict['POSTAG'].index('VB')
        else:
            vb_index = 'NA'
        
        vb_index_feature = 'VB_IND_'+str(vb_index)
        features.append(vb_index_feature)
        
        # 11. Second of Stack
        if len(config[0]) > 1:
            snd_stk = config[0][-2]
            snd_stk_token_feature = 'SND_STK_TOKEN_'+str(sent_dict['FORM'][snd_stk].lower())
        else:
            snd_stk = 'NA'
            snd_stk_token_feature = 'SND_STK_TOKEN_'+str(snd_stk)
        features.append(snd_stk_token_feature)
        
        # 12. POS of Second of Stack
        if len(config[0]) > 1:
            snd_stk = config[0][-2]
            snd_stk_pos_feature = 'SND_STK_POS_'+str(sent_dict['POSTAG'][snd_stk].lower())
        else:
            snd_stk = 'NA'
            snd_stk_pos_feature = 'SND_STK_POS_'+str(snd_stk)
        
        features.append(snd_stk_pos_feature)

    return features
