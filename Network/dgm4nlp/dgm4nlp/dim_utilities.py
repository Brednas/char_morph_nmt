import keras.backend as K

# Flattens a t3 tensor to t2, keeping the final (character) dimension intact
def collapse_BandM(t):  # (B, M, C)
    # B, M, C = K.shape(t)[0], K.shape(t)[1], K.shape(t)[2]
    flat_c = K.reshape(t, (-1, K.shape(t)[-1]))  # (B * M, C)
    return flat_c

# Removes middle dimension left from max pooling
# from (B*M, 1, dx) to (B*M, dx)
def remove_middle_dim(t1_3):
    return K.reshape(t1_3, (K.shape(t1_3)[0], K.shape(t1_3)[2]))

def remove_middle_dim4(t4):
    return K.reshape(t4, (K.shape(t4)[0],K.shape(t4)[1],K.shape(t4)[3]))

# Converts a t3 back to t4
# that is we reintroduce the word-step dimension
def back_to_t4(t3, x):  # (B * M, C, dc)
    return K.reshape(t3, (K.shape(x)[0], K.shape(x)[1], K.shape(x)[2], K.shape(t3)[2]))  # (B, M, C, dc)
# Usage example:
# c = layers.Lambda(lambda t: back_to_t4(t), output_shape=(longest_xsnt, longest_xword, dc), name='c')(emb_c)

# Converts t2 back to t3, reintroducing word dimension
# From (B*M, dx) to (B, M, dx)
def back_to_t3(t2, x):
    return K.reshape(t2, (K.shape(x)[0], K.shape(x)[1], K.shape(t2)[1]))

def collapse_M_Mph(t4):
    return K.reshape(t4, (K.shape(t4)[0], K.shape(t4)[1]*K.shape(t4)[2], K.shape(t4)[3]))
    # return K.reshape(t4, (K.shape(t4)[0], -1, K.shape(t4)[3]))

def remove_last(t3):
    return t3[:,:,:-1]

def M(t4):
    return K.shape(t4[0,:,0,0])