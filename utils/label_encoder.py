import numpy as np

age_label_encoder_utk = { a:(0 if  a < 5 else 
                         1 if  a < 10 else
                         2 if  a < 15 else
                         3 if  a < 20 else
                         4 if  a < 25 else
                         5 if  a < 30 else
                         6 if  a < 35 else
                         7 if  a < 40 else
                         8 if  a < 45 else
                         9 if  a < 50 else
                         10 if  a < 55 else
                         11 if  a < 60 else
                         12 if  a < 65 else
                         13 if  a < 70 else 
                         14 if  a < 75 else 
                         15 if  a < 80 else 
                         16 if a < 85 else
                         17 if a < 90 else
                         18 if a < 95 else
                         19 if a < 100 else
                         20) for a in np.arange(0,120)}


age_label_encoder = { a:(0 if  a < 20 else 
                         1 if  a < 30 else 
                         2 if  a < 40 else 
                         3 if  a < 50 else 
                         4 if  a < 60 else 
                         5 if a < 70 else  
                         6) for a in np.arange(0,101)}

age_label_encoder_2 = { a:(0 if  a < 25 else 
                         1 if  a < 35 else 
                         2 if  a < 45 else 
                         3 if  a < 55 else 
                         4 if  a < 65 else 
                         5 if a < 75 else  
                         6) for a in np.arange(0,101)}

    # mapping from ages 0-100 to age groups used in labels
def age_encode(y):
    return  np.asarray(list(map(age_label_encoder.get, y)))

    # mapping from ages 0-100 to age groups used in labels
def age_encode_2(y):
    return  np.asarray(list(map(age_label_encoder_2.get, y)))

    # mapping from ages 0-100 to age groups used in labels
def age_encode_utk(y):
    return  np.asarray(list(map(age_label_encoder_utk.get, y)))



def age_encode_regression(y):
    return y/np.max(y)


def inverse_mapping(mapping:dict) -> dict:
    return {v:k for k,v in mapping.items()}

def get_age_encoder():
    return age_label_encoder

def get_age_decoder():
    return inverse_mapping(get_age_encoder())

