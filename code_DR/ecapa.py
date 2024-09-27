import torch
import numpy as np
#from speechbrain.pretrained import EncoderClassifier
#from speechbrain
import speechbrain

def get_ecapa_classifier():
    if torch.cuda.is_available():
        classifier = speechbrain.pretrained.EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":"cuda"}) 
    else:
        classifier = speechbrain.pretrained.EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb") 
    return classifier

def get_ecapa_embeddings(signals, classifier=None, fs = 16000, evaluation_length = 4, start_point = 0):
    """
    signals: np array: nMic x signal lenght
    """
    if classifier == None:
     classifier = get_ecapa_classifier()

    signals = signals[:,int(start_point*fs):int(start_point*fs + evaluation_length*fs)]
    embeddings = classifier.encode_batch(signals)[:,0,:]
    if torch.cuda.is_available():
        embeddings = embeddings.detach().cpu().numpy()
    else:
        embeddings = embeddings.detach().numpy()

    return embeddings
