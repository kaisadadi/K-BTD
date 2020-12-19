
import sys
sys.path.append("../../")
from model.model.LSTM import LSTM
from model.model.TextCNN import TextCNN
from model.model.bert import Bert
from model.model.CAML_NACCL import CAML
from model.MODEL_V1 import MODEL_V1
from model.MODEL_V2 import MODEL_V2
from model.MODEL_V3 import MODEL_V3


model_list = {
    "LSTM": LSTM,
    "TextCNN": TextCNN,
    "Bert": Bert,
    "CAML": CAML,
    "MODEL_V1": MODEL_V1,
    "MODEL_V2": MODEL_V2,
    "MODEL_V3": MODEL_V3,
}


def get_model(name, config):
    if name in model_list.keys():
        return model_list[name](config)
    else:
        raise NotImplementedError
