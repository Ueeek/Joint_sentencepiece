import sys
sys.path.append("../src/")
from JointUnigramModel import JointUnigramModel


DIR="../../data/wmtEnDe/"
VOC_DST="../../align_tokenize_deen/vocs/"

def Train():
    arg = {
        "src_corpus":DIR+"train1K.en",
        "tgt_corpus":DIR+"train1K.de",
        "tgt_vocab": "./test.tgt.vocab",
        "src_vocab_size":1000,
        "dummy_vocab_size":10000,
        "seed_vocab_size":10000,
    }

    model = JointUnigramModel(arg)
    model.train()


if __name__ == "__main__":
    Train()
