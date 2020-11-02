from util import arg_parser
from collections import defaultdict

class JointUnigramModel:
    """"
    P(X,Y)でやる
    """

    def __init__ (self,argv):
        """
        get parameter from argv

        まず、tgtを固定する方を実装する。
        """

        if "help" in argv.keys():
            #self.print_arg_help()
            pass

        self.src_corpus = arg_parser(argv,"src_corpus",required=True)
        self.tgt_corpus = arg_parser(argv,"tgt_corpus",required=True)

        self.src_vocab_size = arg_parser(argv,"src_vocab_size",default_val=8000,required=True)
        self.unk_surface=arg_parser(argv,"unk_surface",default_val="⁇")
        self.kSentenceBoundary = arg_parser(argv,"kSentenceBoundary",default_val=chr(0x0000))

        self.character_coverage = arg_parser(argv,"coverage",default_val=1.0)

        self.Trie = None
        self.desired_src_voc_size = int(self.src_vocab_size*1.1)
        self.required_chars = dict()
        self.sep_voc = arg_parser(argv,"sep_voc",default_val=chr(9601))

        self.src_sentences = None
        self.tgt_sentences = None

    def load_sentence_src(self):
        """
        corpusを読み込む
        """
        self.src_sentences = []
        #TODO wordsいらない説(文単位でやるから)
        self.src_words = defaultdict(int) #space区切りの文字たち。(文全体をtokenizeするのは辛いから、wordsを使っていた。)
        chars = defaultdict(int)  #charの出現回数

        with open(self.src_corpus, encoding="utf-8") as f:
            for s in f:
                #spaceを"_"で置換して、文を保持
                s = s.replace("\n","")
                _s = self.sep_voc + self.sep_voc.join(s.split(" "))
                
                tmp=[]
                #charの出現回数
                for w in s.split(" "):
                    self.src_words[self.sep_voc+w] += 1
                    tmp.append(self.sep_voc+w)
                    for c in w:
                        if c=="\t":
                            continue
                        chars[c]+=1
                self.src_sentences.append(tmp)

        #required_charを決める。converageによって、freqが高いやつから残す
        accumulated_chars_count = 0
        all_chars_count = sum(chars.values())

        for key,val in sorted(chars.items(),key=lambda x:-x[1]):
            coverage = accumulated_chars_count/all_chars_count
            if coverage >= self.character_coverage:
                break

            assert key!=chr(0x0020),"space must not be included"
            assert key!="\t","tab must not be included"
            accumulated_chars_count+=val
            self.required_chars[key]=val

        print("Alphavet size=>",len(self.required_chars))
        print("Final character_coverage=>", accumulated_chars_count/all_chars_count)

    def load_sentence_tgt(self):
        """
        corpus を読み込む
        固定する方
        """

        self.tgt_sentences=[]
        with open(self.tgt_corpus) as f:
            for s in f:
                tmp=[]
                for w in s.split(" "):
                    tmp.append(self.sep_voc+w)
                self.tgt_sentences.append(tmp)

    def make_seed(self,key="src"):
        pass
    def run_EM(self):
        pass
    def prune_step(self):
        pass
    def chech_finish(self):
        pass

    def finialize_sentencepiece(self):
        pass

    def train(self):
        """
        training
        """

        self.load_sentence_src()
        self.load_sentence_tgt()
        src_seed_sentencepieces = self.make_seed("src")

        while True:
            self.run_EM()
            if self.chech_finish():
                break
            new_src_sentencepieces = self.prune_step()
            break #TODO TEST
        self.finialize_sentencepiece()
