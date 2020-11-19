class Vocab(object):
    #イニシャライザ
    def __init__(self):        
        #音階
        self.note_w2i = {} #単語⇒IDの辞書
        self.note_i2w = {} #ID⇒単語の辞書
        
        #音の長さ
        self.len_w2i = {} #単語⇒IDの辞書
        self.len_i2w = {} #ID⇒単語の辞書
        
        #コード
        self.chord_w2i = {} #単語⇒IDの辞書
        self.chord_i2w = {} #ID⇒単語の辞書
        
        #特殊文字（0:パッティング文字、1:開始文字等、2:終了文字、3:未知の文字）
        self.special_chars = ['<pad>', '<s>', '</s>', '<unk>']
        self.bos_char = self.special_chars[1]
        self.eos_char = self.special_chars[2]
        self.oov_char = self.special_chars[3]

    #ファイルの読み込みと辞書の作成
    def fit(self, path):
        self._note = set()
        self._len = set()
        self._chord = set()
        
        #文章を保存
        with open(path, 'r', encoding="utf-8") as f:
            sentences = f.read().splitlines() 
        txt_sentence = []
        #文章を単語ごとに保存（'彼', 'は',・・・・・・ ）
        for sentence in sentences:
            one_sentence = sentence.split()
            for i in one_sentence:
                txt_sentence.append(i.split('_'))
                
        for sentence in txt_sentence:
            self._note.update([sentence[0]])
            self._len.update([sentence[1]])
            self._chord.update([sentence[2]+'_'+sentence[3]]) 
        
        #単語⇒IDの辞書の作成(4～8777)
        self.note_w2i = {w: (i + len(self.special_chars))
                    for i, w in enumerate(self._note)}
        self.len_w2i = {w: (i + len(self.special_chars))
                    for i, w in enumerate(self._len)}
        self.chord_w2i = {w: (i + len(self.special_chars))
                    for i, w in enumerate(self._chord)}
        
        #単語⇒IDの辞書の作成(0～3)特殊文字、開始記号等
        for i, w in enumerate(self.special_chars):
            self.note_w2i[w] = i
            self.len_w2i[w] = i
            self.chord_w2i[w] = i
            
        #ID⇒単語の辞書の作成   
        self.note_i2w = {i: w for w, i in self.note_w2i.items()}
        self.len_i2w = {i: w for w, i in self.len_w2i.items()}   
        self.chord_i2w = {i: w for w, i in self.chord_w2i.items()}
        
    #ファイルの読み込み、全体をIDに変換
    def x_transform(self, path, bos=False, eos=False):
        output = []
        
        #文章を保存（'彼 は 走 る の が とても 早 い 。', ）
        with open(path, 'r', encoding="utf-8") as f:
            sentences = f.read().splitlines()

        for sentence in sentences:
            sentence = sentence.split()
            if bos:
                sentence = [str(self.bos_char)+'_'
                            +str(self.bos_char)+'_'
                            +str(self.bos_char)+'_'
                            +str(self.bos_char)] + sentence
            if eos:
                sentence = sentence + [str(self.eos_char)+'_'
                                       +str(self.eos_char)+'_'
                                       +str(self.eos_char)+'_'
                                       +str(self.eos_char)]
            output.append(self.encode(sentence))

        return output
    
    #ファイルの読み込み、全体をIDに変換
    def t_transform(self, path1, path2, path3, bos=False, eos=False):
        output = []
        
        #文章を保存（'彼 は 走 る の が とても 早 い 。', ）
        with open(path1, 'r', encoding="utf-8") as f:
            sentences = f.read().splitlines()
        
        with open(path2, 'r', encoding="utf-8") as f:
            chord_sentences = f.read().splitlines()
        
        with open(path3, 'r', encoding="utf-8") as f:
            num_sentences = f.read().splitlines()
            
        for i in range(len(sentences)):
            sentence = sentences[i]
            chord_sentence = chord_sentences[i]
            num_sentence = num_sentences[i]
            sentence = sentence.split()
            chord_sentence = chord_sentence.split()
            num_sentence = num_sentence.split()
            if bos:
                sentence = [str(self.bos_char)+'_'
                            +str(self.bos_char)+'_'
                            +str(self.bos_char)+'_'
                            +str(self.bos_char)] + sentence
                #chord_sentence = [str(self.bos_char)+'_'+str(self.bos_char)] + chord_sentence
                #num_sentence = [-1] + num_sentence
                
                
            if eos:
                sentence = sentence + [str(self.eos_char)+'_'
                                       +str(self.eos_char)+'_'
                                       +str(self.eos_char)+'_'
                                       +str(self.eos_char)]
                #chord_sentence = chord_sentence + [str(self.eos_char)+'_'+str(self.eos_char)]
                #num_sentence = num_sentence + [-2]
            output.append([self.encode(sentence),
                           self.chord_encode(chord_sentence),
                           [int(n) for n in num_sentence]])

        return output
    
    #単語をIDに変換
    def encode(self, sentence):
        output = []
        #1文章を入力,wは単語
        for w in sentence:
            oto, length, root, kousei = w.split('_')
            if root in self.special_chars:
                chord = root
            else:
                chord = root+'_'+kousei
            
            #辞書にない単語は未知語としてIDを振る,それ以外は単語を辞書を用いてIDに変換
            if oto not in self.note_w2i:
                oto_idx = self.note_w2i[self.oov_char]
            else:
                oto_idx = self.note_w2i[oto]
                
                
            if length not in self.len_w2i:
                length_idx = self.len_w2i[self.oov_char]
            else:
                length_idx = self.len_w2i[length]
            
            
            if chord not in self.chord_w2i:
                chord_idx = self.chord_w2i[self.oov_char]
            else:
                chord_idx = self.chord_w2i[chord]
                
            output.append([oto_idx, length_idx, chord_idx])
            
        return output
    
    #単語をIDに変換
    def chord_encode(self, sentence):
        output = []
        #1文章を入力,wは単語
        for w in sentence:
            root, kousei = w.split('_')
            if root in self.special_chars:
                chord = root
            else:
                chord = root+'_'+kousei
            if chord not in self.chord_w2i:
                chord_idx = self.chord_w2i[self.oov_char]
            else:
                chord_idx = self.chord_w2i[chord]
            output.append(chord_idx)
        return output
    
    
    #IDを単語に変換
    def decode(self, sentence):
        return [self.note_i2w[id[0]]+'_'+self.len_i2w[id[1]]+'_'+self.chord_i2w[id[2]] for id in sentence]
