#torchライブラリ
import torch
from torch import nn
import torch.optim as optimizers
#modelのライブラリ
from model.Attentionmodel import Encoder
from model.Attentionmodel import Decoder
#numpyライブラリ
import numpy as np
#プロット用ライブラリ
import matplotlib.pyplot as plt
#データをtrainとtestに分けるライブラリ
from sklearn.model_selection import train_test_split
#自作ライブラリ
from utils.Vocab import Vocab
from utils.DataLoader import DataLoader
#学習進歩監視ライブラリ
from tqdm import tqdm
#日にちのライブラリ
import datetime
now = datetime.datetime.now()

if __name__ == '__main__':
    
    '''0. パラメータ設定'''
    np.random.seed(123)
    torch.manual_seed(123)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 16
    EPOCH = 700
    
    '''1. データの準備'''
    #単語変換関数
    dict_vocab = Vocab()
    
    #学習データのパス
    en_train_path = './dataset/11_17Bb_allcall_m4s1.txt'
    de_train_path = './dataset/11_17Bb_allresp_m4s1.txt'
    dict_path = './dataset/all.txt'
    filenames = [en_train_path, de_train_path]
    with open(dict_path, 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)

    #データからID辞書を作成
    dict_vocab.fit(dict_path)
    
    #データをIDに変換
    x_data = dict_vocab.transform(en_train_path)
    t_data = dict_vocab.transform(de_train_path, eos=True)
    
    #テストデータと訓練データに分ける
    x_train, x_val, t_train, t_val = train_test_split(x_data, t_data, test_size=0.1, shuffle=True)
    
    #データをバッチ化する（tensor）
    dataloader={
        'train':DataLoader((x_train, t_train),
                           batch_size=BATCH_SIZE,
                           shuffle=True,
                           batch_first=False,
                           device=device),
        'test':DataLoader((x_val, t_val),
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          batch_first=False,
                          device=device)
        }
    

    '''2. モデルの構築'''
    #辞書の長さ
    depth_x = len(dict_vocab.i2w)
    depth_t = len(dict_vocab.i2w)
    input_dim = depth_x #入力層
    hidden_dim = 128 #中間層
    output_dim = depth_t #出力層
    maxlen = 64 #入力データの長さ
    #モデルの設定
    enc = Encoder(input_dim,
                  hidden_dim,
                  device=device,
                  num_layers=2).to(device)

    dec = Decoder(hidden_dim,
                  output_dim,
                  device=device,
                  num_layers=2).to(device)

    '''3. モデルの学習・評価'''
    #損失関数　ignore_index=0　マスク処理
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)
    #勾配法
    enc_optimizer = optimizers.Adam(enc.parameters(),
                                lr=0.001,
                                betas=(0.9, 0.999), amsgrad=True)
    dec_optimizer = optimizers.Adam(dec.parameters(),
                                lr=0.001,
                                betas=(0.9, 0.999), amsgrad=True)
    
    #Encoder-decoderモデル
    def Model_enc_dec(source, target=None, use_teacher_forcing=False):
        batch_size = source.size(1)
        if target is not None:
            len_target_sequences = target.size(0)
        else:
            len_target_sequences = maxlen

        hs, states = enc(source)

        y = torch.ones((1, batch_size),
                       dtype=torch.long,
                       device=device)
        output = torch.zeros((len_target_sequences,
                              batch_size,
                              output_dim),
                              device=device)

        for t in range(len_target_sequences):
            out, states = dec(y, hs, states, source=source)
            output[t] = out
            if use_teacher_forcing and target is not None:
                y = target[t].unsqueeze(0)
            else:
                y = out.max(-1)[1]
        return output
    #学習
    train_allloss=[]
    val_allloss=[]
    for epoch in range(EPOCH):
        for phase in ['train','test']:
            sum_loss = 0.
            corrects = 0.
            idx = 0
            with tqdm(total=len(dataloader[phase]),unit="batch") as pbar:
                pbar.set_description(f"Epoch[{epoch+1}/{EPOCH}]({phase})")
                for x_data, t_data in dataloader[phase]:
                    
                    if phase=='train':
                        import random
                        teacher_forcing_rate=0.5
                        use_teacher_forcing = (random.random() < teacher_forcing_rate)
                        enc.train(), dec.train()
                        preds = Model_enc_dec(x_data, t_data,
                                              use_teacher_forcing=use_teacher_forcing)
                    if phase=='test':
                        enc.eval(), dec.eval()
                        preds = Model_enc_dec(x_data, t_data,
                                              use_teacher_forcing=False)
                    
                    loss = criterion(preds.reshape(-1, preds.size(-1)),
                                     t_data.reshape(-1))
                    sum_loss += loss.item()
                    idx += 1
                    ave_loss = sum_loss / idx
                    
                    if phase=='train':
                        enc_optimizer.zero_grad(), dec_optimizer.zero_grad()
                        loss.backward()
                        enc_optimizer.step(), dec_optimizer.step()
                        
                    pbar.set_postfix({"loss":ave_loss})
                    pbar.update(1)
                    
            if phase=='train':
                train_allloss.append(sum_loss / len(dataloader[phase]))
            if phase=='test':
                val_allloss.append(sum_loss / len(dataloader[phase]))
            #モデルのセーブ
            if (epoch+1) % 50 == 0:
                torch.save(enc.state_dict(), './log/enc/'+str(now.month)+str(now.day)
                           +'encoder_epoch'+str(epoch+1))
                torch.save(dec.state_dict(), './log/dec/'+str(now.month)+str(now.day)
                           +'decoder_epoch'+str(epoch+1)) 

    #result
    plt.plot(range(1,EPOCH+1), train_allloss, label='Train')
    plt.plot(range(1,EPOCH+1), val_allloss, label='Test')
    plt.ylabel("CROSS_ENTROPY_ERROR")
    plt.xlabel("EPOCH")
    plt.title("Loss")
    plt.legend()
    plt.savefig('./log/'+str(now.month)+str(now.day)+'loss.png')
    plt.clf()
    import csv
    with open('./log/'+str(now.month)+str(now.day)+'loss.csv', "w") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['Train'])
        writer.writerow(train_allloss)
        writer.writerow(['Test'])
        writer.writerow(val_allloss)