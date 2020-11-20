#torchライブラリ
import torch
from torch import nn
import torch.optim as optimizers
#modelのライブラリ
from model.Net_enc_dec import EncoderDecoder
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
    EPOCH_SET = {'chord' : 50,
                 'note' : 50,
                 'len' : 300
                 }
    
    '''1. データの準備'''
    #単語変換関数
    dict_vocab = Vocab()
    
    #学習データのパス
    en_train_path = './dataset/11_18Bb_allcall_m4s1.txt'
    de_train_path = './dataset/11_18Bb_allresp_m4s1.txt'
    de_chord_path = './dataset/11_18Bb_chord_m4s1.txt'
    de_num_path = './dataset/11_18Bb_num_m4s1.txt'
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
    x_data = dict_vocab.x_transform(en_train_path)
    t_data = dict_vocab.t_transform(de_train_path, 
                                    de_chord_path,
                                    de_num_path,
                                    eos=True)
    #テストデータと訓練データに分ける
    x_train, x_val, t_train, t_val = train_test_split(x_data, t_data, test_size=0.1, shuffle=True)
    #データをバッチ化する（tensor）
    t = DataLoader((x_val, t_val),
                   batch_size=BATCH_SIZE,
                   shuffle=True,
                   batch_first=False,
                   device=device)
    
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
    note_depth_x = note_depth_t = len(dict_vocab.note_i2w)
    len_depth_x = len_depth_t = len(dict_vocab.len_i2w)
    chord_depth_x = chord_depth_t = len(dict_vocab.chord_i2w) 
    
    input_dim_note = note_depth_x #入力層
    input_dim_len = len_depth_x #入力層
    input_dim_chord = chord_depth_x #入力層
    
    hidden_dim = 128 #中間層
    
    output_dim_note = note_depth_t #出力層
    output_dim_len = len_depth_t #出力層
    output_dim_chord = chord_depth_t #出力層
    
    maxlen = 64 #入力データの長さ
    #モデルの設定
    model = EncoderDecoder(input_dim_note, input_dim_len, input_dim_chord,
                           hidden_dim,
                           output_dim_note, output_dim_len, output_dim_chord,
                           num_layers=2,
                           device=device).to(device)
    '''3. モデルの学習・評価'''
    #損失関数　ignore_index=0　マスク処理
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)
    #勾配法
    enc_optimizer = optimizers.Adam(model.encoder.parameters(),
                                      lr=0.001,
                                      betas=(0.9, 0.999), amsgrad=True)
    dec1_optimizer = optimizers.Adam(model.decoder1.parameters(),
                                      lr=0.001,
                                      betas=(0.9, 0.999), amsgrad=True)
    dec2_optimizer = optimizers.Adam(model.decoder2.parameters(),
                                      lr=0.001,
                                      betas=(0.9, 0.999), amsgrad=True)
    dec3_optimizer = optimizers.Adam(model.decoder3.parameters(),
                                      lr=0.001,
                                      betas=(0.9, 0.999), amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(dec2_optimizer, step_size=1, gamma=0.95)
    path_model = './log/model/1119note_model_epoch50'
    model.load_state_dict(torch.load(path_model, map_location=torch.device(device)))
    #学習
    train_allloss=[]
    val_allloss=[]
    for learning_phase in ['len']:#['chord', 'note', 'len']:
        EPOCH = EPOCH_SET[learning_phase]
        for epoch in range(EPOCH):
            #print('lr: {}'.format(scheduler.get_lr()[0]))
            for phase in ['train','test']:
                sum_loss = 0.
                corrects = 0.
                idx = 0
                with tqdm(total=len(dataloader[phase]),unit="batch") as pbar:
                    pbar.set_description(f"Epoch[{epoch+1}/{EPOCH}]({learning_phase},{phase})")
                    for (x_note, x_len, x_chord, 
                         chord, t_num, 
                         t_note, t_len, t_chord) in dataloader[phase]:

                        if phase=='train':
                            import random
                            teacher_forcing_rate=0.5
                            use_teacher_forcing = (random.random() < teacher_forcing_rate)
                            if learning_phase=='chord':
                                model.encoder.train()
                                model.decoder3.train()
                                
                            if learning_phase=='note':
                                model.decoder1.train()
                                model.encoder.train()
                                model.decoder3.train()
                                
                            if learning_phase=='len':
                                #model.decoder2.train()
                                model.train()
                                
                            preds = model(learning_phase, x_note, x_len, x_chord, 
                                          chord, t_num, t_note, t_len, t_chord,
                                          use_teacher_forcing=use_teacher_forcing)
                        if phase=='test':
                            model.eval()
                            preds = model(learning_phase, x_note, x_len, x_chord, 
                                          chord, t_num, t_note, t_len, t_chord,
                                          use_teacher_forcing=False)
                        
                        if learning_phase=='chord':
                            loss = criterion(preds.reshape(-1, preds.size(-1)),
                                         chord.reshape(-1))
                            sum_loss += loss.item()
                            idx += 1
                            ave_loss = sum_loss / idx
                            if phase=='train':
                                enc_optimizer.zero_grad(), dec3_optimizer.zero_grad()
                                loss.backward()
                                enc_optimizer.step(), dec3_optimizer.step()
                        
                        if learning_phase=='note':
                            loss = criterion(preds.reshape(-1, preds.size(-1)),
                                             t_note.reshape(-1))
                            sum_loss += loss.item()
                            idx += 1
                            ave_loss = sum_loss / idx
                            if phase=='train':
                                dec1_optimizer.zero_grad()
                                loss.backward()
                                dec1_optimizer.step()
                        
                        if learning_phase=='len':
                            loss = criterion(preds.reshape(-1, preds.size(-1)),
                                             t_len.reshape(-1))
                            sum_loss += loss.item()
                            idx += 1
                            ave_loss = sum_loss / idx
                            if phase=='train':
                                dec2_optimizer.zero_grad()
                                loss.backward()
                                dec2_optimizer.step()
                            
                        pbar.set_postfix({"loss":ave_loss})
                        pbar.update(1)
                        
                if phase=='train':
                    train_allloss.append(sum_loss / len(dataloader[phase]))
                if phase=='test':
                    val_allloss.append(sum_loss / len(dataloader[phase]))
            #モデルのセーブ
            if ((epoch+1) % 50 == 0 and epoch<350) or ((epoch+1)%100 ==0):
                torch.save(model.state_dict(), './log/model/'+str(now.month)+str(now.day)
                           +str(learning_phase)+'_model_epoch'+str(epoch+1))
            scheduler.step()

        #result
        plt.plot(range(1,EPOCH+1), train_allloss, label='Train')
        plt.plot(range(1,EPOCH+1), val_allloss, label='Test')
        plt.ylabel("CROSS_ENTROPY_ERROR")
        plt.xlabel("EPOCH")
        plt.title("Loss")
        plt.legend()
        plt.savefig('./log/'+str(now.month)+str(now.day)+str(learning_phase)+'loss.png')
        plt.clf()
        import csv
        with open('./log/'+str(now.month)+str(now.day)+str(learning_phase)+'loss.csv', "w") as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(['Train'])
            writer.writerow(train_allloss)
            writer.writerow(['Test'])
            writer.writerow(val_allloss)