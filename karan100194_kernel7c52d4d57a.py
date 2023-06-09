#!/usr/bin/env python
# coding: utf-8



# -*- coding: utf-8 -*-
"""PytorchSampleCode.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1snQZ-QFjMsV-VkXdWroIRIXN9ZrRcr61
"""

#There are 2 clear advantages of PyTorch:

#Speed. The PyTorch version runs about 20 minutes faster.
#Determinism. The PyTorch version is fully deterministic. Especially when it gets harder to improve your score later in the competition, determinism is very important.

debug = False;
#Imports
import numpy as np
import pandas as pd
import os
import time
import gc
import random
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
from keras.preprocessing import text, sequence
import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F

from nltk import TweetTokenizer
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer

from scipy.stats import rankdata
from nltk.tokenize.treebank import TreebankWordTokenizer
Ttokenizer = TreebankWordTokenizer()

import time
start_time = time.time()
import sys
from gensim.models import KeyedVectors
package_dir = "../input/pytorchpretrainedberthaqishen/pytorch-pretrained-bert/pytorch-pretrained-BERT/"
sys.path = [package_dir] + sys.path


import os
import pickle
import argparse
import multiprocessing
import os




import re
import json

import emoji
import unicodedata
import multiprocessing
from functools import partial, lru_cache




import torch
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig, BertModel, BertPreTrainedModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling_gpt2 import GPT2Config, GPT2Model, GPT2PreTrainedModel
from pytorch_pretrained_bert.tokenization_gpt2 import GPT2Tokenizer
from pytorch_pretrained_bert.modeling_xlnet import XLNetConfig, XLNetModel, XLNetPreTrainedModel
from pytorch_pretrained_bert.tokenization_xlnet import XLNetTokenizer

from scipy.stats import rankdata
from nltk.tokenize.treebank import TreebankWordTokenizer
Ttokenizer = TreebankWordTokenizer()

import warnings
import traceback
warnings.filterwarnings(action='once')
device = torch.device('cuda')




# Adding progress bars for monitoring
def is_interactive():
   return 'SHLVL' not in os.environ
  
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

if not is_interactive():
    def nop(it, *a, **k):
        return it

    tqdm = nop


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()

CRAWL_EMBEDDING_PATH = "../input/gensim-embeddings-dataset/crawl-300d-2M.gensim"
GLOVE_EMBEDDING_PATH = "../input/gensim-embeddings-dataset/glove.840B.300d.gensim"
GOOGLE_EMBEDDING_PATH = "../input/gensim-embeddings-dataset/GoogleNews-vectors-negative300.gensim"
PARAGRAM_EMBEDDING_PATH = "../input/gensim-embeddings-dataset/paragram_300_sl999.gensim"


NUM_MODELS = 1
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
MAX_LEN = 220

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in tqdm(f))


def build_matrix(word_index, path):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    unknown_words = []
    
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            unknown_words.append(word)
    return embedding_matrix, unknown_words

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_model(model, train, test, loss_fn, output_dim, lr=0.001,
                batch_size=512, n_epochs=4,
                enable_checkpoint_ensemble=True):
    param_lrs = [{'params': param, 'lr': lr} for param in model.parameters()]
    optimizer = torch.optim.Adam(param_lrs, lr=lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.6 ** epoch)
    
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    all_test_preds = []
    checkpoint_weights = [2 ** epoch for epoch in range(n_epochs)]
    
    for epoch in range(n_epochs):
        start_time = time.time()
        
        scheduler.step()
        
        model.train()
        avg_loss = 0.
        
        for data in tqdm(train_loader, disable=False):
            x_batch = data[:-1]
            y_batch = data[-1]

            y_pred = model(*x_batch)            
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
            
        model.eval()
        test_preds = np.zeros((len(test), output_dim))
    
        for i, x_batch in enumerate(test_loader):
            y_pred = sigmoid(model(*x_batch).detach().cpu().numpy())

            test_preds[i * batch_size:(i+1) * batch_size, :] = y_pred

        all_test_preds.append(test_preds)
        elapsed_time = time.time() - start_time
        print('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(
              epoch + 1, n_epochs, avg_loss, elapsed_time))

    if enable_checkpoint_ensemble:
        test_preds = np.average(all_test_preds, weights=checkpoint_weights, axis=0)    
    else:
        test_preds = all_test_preds[-1]
        
    return test_preds

class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x
    
class NeuralNet(nn.Module):
    def __init__(self, embedding_matrix, num_aux_targets):
        super(NeuralNet, self).__init__()
        embed_size = embedding_matrix.shape[1]
        
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)
        
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)
    
        self.linear1 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
        self.linear2 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
        
        self.linear_out = nn.Linear(DENSE_HIDDEN_UNITS, 1)
        self.linear_aux_out = nn.Linear(DENSE_HIDDEN_UNITS, num_aux_targets)
        
    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)
        
        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)
        
        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1)
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1)
        
        h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear1  = F.relu(self.linear1(h_conc))
        h_conc_linear2  = F.relu(self.linear2(h_conc))
        
        hidden = h_conc + h_conc_linear1 + h_conc_linear2
        
        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        out = torch.cat([result, aux_result], 1)
        
        return out






symbols_to_isolate = '.,?!-;*"…:—()%#$&_/@＼・ω+=”“[]^–>\\°<~•≠™ˈʊɒ∞§{}·τα❤☺ɡ|¢→̶`❥━┣┫┗Ｏ►★©―ɪ✔®\x96\x92●£♥➤´¹☕≈÷♡◐║▬′ɔː€۩۞†μ✒➥═☆ˌ◄½ʻπδηλσερνʃ✬ＳＵＰＥＲＩＴ☻±♍µº¾✓◾؟．⬅℅»Вав❣⋅¿¬♫ＣＭβ█▓▒░⇒⭐›¡₂₃❧▰▔◞▀▂▃▄▅▆▇↙γ̄″☹➡«φ⅓„✋：¥̲̅́∙‛◇✏▷❓❗¶˚˙）сиʿ✨。ɑ\x80◕！％¯−ﬂﬁ₁²ʌ¼⁴⁄₄⌠♭✘╪▶☭✭♪☔☠♂☃☎✈✌✰❆☙○‣⚓年∎ℒ▪▙☏⅛ｃａｓǀ℮¸ｗ‚∼‖ℳ❄←☼⋆ʒ⊂、⅔¨͡๏⚾⚽Φ×θ￦？（℃⏩☮⚠月✊❌⭕▸■⇌☐☑⚡☄ǫ╭∩╮，例＞ʕɐ̣Δ₀✞┈╱╲▏▕┃╰▊▋╯┳┊≥☒↑☝ɹ✅☛♩☞ＡＪＢ◔◡↓♀⬆̱ℏ\x91⠀ˤ╚↺⇤∏✾◦♬³の｜／∵∴√Ω¤☜▲↳▫‿⬇✧ｏｖｍ－２０８＇‰≤∕ˆ⚜☁'
symbols_to_delete = '\n🍕\r🐵😑\xa0\ue014\t\uf818\uf04a\xad😢🐶️\uf0e0😜😎👊\u200b\u200e😁عدويهصقأناخلىبمغر😍💖💵Е👎😀😂\u202a\u202c🔥😄🏻💥ᴍʏʀᴇɴᴅᴏᴀᴋʜᴜʟᴛᴄᴘʙғᴊᴡɢ😋👏שלוםבי😱‼\x81エンジ故障\u2009🚌ᴵ͞🌟😊😳😧🙀😐😕\u200f👍😮😃😘אעכח💩💯⛽🚄🏼ஜ😖ᴠ🚲‐😟😈💪🙏🎯🌹😇💔😡\x7f👌ἐὶήιὲκἀίῃἴξ🙄Ｈ😠\ufeff\u2028😉😤⛺🙂\u3000تحكسة👮💙فزط😏🍾🎉😞\u2008🏾😅😭👻😥😔😓🏽🎆🍻🍽🎶🌺🤔😪\x08‑🐰🐇🐱🙆😨🙃💕𝘊𝘦𝘳𝘢𝘵𝘰𝘤𝘺𝘴𝘪𝘧𝘮𝘣💗💚地獄谷улкнПоАН🐾🐕😆ה🔗🚽歌舞伎🙈😴🏿🤗🇺🇸мυтѕ⤵🏆🎃😩\u200a🌠🐟💫💰💎эпрд\x95🖐🙅⛲🍰🤐👆🙌\u2002💛🙁👀🙊🙉\u2004ˢᵒʳʸᴼᴷᴺʷᵗʰᵉᵘ\x13🚬🤓\ue602😵άοόςέὸתמדףנרךצט😒͝🆕👅👥👄🔄🔤👉👤👶👲🔛🎓\uf0b7\uf04c\x9f\x10成都😣⏺😌🤑🌏😯ех😲Ἰᾶὁ💞🚓🔔📚🏀👐\u202d💤🍇\ue613小土豆🏡❔⁉\u202f👠》कर्मा🇹🇼🌸蔡英文🌞🎲レクサス😛外国人关系Сб💋💀🎄💜🤢َِьыгя不是\x9c\x9d🗑\u2005💃📣👿༼つ༽😰ḷЗз▱ц￼🤣卖温哥华议会下降你失去所有的钱加拿大坏税骗子🐝ツ🎅\x85🍺آإشء🎵🌎͟ἔ油别克🤡🤥😬🤧й\u2003🚀🤴ʲшчИОРФДЯМюж😝🖑ὐύύ特殊作戦群щ💨圆明园קℐ🏈😺🌍⏏ệ🍔🐮🍁🍆🍑🌮🌯🤦\u200d𝓒𝓲𝓿𝓵안영하세요ЖљКћ🍀😫🤤ῦ我出生在了可以说普通话汉语好极🎼🕺🍸🥂🗽🎇🎊🆘🤠👩🖒🚪天一家⚲\u2006⚭⚆⬭⬯⏖新✀╌🇫🇷🇩🇪🇮🇬🇧😷🇨🇦ХШ🌐\x1f杀鸡给猴看ʁ𝗪𝗵𝗲𝗻𝘆𝗼𝘂𝗿𝗮𝗹𝗶𝘇𝗯𝘁𝗰𝘀𝘅𝗽𝘄𝗱📺ϖ\u2000үսᴦᎥһͺ\u2007հ\u2001ɩｙｅ൦ｌƽｈ𝐓𝐡𝐞𝐫𝐮𝐝𝐚𝐃𝐜𝐩𝐭𝐢𝐨𝐧Ƅᴨןᑯ໐ΤᏧ௦Іᴑ܁𝐬𝐰𝐲𝐛𝐦𝐯𝐑𝐙𝐣𝐇𝐂𝐘𝟎ԜТᗞ౦〔Ꭻ𝐳𝐔𝐱𝟔𝟓𝐅🐋ﬃ💘💓ё𝘥𝘯𝘶💐🌋🌄🌅𝙬𝙖𝙨𝙤𝙣𝙡𝙮𝙘𝙠𝙚𝙙𝙜𝙧𝙥𝙩𝙪𝙗𝙞𝙝𝙛👺🐷ℋ𝐀𝐥𝐪🚶𝙢Ἱ🤘ͦ💸ج패티Ｗ𝙇ᵻ👂👃ɜ🎫\uf0a7БУі🚢🚂ગુજરાતીῆ🏃𝓬𝓻𝓴𝓮𝓽𝓼☘﴾̯﴿₽\ue807𝑻𝒆𝒍𝒕𝒉𝒓𝒖𝒂𝒏𝒅𝒔𝒎𝒗𝒊👽😙\u200cЛ‒🎾👹⎌🏒⛸公寓养宠物吗🏄🐀🚑🤷操美𝒑𝒚𝒐𝑴🤙🐒欢迎来到阿拉斯ספ𝙫🐈𝒌𝙊𝙭𝙆𝙋𝙍𝘼𝙅ﷻ🦄巨收赢得白鬼愤怒要买额ẽ🚗🐳𝟏𝐟𝟖𝟑𝟕𝒄𝟗𝐠𝙄𝙃👇锟斤拷𝗢𝟳𝟱𝟬⦁マルハニチロ株式社⛷한국어ㄸㅓ니͜ʖ𝘿𝙔₵𝒩ℯ𝒾𝓁𝒶𝓉𝓇𝓊𝓃𝓈𝓅ℴ𝒻𝒽𝓀𝓌𝒸𝓎𝙏ζ𝙟𝘃𝗺𝟮𝟭𝟯𝟲👋🦊多伦🐽🎻🎹⛓🏹🍷🦆为和中友谊祝贺与其想象对法如直接问用自己猜本传教士没积唯认识基督徒曾经让相信耶稣复活死怪他但当们聊些政治题时候战胜因圣把全堂结婚孩恐惧且栗谓这样还♾🎸🤕🤒⛑🎁批判检讨🏝🦁🙋😶쥐스탱트뤼도석유가격인상이경제황을렵게만들지않록잘관리해야합다캐나에서대마초와화약금의품런성분갈때는반드시허된사용🔫👁凸ὰ💲🗯𝙈Ἄ𝒇𝒈𝒘𝒃𝑬𝑶𝕾𝖙𝖗𝖆𝖎𝖌𝖍𝖕𝖊𝖔𝖑𝖉𝖓𝖐𝖜𝖞𝖚𝖇𝕿𝖘𝖄𝖛𝖒𝖋𝖂𝕴𝖟𝖈𝕸👑🚿💡知彼百\uf005𝙀𝒛𝑲𝑳𝑾𝒋𝟒😦𝙒𝘾𝘽🏐𝘩𝘨ὼṑ𝑱𝑹𝑫𝑵𝑪🇰🇵👾ᓇᒧᔭᐃᐧᐦᑳᐨᓃᓂᑲᐸᑭᑎᓀᐣ🐄🎈🔨🐎🤞🐸💟🎰🌝🛳点击查版🍭𝑥𝑦𝑧ＮＧ👣\uf020っ🏉ф💭🎥Ξ🐴👨🤳🦍\x0b🍩𝑯𝒒😗𝟐🏂👳🍗🕉🐲چی𝑮𝗕𝗴🍒ꜥⲣⲏ🐑⏰鉄リ事件ї💊「」\uf203\uf09a\uf222\ue608\uf202\uf099\uf469\ue607\uf410\ue600燻製シ虚偽屁理屈Г𝑩𝑰𝒀𝑺🌤𝗳𝗜𝗙𝗦𝗧🍊ὺἈἡχῖΛ⤏🇳𝒙ψՁմեռայինրւդձ冬至ὀ𝒁🔹🤚🍎𝑷🐂💅𝘬𝘱𝘸𝘷𝘐𝘭𝘓𝘖𝘹𝘲𝘫کΒώ💢ΜΟΝΑΕ🇱♲𝝈↴💒⊘Ȼ🚴🖕🖤🥘📍👈➕🚫🎨🌑🐻𝐎𝐍𝐊𝑭🤖🎎😼🕷ｇｒｎｔｉｄｕｆｂｋ𝟰🇴🇭🇻🇲𝗞𝗭𝗘𝗤👼📉🍟🍦🌈🔭《🐊🐍\uf10aლڡ🐦\U0001f92f\U0001f92a🐡💳ἱ🙇𝗸𝗟𝗠𝗷🥜さようなら🔼'
CONTRACTION_MAPPING = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }


isolate_dict = {ord(c):f' {c} ' for c in symbols_to_isolate}
remove_dict = {ord(c):f'' for c in symbols_to_delete}


def preprocessSymbols(x):
    for k, v in CONTRACTION_MAPPING.items():
        x = x.replace(' %s ' % k, ' %s ' % v)
    x = x.str.translate(remove_dict)
    x = x.str.translate(isolate_dict)
    return x


# Pandas multiprocessing
def _apply_df(args):
    df, func, kwargs = args
    return df.apply(func, **kwargs)


def apply_by_multiprocessing(df, func, **kwargs):
    workers = kwargs.pop('workers')
    pool = multiprocessing.Pool(processes=workers)
    result = pool.map(_apply_df, [(d, func, kwargs)
            for d in np.array_split(df, workers)])
    pool.close()
    return pd.concat(list(result))


def preprocess(data):
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    def clean_special_chars(text, punct):
        for p in punct:
            text = text.replace(p, ' ')
        return text

    data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))
    data= preprocessSymbols(data)
    return data




CUSTOM_TABLE = str.maketrans(
    {
        "\xad": None,
        "\x7f": None,
        "\ufeff": None,
        "\u200b": None,
        "\u200e": None,
        "\u202a": None,
        "\u202c": None,
        "‘": "'",
        "’": "'",
        "`": "'",
        "“": '"',
        "”": '"',
        "«": '"',
        "»": '"',
        "ɢ": "G",
        "ɪ": "I",
        "ɴ": "N",
        "ʀ": "R",
        "ʏ": "Y",
        "ʙ": "B",
        "ʜ": "H",
        "ʟ": "L",
        "ғ": "F",
        "ᴀ": "A",
        "ᴄ": "C",
        "ᴅ": "D",
        "ᴇ": "E",
        "ᴊ": "J",
        "ᴋ": "K",
        "ᴍ": "M",
        "Μ": "M",
        "ᴏ": "O",
        "ᴘ": "P",
        "ᴛ": "T",
        "ᴜ": "U",
        "ᴡ": "W",
        "ᴠ": "V",
        "ĸ": "K",
        "в": "B",
        "м": "M",
        "н": "H",
        "т": "T",
        "ѕ": "S",
        "—": "-",
        "–": "-",
    }
)

WORDS_REPLACER = [
    ("sh*t", "shit"),
    ("s**t", "shit"),
    ("f*ck", "fuck"),
    ("fu*k", "fuck"),
    ("f**k", "fuck"),
    ("f*****g", "fucking"),
    ("f***ing", "fucking"),
    ("f**king", "fucking"),
    ("p*ssy", "pussy"),
    ("p***y", "pussy"),
    ("pu**y", "pussy"),
    ("p*ss", "piss"),
    ("b*tch", "bitch"),
    ("bit*h", "bitch"),
    ("h*ll", "hell"),
    ("h**l", "hell"),
    ("cr*p", "crap"),
    ("d*mn", "damn"),
    ("stu*pid", "stupid"),
    ("st*pid", "stupid"),
    ("n*gger", "nigger"),
    ("n***ga", "nigger"),
    ("f*ggot", "faggot"),
    ("scr*w", "screw"),
    ("pr*ck", "prick"),
    ("g*d", "god"),
    ("s*x", "sex"),
    ("a*s", "ass"),
    ("a**hole", "asshole"),
    ("a***ole", "asshole"),
    ("a**", "ass"),
]

REGEX_REPLACER = [
    (re.compile(pat.replace("*", "\*"), flags=re.IGNORECASE), repl)
    for pat, repl in WORDS_REPLACER
]

RE_SPACE = re.compile(r"\s")
RE_MULTI_SPACE = re.compile(r"\s+")

NMS_TABLE = dict.fromkeys(
    i for i in range(sys.maxunicode + 1) if unicodedata.category(chr(i)) == "Mn"
)

EMOJI_REGEXP = emoji.get_emoji_regexp()

UNICODE_EMOJI_MY = {
    k: f" EMJ {v.strip(':').replace('_', ' ')} "
    for k, v in emoji.UNICODE_EMOJI_ALIAS.items()
}



HEBREW_TABLE = {i: "א" for i in range(0x0590, 0x05FF)}
ARABIC_TABLE = {i: "ا" for i in range(0x0600, 0x06FF)}
CHINESE_TABLE = {i: "是" for i in range(0x4E00, 0x9FFF)}
KANJI_TABLE = {i: "ッ" for i in range(0x2E80, 0x2FD5)}
HIRAGANA_TABLE = {i: "ッ" for i in range(0x3041, 0x3096)}
KATAKANA_TABLE = {i: "ッ" for i in range(0x30A0, 0x30FF)}

TABLE = dict()
TABLE.update(CUSTOM_TABLE)
TABLE.update(NMS_TABLE)
# Non-english languages
TABLE.update(CHINESE_TABLE)
TABLE.update(HEBREW_TABLE)
TABLE.update(ARABIC_TABLE)
TABLE.update(HIRAGANA_TABLE)
TABLE.update(KATAKANA_TABLE)
TABLE.update(KANJI_TABLE)

def my_demojize(string: str) -> str:
    def replace(match):
        return UNICODE_EMOJI_MY.get(match.group(0), match.group(0))

    return re.sub("\ufe0f", "", EMOJI_REGEXP.sub(replace, string))


def normalize(text: str) -> str:
    text = my_demojize(text)

    text = RE_SPACE.sub(" ", text)
    text = unicodedata.normalize("NFKD", text)
    text = text.translate(TABLE)
    text = RE_MULTI_SPACE.sub(" ", text).strip()

    for pattern, repl in REGEX_REPLACER:
        text = pattern.sub(repl, text)
#         print(text)

    return text




class XLNetForJigSaw(XLNetPreTrainedModel):
    def __init__(self, config, out_dim):
        
        super(XLNetForJigSaw, self).__init__(config)
        self.attn_type = config.attn_type
        self.same_length = config.same_length
        self.summary_type = "last"

        self.transformer = XLNetModel(config, output_attentions=False, keep_multihead_output=False)
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.activation = nn.Tanh()
        self.linear = nn.Linear(config.d_model, out_dim, bias=True)
        self.apply(self.init_xlnet_weights)

    def forward(self, input_ids, seg_id=None, input_mask=None,
                mems=None, perm_mask=None, target_mapping=None, inp_q=None,
                target=None, output_all_encoded_layers=True, head_mask=None, **kargs):

        output, hidden_states, new_mems = self.transformer(input_ids, seg_id, input_mask,
                                            mems, perm_mask, target_mapping, inp_q,
                                            output_all_encoded_layers, head_mask)
        first_token_tensor = output[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)

        return self.linear(pooled_output)
    
class GPT2ClassificationHeadModel(GPT2PreTrainedModel):

    def __init__(self, config, clf_dropout=0.4, out_dim=8):
        super(GPT2ClassificationHeadModel, self).__init__(config)
        self.transformer = GPT2Model(config)
        self.linear = nn.Linear(config.n_embd * 2, out_dim)

        nn.init.normal_(self.linear.weight, std = 0.02)
        nn.init.normal_(self.linear.bias, 0)
        self.apply(self.init_weights)

    def set_num_special_tokens(self, num_special_tokens):
        pass

    def forward(self, input_ids, position_ids=None, token_type_ids=None, lm_labels=None, past=None, **kwargs):
        hidden_states, presents = self.transformer(input_ids, position_ids, token_type_ids, past)
        if isinstance(hidden_states, list):
            hidden_states = hidden_states[-1]
        avg_pool = torch.mean(hidden_states, 1)
        max_pool, _ = torch.max(hidden_states, 1)
        h_conc = torch.cat((avg_pool, max_pool), 1)
        return self.linear(h_conc)
    
    
class BertForJigsaw(BertPreTrainedModel):

    def __init__(self, config, out_dim=7):
        super(BertForJigsaw, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, out_dim)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        return logits




def convert_line(row, max_seq_length, tokenizer, model_name='bert'):
    guid = row['id']
    text_a = row['comment_text']

    if 'label' in row.keys():
        label = row['label']
    else:
        label = None

    tokens_a = tokenizer.tokenize(text_a)

    if 'bert' in model_name:
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
    else:
        if len(tokens_a) > max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
        tokens = tokens_a
        input_ids = tokenizer.convert_tokens_to_ids(tokens_a)

    segment_ids = [0] * len(tokens)
    input_mask = [1] * len(input_ids)

    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    return input_ids, input_mask, segment_ids, label




def get_input_data(test_data):
    all_input_ids, all_input_mask,  all_segment_ids, all_label_ids = [], [], [], []
    for i, (input_ids, input_mask, segment_ids, label) in test_data.items():
        all_input_ids.append(input_ids)
        all_input_mask.append(input_mask)
        all_segment_ids.append(segment_ids)
        all_label_ids.append(label)
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(all_input_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)
    try:
        all_label_ids = torch.tensor(all_label_ids, dtype=torch.float32)
    except:
        pass
    
    return all_input_ids, all_input_mask,  all_segment_ids, all_label_ids

print('Def functions done! Time past %.2f secs' % (time.time() - start_time))




print('Loading data...')
df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
if debug:
    df = df.loc[:25]
test_ids = df['id'].tolist()
df['comment_text'] = df['comment_text'].astype(str)
print('Preprocessing...')
#df['comment_text'] = apply_by_multiprocessing(df['comment_text'], preprocess, workers=1)
df['comment_text']= preprocess(df['comment_text'])
print('Done! Time past %.2f secs' % (time.time() - start_time))




#train = pd.read_csv('drive/My Drive/train.csv')
# #test = pd.read_csv('drive/My Drive/test.csv')

train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

# x_train = preprocess(train['comment_text'])
# y_train = np.where(train['target'] >= 0.5, 1, 0)
# y_aux_train = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']]
# x_test = preprocess(test['comment_text'])

# max_features = None

# tokenizer = text.Tokenizer()

# tokenizer.fit_on_texts(list(x_train) + list(x_test))

# x_train = tokenizer.texts_to_sequences(x_train)
# x_test = tokenizer.texts_to_sequences(x_test)
# x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
# x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)
# print('Done! Time past %.2f secs' % (time.time() - start_time))
# max_features = max_features or len(tokenizer.word_index) + 1
# max_features




PORTER_STEMMER = PorterStemmer()
LANCASTER_STEMMER = LancasterStemmer()
SNOWBALL_STEMMER = SnowballStemmer("english")

def word_forms(word):
    yield word
    yield word.lower()
    yield word.upper()
    yield word.capitalize()
    yield PORTER_STEMMER.stem(word)
    yield LANCASTER_STEMMER.stem(word)
    yield SNOWBALL_STEMMER.stem(word)




def maybe_get_embedding(word, model):
    for form in word_forms(word):
        if form in model:
            return model[form]

    word = word.strip("-'")
    for form in word_forms(word):
        if form in model:
            return model[form]

    return None




# def gensim_to_embedding_matrix(word2index, path):
#     model = KeyedVectors.load(path, mmap="r")
#     embedding_matrix = np.zeros((max(word2index.values()) + 1, model.vector_size), dtype=np.float32)
#     unknown_words = []

#     for word, i in word2index.items():
#         maybe_embedding = maybe_get_embedding(word, model)
#         if maybe_embedding is not None:
#             embedding_matrix[i] = maybe_embedding
#         else:
#             unknown_words.append(word)

#     return embedding_matrix, unknown_words






# glove_matrix, unknown_words_glove = gensim_to_embedding_matrix(tokenizer.word_index, GLOVE_EMBEDDING_PATH)
# print('n unknown words (glove): ', len(unknown_words_glove))
# print('Done! Time past %.2f secs' % (time.time() - start_time))




# crawl_matrix, unknown_words_crawl = gensim_to_embedding_matrix(tokenizer.word_index, CRAWL_EMBEDDING_PATH)
# print('n unknown words (crawl): ', len(unknown_words_crawl))
# print('Done! Time past %.2f secs' % (time.time() - start_time))




# google_matrix, unknown_words_crawl = gensim_to_embedding_matrix(tokenizer.word_index, GOOGLE_EMBEDDING_PATH)
# print('n unknown words (crawl): ', len(unknown_words_crawl))


# paragram, unknown_words_crawl = gensim_to_embedding_matrix(tokenizer.word_index, PARAGRAM_EMBEDDING_PATH)
# print('n unknown words (crawl): ', len(unknown_words_crawl))
# print('Done! Time past %.2f secs' % (time.time() - start_time))





# embedding_matrix = np.concatenate([crawl_matrix, glove_matrix,google_matrix,paragram], axis=-1)
# print(embedding_matrix.shape)

# del crawl_matrix
# del glove_matrix
# del google_matrix
# del paragram
# gc.collect()
# print('Done! Time past %.2f secs' % (time.time() - start_time))





try:
    model_dir = '../input/jigsawmodels/bert_large_v2_99bin_250seq/bert_large_v2_99bin_250seq/'
    max_seq_length = 250
    short_length = 100
    tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=True)
    print('Converting data to sequences...')
    test_data = apply_by_multiprocessing(df, convert_line, axis=1, max_seq_length=max_seq_length, tokenizer=tokenizer, model_name='bert', workers=4)  # takes 2 mins
    all_input_ids, all_input_mask,  all_segment_ids, all_label_ids = get_input_data(test_data)
    long_idx = (all_input_ids[:, short_length-max_seq_length:].sum(1) > 0).nonzero().squeeze().numpy()
    short_idx = (all_input_ids[:, short_length-max_seq_length:].sum(1) == 0).nonzero().squeeze().numpy()
    print('Done! Time past %.2f secs' % (time.time() - start_time))

    # Load a trained model and vocabulary that you have fine-tuned
    print('Loading model from %s ...' % model_dir)
    bert_config = BertConfig(os.path.join(model_dir, 'config.json'))
    model = BertForJigsaw(bert_config, out_dim=99)
    model.load_state_dict(torch.load('%s/pytorch_model.bin' % model_dir))
    model.to(device)
    model.eval()
    print('Done! Time past %.2f secs' % (time.time() - start_time))

    print('Predicting model Bert Large')
    predictions_bert_large = np.zeros(df.shape[0])
    with torch.no_grad():
        for i, idx in enumerate([short_idx, long_idx]):
            test_data = TensorDataset(all_input_ids[idx], all_input_mask[idx], all_segment_ids[idx]) if i == 1 else                         TensorDataset(all_input_ids[idx, :short_length], all_input_mask[idx, :short_length], all_segment_ids[idx, :short_length])
            test_sampler = SequentialSampler(test_data)
            test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=64)

            pred = []
            for input_ids, input_mask, segment_ids in tqdm(test_dataloader):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)

                logits = model(input_ids, segment_ids, input_mask, labels=None)
                logits = torch.sigmoid(logits).detach().cpu().numpy()
                pred.append(logits)
            predictions_bert_large[idx] = np.vstack(pred).mean(1)

    print('Done! Time past %.2f secs' % (time.time() - start_time))
except:
    print('Something wrong with Bert Large.')
    traceback.print_exc()




predictions = np.zeros(df.shape[0])



try:
    predictions += predictions_bert_large * 1.0 
except: pass




# x_train_torch = torch.tensor(x_train, dtype=torch.long).cuda()
# x_test_torch = torch.tensor(x_test, dtype=torch.long).cuda()
# y_train_torch = torch.tensor(np.hstack([y_train[:, np.newaxis], y_aux_train]), dtype=torch.float32).cuda()

# # Training
# train_dataset = data.TensorDataset(x_train_torch, y_train_torch)
# test_dataset = data.TensorDataset(x_test_torch)

# all_test_preds2 = []




# for model_idx in range(NUM_MODELS):
#     print('Model ', model_idx)
#     seed_everything(1234 + model_idx)
#     model = NeuralNet(embedding_matrix, y_aux_train.shape[-1])
#     model.cuda()
#     test_preds = train_model(model, train_dataset, test_dataset, output_dim=y_train_torch.shape[-1], 
#                              loss_fn=nn.BCEWithLogitsLoss(reduction='mean'))
#     all_test_preds2.append(test_preds)




submission = pd.DataFrame.from_dict({
    'id': df['id'],
    'prediction': predictions_bert_large 
})

submission.to_csv('submission.csv', index=False)

