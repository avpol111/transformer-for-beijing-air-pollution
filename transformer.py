import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('Beijing_PM.csv')
dataset = df[["pm2.5"]]
dataset.fillna(0, inplace=True)
dataset = dataset[24:]
timeseries = dataset.values.astype('float32')

scaler = MinMaxScaler(feature_range=(-1, 1))
timeseries = scaler.fit_transform(timeseries.reshape(-1, 1)).reshape(-1)

input_window = 50 # number of input steps
output_window = 1 # number of prediction steps
block_len = input_window + output_window
batch_size = 200
train_size = 0.8

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :].repeat(1, x.shape[1], 1)

class TransNet(nn.Module):
    def __init__(self, feature_size=100, num_layers=1, dropout=0.1):
        super(TransNet, self).__init__()
        self.model_type = 'Transformer'
        self.input_embedding  = nn.Linear(1, feature_size)
        self.src_mask = None
		
		self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            mask = self._generate_square_subsequent_mask(len(src))
            self.src_mask = mask
			
	    src = self.input_embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

def create_inout_sequences(input_data, input_window, output_window):
    inout_seq = []
    L = len(input_data)
    block_num =  L - block_len + 1

    for i in range(block_num):
        train_seq = input_data[i : i + input_window]
        train_label = input_data[i + output_window : i + input_window + output_window]
        inout_seq.append((train_seq ,train_label))
	return torch.FloatTensor(np.array(inout_seq))

def get_data():
    samples = int(len(timeseries) * train_size)
    train_data = timeseries[:samples]
    test_data = timeseries[samples:]

    train_sequence = create_inout_sequences(train_data, input_window, output_window)
    train_sequence = train_sequence[:-output_window]

    test_data = create_inout_sequences(test_data, input_window, output_window)
    test_data = test_data[:-output_window]
    return train_sequence, test_data

def get_batch(input_data, i , batch_size):
    batch_len = min(batch_size, len(input_data) - i)
    data = input_data[i:i + batch_len]
    input = torch.stack([item[0] for item in data]).view((input_window, batch_len, 1))
    target = torch.stack([item[1] for item in data]).view((input_window, batch_len, 1))
    return input, target

def train(train_data):
    model.train()

    for batch, i in enumerate(range(0, len(train_data), batch_size)):
        data, targets = get_batch(train_data, i , batch_size)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()
		
def evaluate(eval_model, data_source):
    eval_model.eval()
    total_loss = 0.
    eval_batch_size = 500
    with torch.no_grad():
        for i in range(0, len(data_source), eval_batch_size):
            data, targets = get_batch(data_source, i, eval_batch_size)
            output = eval_model(data)
            output = torch.squeeze(output)
            output = scaler.inverse_transform(output)
            output = torch.from_numpy(output)
            targets = torch.squeeze(targets)
            targets = scaler.inverse_transform(targets)
            targets = torch.from_numpy(targets) # a separate inverse transform function may be a neater solution
            total_loss += len(data[0]) * criterion(output, targets).item()
    print ("test RMSE ", np.sqrt(total_loss / len(data_source)))
	
train_data, val_data = get_data()
model = TransNet()

criterion = nn.MSELoss()
lr = 0.005
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

epochs = 20
for epoch in range(1, epochs + 1):
    print(f"Epoch {epoch}:")
    train(train_data)
    evaluate(model, val_data)
    scheduler.step()