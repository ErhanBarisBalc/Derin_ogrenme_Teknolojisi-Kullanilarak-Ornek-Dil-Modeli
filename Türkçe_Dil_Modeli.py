import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Veri kümesi oluşturma
class LanguageModelDataset(Dataset):
    def __init__(self, data, seq_len):
        self.seq_len = seq_len
        self.vocab = sorted(set(data))
        self.vocab_size = len(self.vocab)
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for i, w in enumerate(self.vocab)}
        self.data = torch.LongTensor([self.word2idx[w] for w in data])

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        inputs = self.data[idx:idx+self.seq_len]
        targets = self.data[idx+1:idx+self.seq_len+1]
        return inputs, targets

# Dil modeli sınıfı
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.5):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs, hidden=None):
        embeds = self.embedding(inputs)
        outputs, hidden = self.rnn(embeds, hidden)
        logits = self.fc(outputs)
        return logits, hidden

# Eğitim işlevi
def train(model, data_loader, optimizer, criterion, clip):
    model.train()
    total_loss = 0.
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, hidden = model(inputs)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(data_loader.dataset)

# Ayarlar
data_file = "data.txt"
seq_len = 10
batch_size = 128
embedding_dim = 128
hidden_dim = 256
num_layers = 2
dropout = 0.5
lr = 0.001
clip = 5.0
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Verileri yükleme ve topluca işleme
with open(data_file, "r") as f:
    data = f.read().replace("\n", " ")
dataset = LanguageModelDataset(data, seq_len)
vocab = dataset.vocab
vocab_size = dataset.vocab_size
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Dil modeli
model = LanguageModel(vocab_size, embedding_dim, hidden_dim, num_layers, dropout).to(device)

# Optimizer ve kayıp fonksiyonu
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# Eğitim
for epoch in range(epochs):
    train_loss = train(model, train_loader, optimizer, criterion, clip)
    print(f"Epoch: {epoch+1:}")
