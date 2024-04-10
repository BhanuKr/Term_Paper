# import torch
# from torch import nn
# from torch.autograd import Variable
# import torch.utils.data as Data
# import numpy as np
# import torch.nn.functional as F
# import datetime
# from layers import GraphConvolution
# import pickle
# from scipy.sparse import csr_matrix
# import torch.nn.init as init
# import warnings
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# from sklearn.metrics import f1_score, fbeta_score


# def kmax_pooling(x, dim, k):
#     index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
#     return x.gather(dim, index)


# def get_sentense_marix(x):
#     one_matrix = np.zeros((140, 140), dtype=np.float32)
#     for index, item in enumerate(x):
#         one_matrix[index][index] = 1
#         if not item:
#             one_matrix[index, item-1] = 2
#             one_matrix[item-1, index] = 3
#     return torch.FloatTensor(one_matrix)




# # h.p. define
# torch.manual_seed(1)
# EPOCH = 200
# BATCH_SIZE = 32
# LR = 0.001
# HIDDEN_NUM = 64
# HIDDEN_LAYER = 2
# # process data
# print("Loading data...")
# max_document_length = 140

# fr = open('data_train_noRen_noW2v.txt', 'rb')
# x_train = pickle.load(fr)
# y_train = pickle.load(fr)
# length_train = pickle.load(fr)

# fr = open('data_test.txt', 'rb')
# x_dev = pickle.load(fr)
# y_dev = pickle.load(fr)
# length_dev = pickle.load(fr)
# # Randomly shuffle data
# np.random.seed(10)
# shuffle_indices = np.random.permutation(np.arange(len(y_train)))
# print(shuffle_indices.shape)
# print('x_train shape ', x_train.shape)

# x_train = x_train[shuffle_indices]
# y_train = y_train[shuffle_indices]

# length_shuffled_train = length_train[shuffle_indices]
# x_train = torch.from_numpy(x_train)
# y_train = torch.from_numpy(y_train).long()

# length_dev = []
# for item in x_dev:
#     length_dev.append(list(item).index(0))
# print(len(length_dev))

# x_dev = torch.from_numpy(x_dev)
# y_dev = torch.max(torch.from_numpy(y_dev).long(), dim=1)[1]

# train_x = torch.LongTensor(x_train).cuda()
# train_y = torch.LongTensor(y_train).cuda()

# #   y = torch.LongTensor(y)
# #test_x = torch.cat(test_x, dim=0)
# #test_y = torch.LongTensor(test_y)
# test_x = torch.LongTensor(x_dev).cuda()
# test_y = torch.LongTensor(y_dev).cuda()

# torch_dataset = Data.TensorDataset(train_x, train_y)
# torch_testset = Data.TensorDataset(test_x, test_y)
# loader = Data.DataLoader(
#     dataset=torch_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True
# )

# test_loader = Data.DataLoader(
#     dataset=torch_testset,
#     batch_size=128
# )
# print("data process finished")


# class LSTM_GCN(nn.Module):
#     def __init__(self):
#         super(LSTM_GCN, self).__init__()
#         self.embedding = nn.Embedding(76215, 300).cuda()
#         self.lstm = nn.LSTM(
#             input_size=300,  # dim of word vector
#             hidden_size=180,  # dim of output of lstm nn`
#             num_layers=2,  # num of hidden layers
#             batch_first=True,
#             dropout=0.5,
#             bidirectional=True
#         ).cuda()
#         self.batch1 = nn.BatchNorm1d(max_document_length).cuda()
#         self.gc = GraphConvolution(360, 7)
#         init.xavier_normal_(self.lstm.all_weights[0][0], gain=1)
#         init.xavier_normal_(self.lstm.all_weights[0][1], gain=1)
#         init.xavier_normal_(self.lstm.all_weights[1][0], gain=1)
#         init.xavier_normal_(self.lstm.all_weights[1][1], gain=1)

#     def forward(self, x_and_adj):
#         x = x_and_adj[:, :max_document_length].cuda()
#         adj = x_and_adj[:, -max_document_length:]
#         x = self.embedding(x)
#         lstm_out, _ = self.lstm(x, None)
#         out = self.batch1(lstm_out)
#         out = F.relu(out)
#         adj_Metrix = []
#         for item in adj:
#             adj_Metrix.append(torch.unsqueeze(get_sentense_marix(item), dim=0))
#         adj_Metrix = torch.cat(adj_Metrix, dim=0)
#         out_g1 = self.gc(out, adj_Metrix)
#         out = torch.median(out_g1, 1)[0]
#         return out


# model = LSTM_GCN()
# #model.cuda()
# model.train()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-8)
# loss_func = nn.CrossEntropyLoss()
# print(model)
# best = 0


# def get_test():
#     global best
#     model.eval()
#     print('start dev test')
#     record = []
#     for index, (batch_x, batch_y) in enumerate(test_loader):
#         test_output = model(batch_x)
#         test_output = list(torch.max(test_output, dim=1)[1].cpu().numpy())
#         record.extend(test_output)
#     label = list(test_y.cpu().numpy())
#     y_true = label
#     y_pred = record

#     print("accuracy:", accuracy_score(y_true, y_pred))    # Return the number of correctly classified samples
#     if accuracy_score(y_true, y_pred) > best:
#         torch.save(model, "best_model.pth")
#     print("macro_precision", precision_score(y_true, y_pred, average='macro'))
#     print("micro_precision", precision_score(y_true, y_pred, average='micro'))

#     # Calculate recall score
#     print("macro_recall", recall_score(y_true, y_pred, average='macro'))
#     print("micro_recall", recall_score(y_true, y_pred, average='micro'))

#     # Calculate f1 score
#     print("macro_f", f1_score(y_true, y_pred, average='macro'))
#     print("micro_f", f1_score(y_true, y_pred, average='micro'))

#     model.train()


# f = open('accuracy_record.txt', 'w+')
# f2 = open('loss_record.txt', 'w+')
# loss_sum = 0
# accuracy_sum = 0

# for epoch in range(EPOCH):
#     for index, (batch_x, batch_y) in enumerate(loader):
#         right = 0
#         if index == 0:
#             get_test()
#             loss_sum = 0
#             accuracy_sum = 0
#         #   one hot to scalar
#         #batch_y = batch_y.cuda()
#         output = model(batch_x)
#         optimizer.zero_grad()
#         #output = output.cuda()
#         batch_y = torch.argmax(batch_y, dim=1)
#         #print(batch_y)
#         #print(output.size())
#         #print(batch_y.size())
#         loss = loss_func(output, batch_y)
#         #   gcnloss = ((torch.matmul(model.gc.weight.t(), model.gc.weight) - i)**2).sum().cuda()
#         #   loss += gcnloss * 0.000005
#         lstmloss = 0
#         for item in model.lstm.parameters():
#             if len(item.shape) == 2:
#                 I = torch.eye(item.shape[1]).cuda()
#                 lstmloss += ((torch.matmul(item.t(), item)-I)**2).sum().cuda()
#         loss += lstmloss * 0.00000005
#         loss.backward()
#         predict = torch.argmax(output, dim=1).cpu().numpy().tolist()
#         label = batch_y.cpu().numpy().tolist()

#         for i in range(0, batch_y.size(0)):
#             if predict[i] == label[i]:
#                 right += 1
#         optimizer.step()
#         accuracy_sum += right/batch_y.size(0)
#         loss_sum += float(loss)
#         if index % 50 == 0:
#             print("batch", index, "/ "+str(len(loader))+": ",  "\tloss: ", float(loss), "\taccuracy: ", right/batch_y.size(0))
#     print('epoch: ', epoch, 'has been finish')


import torch
from torch import nn
import torch.utils.data as Data
import numpy as np
import torch.nn.functional as F
import datetime
from layers import GraphConvolution
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np
import torch.nn.functional as F
import datetime
from layers import GraphConvolution
import pickle
from scipy.sparse import csr_matrix
import torch.nn.init as init
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, fbeta_score


def get_sentense_matrix(x):
    # Function to create a sentence matrix based on input x
    one_matrix = np.zeros((140, 140), dtype=np.float32)
    for index, item in enumerate(x):
        one_matrix[index][index] = 1
        if not item:
            one_matrix[index, item-1] = 2
            one_matrix[item-1, index] = 3
    return torch.FloatTensor(one_matrix)


# Setting seed for reproducibility
torch.manual_seed(1)

# Define hyperparameters
EPOCH = 200
BATCH_SIZE = 32
LR = 0.001
HIDDEN_NUM = 64
HIDDEN_LAYER = 2

# Load the CSV file
df = pd.read_csv('Train.csv')

# Extract features (X) and labels (Y)
X = df['Product_Description'].astype(str)  # Assuming 'Product_description' is the feature column
Y = df['Sentiment']  # Assuming 'Sentiment' is the target column

# Use label encoding to convert categorical labels to numerical labels
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)

# Split the dataset into training (80%) and validation (20%) sets
X_train, X_dev, y_train, y_dev = train_test_split(X, Y, test_size=0.2, random_state=42)

# Tokenize text using CountVectorizer (or any other method you prefer)
vectorizer = CountVectorizer(max_features=76215)  # Use the same max_features as in your original code
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_dev_vec = vectorizer.transform(X_dev).toarray()

# Convert to PyTorch tensors
train_x = torch.tensor(X_train_vec, dtype=torch.long)
train_y = torch.tensor(y_train, dtype=torch.long)
test_x = torch.tensor(X_dev_vec, dtype=torch.long)
test_y = torch.tensor(y_dev, dtype=torch.long)

# Create PyTorch datasets and data loaders
torch_dataset = Data.TensorDataset(train_x, train_y)
torch_testset = Data.TensorDataset(test_x, test_y)

loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = Data.DataLoader(
    dataset=torch_testset,
    batch_size=128
)

print("Data process finished")


class LSTM_GCN(nn.Module):
    def __init__(self):
        super(LSTM_GCN, self).__init__()
        self.embedding = nn.Embedding(76215, 300)
        self.lstm = nn.LSTM(
            input_size=300,  # Dimension of word vector
            hidden_size=180,  # Dimension of LSTM output
            num_layers=2,  # Number of LSTM layers
            batch_first=True,
            dropout=0.5,
            bidirectional=True
        )
        self.batch1 = nn.BatchNorm1d(140)
        self.gc = GraphConvolution(360, 7)  # Assuming input features for GCN is 360

        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                init.xavier_normal_(param, gain=1)

    def forward(self, x_and_adj):
        x = x_and_adj[:, :140]  # Assuming max_document_length is 140
        adj = x_and_adj[:, -140:]  # Assuming max_document_length is 140
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        out = self.batch1(lstm_out)
        adj_matrix = []
        for item in adj:
            adj_matrix.append(torch.unsqueeze(get_sentense_matrix(item), dim=0))
        adj_matrix = torch.cat(adj_matrix, dim=0)
        out_gcn = self.gc(out, adj_matrix)
        out = torch.median(out_gcn, 1)[0]
        return out


model = LSTM_GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-8)
loss_func = nn.CrossEntropyLoss()

best_accuracy = 0

def evaluate_model():
    global best_accuracy
    model.eval()
    predictions = []
    true_labels = []
    for batch_x, batch_y in test_loader:
        with torch.no_grad():
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(batch_y.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    precision_macro = precision_score(true_labels, predictions, average='macro')
    recall_macro = recall_score(true_labels, predictions, average='macro')
    f1_macro = f1_score(true_labels, predictions, average='macro')
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), "best_model.pth")
    
    print(f"Accuracy: {accuracy}")
    print(f"Macro Precision: {precision_macro}")
    print(f"Macro Recall: {recall_macro}")
    print(f"Macro F1-score: {f1_macro}")

    model.train()

# Training loop
for epoch in range(EPOCH):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = loss_func(outputs, batch_y)
        loss.backward()
        optimizer.step()

    # Evaluate the model after each epoch
    print(f"Epoch {epoch + 1}/{EPOCH}")
    evaluate_model()

print("Training finished.")
