import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.dropout_prob = 0.5
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size,hidden_size, num_layers, 
                            dropout=self.dropout_prob, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    
    def forward(self, features, captions):
        caps = self.embedding(captions[:, :-1])
        inputs = torch.cat((features.unsqueeze(1), caps), 1)
        self.hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
        lstm_out, self.hidden = self.lstm(inputs)
        
        outputs = self.linear(lstm_out)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        captions = []
        hidden = (torch.randn(1, 1, self.hidden_size).to(inputs.device),
                  torch.randn(1, 1, self.hidden_size).to(inputs.device))
        for i in range(max_len):
            lstm_out, hidden = self.lstm(inputs, hidden)
            outputs = self.linear(lstm_out)
            outputs = outputs.squeeze(1)
            best = outputs.argmax(1)
            captions.append(best.item())
            inputs = self.embedding(best.unsqueeze(0))
        
        return captions