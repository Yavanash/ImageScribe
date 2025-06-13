import re
from pickle import load
import gradio as gr

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torchvision.transforms as transforms

class Vocabulary:
    def __init__(self, min_freq=5):
        self.min_freq = min_freq
        self.itos = {0:"pad", 1:"startofseq", 2:"endofseq", 3:"unk"}
        self.stoi = {v:k for k,v in self.itos.items()}
        self.index = 4
    
    def __len__(self):
        return len(self.itos)
    
    def tokenizer(self, text):
        text = text.lower()
        tokens = re.findall(r"\w+", text) #splits the text into tokens based on punctuation
        return tokens

    def build_vocab(self, sentence_list):
        count = {}
        for sentence in sentence_list:
            tokens = self.tokenizer(sentence)
            for token in tokens:
                if token not in count:
                    count[token] = 1
                else:
                    count[token] += 1
        
        for token, freq in count.items():
            # print(f"{token}:{freq}")
            if freq>=self.min_freq:
                #if freq>=min_freq then add it to vocab
                self.itos[self.index] = token
                self.stoi[token] = self.index
                self.index += 1
    
    def change_to_nums(self, text):
        tokens = self.tokenizer(text)
        nums = []
        for token in tokens:
            if token in self.stoi:
                nums.append(self.stoi[token])
            else:
                nums.append(self.stoi["unk"])
        return nums
    
EMBED_DIM = 256
HIDDEN_DIM = 512
MAX_SEQ_LENGTH = 25
DEVICE = "cpu"

MODEL_SAVE_PATH = "checkpoints/final_model.pth"

with open("vocab/vocab.pkl", "rb") as f:
    vocab = load(f)
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

class ResNetEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad=True #this is so that we can fine tune the model
        #we need to drop the last linear layer and change it to an embedding 
        modules = list(model.children())[:-1] #dropping the last layer

        self.model = nn.Sequential(*modules)
        self.fc = nn.Linear(model.fc.in_features, embed_dim) #input is same as fc but output is embedding space
        self.batchnorm = nn.BatchNorm1d(embed_dim, momentum=0.01)

    def forward(self, imgs):
        print(f"img before model {imgs.size()}")
        with torch.no_grad():
            features = self.model(imgs)
        #these are as [batch size, model.fc.in_features, 1, 1] therefore we need to flatten
        features = features.view(features.size(0), -1)
        features = self.fc(features) #this means features is [batch, embed dim]
        features = self.batchnorm(features)
        print(f"img after model {imgs.size()}")
        
        return features
    
class LSTMDecoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size, num_layers=1):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim) #it creates a dense vector of size vocab_size and dims of embed_dim
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions, states):
        captions_in = captions #captions is the padded tensor
        embed = self.embeddings(captions_in)
        #lstm needs shape as [batch size, seq length, vocab size]
        features = features.unsqueeze(1) #as features are of shape [batchsize, 2048] so we make [batchsize, 1, 2048] as lstm needs this shape
        # print(f"shapes: {features.size()} {embed.size()}")
        lstm_input = torch.cat((features, embed), dim=1) #we are concating these 2 along dim=1(seq length) so final is (seq length+1) there fore first token is image
        outputs, returned_states = self.lstm(lstm_input, states)
        logits = self.fc(outputs)
        
        return logits, returned_states
    
    def generate(self, features, max_len=20):
        batch_size = features.size(0)
        states = None
        generated_captions = []

        start_idx = 1#startofseq
        end_idx = 2#endofseq
        current_tokens = [start_idx]

        for _ in range(max_len):
            input_tokens = torch.LongTensor(current_tokens).to(features.device).unsqueeze(0)
            logits, states = self.forward(features, input_tokens, states)
            logits = logits.contiguous().view(-1, vocab_size)
            preds = logits.argmax(dim=1)[-1].item()

            generated_captions.append(preds)
            current_tokens.append(preds)

        return generated_captions
    
class FullModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def generate(self, imgs, max_len=MAX_SEQ_LENGTH):
        features = self.encoder(imgs)
        outputs = self.decoder.generate(features, max_len=MAX_SEQ_LENGTH)

        return outputs
    
def load_model():
    encoder = ResNetEncoder(EMBED_DIM)
    decoder = LSTMDecoder(EMBED_DIM, HIDDEN_DIM, vocab_size)
    model = FullModel(encoder, decoder)

    state_dict = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict["model_state_dict"])

    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = load_model()

def generate_caption(img):
    img = img.convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output_indices = model.generate(img_tensor, max_len=MAX_SEQ_LENGTH)

    result_words = []
    end_token_idx = 2

    for idx in output_indices:
        if idx == end_token_idx:
            break
        
        word = vocab.itos.get(idx, "unk")
        if word not in ["startofseq", "pad", "endofseq"]:
            result_words.append(word)
    
    return " ".join(result_words)

def main():
    iface = gr.Interface(
        fn=generate_caption,
        inputs=gr.Image(type="pil"),
        outputs="text",
        title="Image Captioning.....(Incomplete)",
        description="Upload an image to get a generated caption from the trained model.",
    )

    iface.launch(share=True)

if __name__=="__main__":
    print("Loaded model...Loading Gradio Interface")
    main()