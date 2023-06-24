import lightning.pytorch as pl
import torch
import torchmetrics
from natasha import Segmenter, Doc
from torch import nn
from torch.utils.data import random_split, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


device = "cpu"
max_text_len = 945
max_token_len = 42
vocab_to_int = \
    {'я': 1,
     ' ': 2,
     'з': 3,
     'а': 4,
     'м': 5,
     'е': 6,
     'р': 7,
     'н': 8,
     'с': 9,
     'т': 10,
     ',': 11,
     'г': 12,
     'л': 13,
     'д': 14,
     'ы': 15,
     'в': 16,
     'п': 17,
     'у': 18,
     '.': 19,
     'и': 20,
     '-': 21,
     'о': 22,
     'э': 23,
     'ч': 24,
     'б': 25,
     'й': 26,
     'ь': 27,
     'к': 28,
     'ж': 29,
     'ш': 30,
     'ф': 31,
     'ц': 32,
     'щ': 33,
     'х': 34,
     'ю': 35,
     '!': 36,
     ':': 37,
     '«': 38,
     '»': 39,
     '—': 40,
     '(': 41,
     ')': 42,
     'ё': 43,
     'ъ': 44,
     '?': 45,
     ';': 46,
     'i': 47,
     'd': 48,
     'e': 49,
     's': 50,
     't': 51,
     '–': 52,
     '2': 53,
     '5': 54,
     '1': 55,
     '9': 56,
     '6': 57,
     '8': 58,
     '4': 59,
     'c': 60,
     'o': 61,
     'm': 62,
     'u': 63,
     'n': 64,
     'b': 65,
     'a': 66,
     '0': 67,
     '7': 68,
     'h': 69,
     'r': 70,
     'g': 71,
     'j': 72,
     'l': 73,
     "'": 74,
     'p': 75,
     'f': 76,
     '3': 77,
     '…': 78,
     'x': 79,
     'q': 80,
     'v': 81,
     '"': 82,
     '№': 83,
     '/': 84,
     '@': 85,
     'y': 86,
     'z': 87,
     '¬': 88,
     '%': 89,
     '[': 90,
     ']': 91,
     '_': 92,
     'k': 93,
     'w': 94,
     '<': 95,
     '>': 96,
     '+': 97,
     '“': 98,
     '”': 99,
     '•': 100,
     '€': 101,
     '$': 102,
     '`': 103,
     '&': 104,
     '{': 105,
     '}': 106,
     '*': 107,
     '°': 108,
     '·': 109,
     '§': 110,
     '\xa0': 111,
     '’': 112,
     'PAD': 0}
vocab_size = len(vocab_to_int.keys())


class CustomConv1D(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 dilation: int):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                                padding=kernel_size // 2, dilation=dilation)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.activation(x)
        return x


class CharCNN(nn.Module):
    def __init__(self,
                 char_embed_size: int,
                 layers_n: int,
                 kernel_size: int,
                 dilation: int):
        super().__init__()
        self.block_list = nn.ModuleList([CustomConv1D(in_channels=char_embed_size, out_channels=char_embed_size,
                                                      kernel_size=kernel_size, dilation=dilation) for i in
                                         range(layers_n)])

    def forward(self, x):
        for block in self.block_list:
            x = x + block(x)
        return x


class ClassificationHead(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class TokenCNN(nn.Module):
    def __init__(self,
                 in_embed_size: int,
                 context_embed_size: int):
        super().__init__()
        self.conv_1_1 = CustomConv1D(in_channels=in_embed_size, out_channels=context_embed_size, kernel_size=7,
                                     dilation=1)
        self.conv_1_2 = CustomConv1D(in_channels=context_embed_size, out_channels=context_embed_size, kernel_size=3,
                                     dilation=1)
        self.pooling_1 = nn.MaxPool1d(kernel_size=3, stride=3)

        self.conv_2_1 = CustomConv1D(in_channels=context_embed_size, out_channels=context_embed_size, kernel_size=7,
                                     dilation=1)
        self.conv_2_2 = CustomConv1D(in_channels=context_embed_size, out_channels=context_embed_size, kernel_size=3,
                                     dilation=1)
        self.pooling_2 = nn.MaxPool1d(kernel_size=3, stride=3)

        self.conv_3_1 = CustomConv1D(in_channels=context_embed_size, out_channels=context_embed_size, kernel_size=7,
                                     dilation=1)
        self.conv_3_2 = CustomConv1D(in_channels=context_embed_size, out_channels=context_embed_size, kernel_size=3,
                                     dilation=1)
        self.pooling_3 = nn.MaxPool1d(kernel_size=3, stride=3)

        self.conv_4_1 = CustomConv1D(in_channels=context_embed_size, out_channels=context_embed_size, kernel_size=7,
                                     dilation=1)
        self.conv_4_2 = CustomConv1D(in_channels=context_embed_size, out_channels=context_embed_size, kernel_size=3,
                                     dilation=1)
        self.pooling_4 = nn.MaxPool1d(kernel_size=3, stride=3)

    def forward(self, x):
        x = self.conv_1_1(x)
        x = x + self.conv_1_2(x)
        x = self.pooling_1(x)

        x = self.conv_2_1(x)
        x = x + self.conv_2_2(x)
        x = self.pooling_2(x)

        x = self.conv_3_1(x)
        x = x + self.conv_3_2(x)
        x = self.pooling_3(x)

        x = self.conv_4_1(x)
        x = x + self.conv_4_2(x)
        x = self.pooling_4(x)
        return x


class TokenCNNRNN(nn.Module):
    def __init__(self,
                 max_text_len: int,
                 in_embed_size: int,
                 context_embed_size: int,
                 num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.context_embed_size = context_embed_size

        self.conv_1_1 = CustomConv1D(in_channels=in_embed_size, out_channels=context_embed_size, kernel_size=7,
                                     dilation=1)
        self.conv_1_2 = CustomConv1D(in_channels=context_embed_size, out_channels=context_embed_size, kernel_size=3,
                                     dilation=1)
        self.pooling_1 = nn.MaxPool1d(kernel_size=3, stride=3)

        self.ltsm = nn.LSTM(max_text_len // 3,
                            context_embed_size,
                            num_layers=num_layers,
                            bidirectional=True,
                            dropout=0.1,
                            batch_first=True
                            )
        self.global_pooling = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = self.conv_1_1(x)
        x = x + self.conv_1_2(x)
        x = self.pooling_1(x)

        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.context_embed_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.context_embed_size).to(device)
        out, (hidden_state, cell_state) = self.ltsm(x, (h0, c0))
        return self.global_pooling(out).squeeze(-1)


class Network(nn.Module):
    def __init__(self,
                 cnn_rnn,
                 vocab_size,
                 num_classes,
                 max_text_len,
                 char_embedding_size=64,
                 token_embedding_size=256,
                 classifier_dropout=0.5):
        super().__init__()
        self.cnn_rnn = cnn_rnn
        self.char_embedding_size = char_embedding_size
        self.char_embeddings = nn.Embedding(vocab_size, char_embedding_size, padding_idx=0)
        self.char_cnn = CharCNN(char_embedding_size, layers_n=10, kernel_size=3, dilation=1)
        self.global_pooling_chars = nn.AdaptiveMaxPool1d(1)
        self.token_cnn = TokenCNN(in_embed_size=char_embedding_size, context_embed_size=token_embedding_size)
        self.token_rnn = TokenCNNRNN(max_text_len=max_text_len, in_embed_size=char_embedding_size,
                                     context_embed_size=token_embedding_size, num_layers=1)
        self.global_pooling_context = nn.AdaptiveMaxPool1d(1)
        self.classification_head = ClassificationHead(in_features=token_embedding_size, out_features=num_classes,
                                                      dropout=classifier_dropout)

    def forward(self, tokens):
        batch_size, max_text_len, max_token_len = tokens.shape  # BatchSize x MaxTextLen x MaxTokenLen
        tokens_flat = tokens.view(batch_size * max_text_len, max_token_len)  # BatchSize*MaxTextLen x MaxTokenLen

        char_embeddings = self.char_embeddings(tokens_flat)  # BatchSize*MaxTextLen x MaxTokenLen x EmbSize
        char_embeddings = char_embeddings.permute(0, 2, 1)  # BatchSize*MaxTextLen x EmbSize x MaxTokenLen
        char_features = self.char_cnn(char_embeddings)  # BatchSize*MaxTextLen x EmbSize x MaxTokenLen

        token_features = self.global_pooling_chars(char_features).squeeze(-1)  # BatchSize*MaxTextLen x EmbSize
        token_features = token_features.view(batch_size, max_text_len,
                                             self.char_embedding_size)  # BatchSize x MaxTextLen x EmbSize
        token_features = token_features.permute(0, 2, 1)  # BatchSize x EmbSize x MaxTextLen

        if not self.cnn_rnn:
            context_features = self.token_cnn(token_features)  # BatchSize x EmbSize x MaxTextLen
            text_features = self.global_pooling_context(context_features).squeeze(-1)  # BatchSize x EmbSize
        else:
            text_features = self.token_rnn(token_features)  # BatchSize x EmbSize
        logits = self.classification_head(text_features)  # BatchSize x num_classes
        return logits


class DataModule(pl.LightningDataModule):
    def __init__(self,
                 tensor_dataset,
                 batch_size):
        super().__init__()
        self.tensor_dataset = tensor_dataset
        self.batch_size = batch_size
        self.prepare_data()

    def setup(self, stage=None):
        train_size = int(0.8 * len(self.tensor_dataset))
        val_size = len(self.tensor_dataset) - train_size
        self.train_data, self.val_data = random_split(self.tensor_dataset, [train_size, val_size])
        return self.train_data, self.val_data

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=2)


class ModelCompilation(pl.LightningModule):
    def __init__(self,
                 model: torch.nn.Module,
                 metrics: dict,
                 loss_function,
                 optimizer: torch.optim,
                 learning_rate: float):
        super().__init__()
        self.model = model
        self.metrics = metrics
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.learning_rate = learning_rate

    def forward(self, x):
        pred = self.model.forward(x)
        return pred

    def configure_optimizers(self):
        train_optimizer = self.optimizer(self.parameters(), lr=self.learning_rate, weight_decay=0.05, betas=(0.9, 0.98),
                                         eps=1.0e-9)
        train_scheduler = {
            "scheduler": ReduceLROnPlateau(optimizer=train_optimizer, mode="min", factor=0.5, patience=3, min_lr=5e-6),
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
        }
        return [train_optimizer], [train_scheduler]

    def training_step(self, batch, batch_idx):
        loss, pred, y = self.common_step(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred, y = self.common_step(batch, batch_idx, 'val')
        return loss

    def test_step(self, batch, batch_idx):
        loss, pred, y = self.common_step(batch, batch_idx, 'test')
        return loss

    def common_step(self, batch, batch_idx, stage):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss_function(pred, y)
        if (stage == 'test') or (stage == 'val'):
            on_step = False
        else:
            on_step = True

        [self.log(stage + '_' + metric_name, metric(pred, y).to(device), on_step=on_step, on_epoch=True, prog_bar=True,
                  logger=True) for metric_name, metric in self.metrics.items()]
        self.log(stage + '_' + 'loss', loss, on_step=on_step, on_epoch=True, prog_bar=True, logger=True)
        return loss, pred, y


# class History(Callback):
#     def __init__(self):
#         self.history = {'val_loss': np.array([]), 'val_accuracy': np.array([]), 'train_loss_epoch': np.array([]),
#                         'train_accuracy_epoch': np.array([])}
#
#     def on_train_epoch_end(self, trainer, module):
#         logs = trainer.logged_metrics
#         self.history['train_loss_epoch'] = np.append(self.history['train_loss_epoch'], logs['train_loss_epoch'].cpu())
#         self.history['val_loss'] = np.append(self.history['val_loss'], logs['val_loss'].cpu())
#         self.history['train_accuracy_epoch'] = np.append(self.history['train_accuracy_epoch'],
#                                                          logs['train_accuracy_epoch'].cpu())
#         self.history['val_accuracy'] = np.append(self.history['val_accuracy'], logs['val_accuracy'].cpu())
#

#
# def fit(iteration):
#     num_classes = 5
#     task = 'multiclass'
#     network = Network(cnn_rnn=False,
#                       vocab_size=vocab_size,
#                       num_classes=num_classes,
#                       max_text_len=max_text_len,
#                       char_embedding_size=64,
#                       token_embedding_size=256,
#                       classifier_dropout=0.5
#                       )
#
#     metrics = {'accuracy': torchmetrics.Accuracy(task=task, num_classes=num_classes).to(device)}
#     loss_function = nn.CrossEntropyLoss()
#     optimizer = torch.optim.AdamW
#     learning_rate = 0.0005
#
#     earlystopping_callback = EarlyStopping(monitor="val_loss", mode="min", patience=5)
#     checkpoint_callback = ModelCheckpoint(monitor='val_loss', filename='model-{epoch:02d}-{val_loss:.2f}')
#     history_callback = History()
#
#     modelC = ModelCompilation(network, metrics, loss_function, optimizer, learning_rate)
#     trainer = pl.Trainer(callbacks=[earlystopping_callback, history_callback, checkpoint_callback], precision='32',
#                          accelerator=device, devices=1, max_epochs=50)
#
#     trainer.fit(modelC, datamodule=data_module)
#     print("Best model score for iteration " + str(iteration) + " : " + str(checkpoint_callback.best_model_score))
#     return checkpoint_callback, history_callback


# device = 'cuda'
# trials = 10
# best_score = 100
# best_model_path = ''
# data_module = DataModule(tensor_dataset=train_dataset, batch_size=8)
# for i in range(trials):
#     checkpoint, history = fit(i)
#     if checkpoint.best_model_score < best_score:
#         best_score = checkpoint.best_model_score
#         best_model_path = checkpoint.best_model_path
#         best_history = history
#
# print("Best model score for " + str(trials) + " trials: " + str(best_score))
# print("Best model path: " + best_model_path)


# def plot_train_metrics(history_callback):
#     plt.figure()
#     plt.plot(list(range(len(history_callback.history['train_loss_epoch']))),
#              history_callback.history['train_loss_epoch'])
#     plt.plot(list(range(len(history_callback.history['val_loss']))), history_callback.history['val_loss'])
#     plt.legend(['train_loss', 'val_loss'])
#
#     plt.figure()
#     plt.plot(list(range(len(history_callback.history['train_accuracy_epoch']))),
#              history_callback.history['train_accuracy_epoch'])
#     plt.plot(list(range(len(history_callback.history['val_accuracy']))), history_callback.history['val_accuracy'])
#     plt.legend(['train_accuracy', 'val_accuracy'])
#     plt.show()
#
#
# plot_train_metrics(best_history)


def load_pretrained_model(checkpoint_path):
    num_classes = 5

    task = 'multiclass'
    network = Network(num_classes=num_classes,
                      cnn_rnn=False,
                      vocab_size=vocab_size,
                      max_text_len=max_text_len,
                      token_embedding_size=256,
                      classifier_dropout=0.5)

    metrics = {'accuracy': torchmetrics.Accuracy(task=task, num_classes=num_classes).to("cpu")}
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW
    learning_rate = 0.0005

    model = ModelCompilation(network, metrics, loss_function, optimizer, learning_rate)
    model = model.load_from_checkpoint(checkpoint_path, map_location=torch.device('cpu'))
    return model


def predictStyleForText(text, model):
    texts_tensor = torch.zeros(1, max_text_len, max_token_len + 2).long()
    text = text.lower()
    segmenter = Segmenter()
    doc = Doc(text)
    doc.segment(segmenter)
    text_tokens = [token.text for token in doc.tokens]
    for token_i, token in enumerate(text_tokens):
        for char_i, char in enumerate(token):
            texts_tensor[0][token_i, char_i + 1] = vocab_to_int[char]

    model.eval()
    logits = model.forward(texts_tensor)
    proba = torch.nn.functional.softmax(logits.data, dim=1)
    return proba


def predict(text, mode):
    proba = None
    if mode == "CharCNN + TokenCNN":
        model = load_pretrained_model("models/model_sentence_cnn.ckpt")
        proba = predictStyleForText(text, model)
    elif mode == "CharCNN + TokenCNNRNN":
        model = load_pretrained_model("models/model_sentence_cnnrnn.ckpt")
        proba = predictStyleForText(text, model)
    return proba
