import torch
import torch.nn as nn
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from pythainlp.tokenize import word_tokenize


def tokenize_th(s: str):
    return word_tokenize(s, engine="newmm")


SPECIALS = ["<pad>", "<sos>", "<eos>", "<unk>"]
PAD, SOS, EOS, UNK = SPECIALS


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(emb_dim, hid_dim, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)

    def forward(self, src):
        emb = self.embedding(src)       
        outputs, hidden = self.rnn(emb)     
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2], hidden[-1]), dim=1))) 
        return outputs, hidden


class AdditiveAttention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.W_h = nn.Linear(hid_dim, hid_dim, bias=False)
        self.W_s = nn.Linear(hid_dim, hid_dim, bias=False)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, s, h):
        src_len = h.size(0)
        s_rep = s.unsqueeze(0).repeat(src_len, 1, 1)  
        energy = torch.tanh(self.W_h(h) + self.W_s(s_rep))
        scores = self.v(energy).squeeze(2)              
        return torch.softmax(scores, dim=0)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, pad_idx, fc_in_dim):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)
        self.attn = AdditiveAttention(hid_dim)
        self.enc_proj = nn.Linear(hid_dim * 2, hid_dim)
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim)
        self.fc_out = nn.Linear(fc_in_dim, output_dim)

    def forward(self, input_tok, hidden, enc_outputs):
 
        input_tok = input_tok.unsqueeze(0)          
        emb = self.embedding(input_tok)              

        enc_h = self.enc_proj(enc_outputs)          
        attn = self.attn(hidden, enc_h)            
        context = torch.sum(attn.unsqueeze(2) * enc_h, dim=0).unsqueeze(0) 

        rnn_input = torch.cat((emb, context), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        output = output.squeeze(0) 
        hidden = hidden.squeeze(0)  
        emb = emb.squeeze(0)        
        context = context.squeeze(0)
        features = torch.cat((output, context, emb), dim=1)  
        pred = self.fc_out(features)                     
        return pred, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

CKPT_PATH = "mt_attention_ckpt.pt"

ckpt = torch.load(CKPT_PATH, map_location="cpu")
src_stoi, src_itos = ckpt["src_stoi"], ckpt["src_itos"]
tgt_stoi, tgt_itos = ckpt["tgt_stoi"], ckpt["tgt_itos"]
cfg = ckpt["config"]

STATE_KEY = "additive_state_dict"  
fc_w_key = "decoder.fc_out.weight"
if fc_w_key not in ckpt[STATE_KEY]:
    raise KeyError(f"Missing '{fc_w_key}' in ckpt[{STATE_KEY}] keys. "
                   f"Available example keys: {list(ckpt[STATE_KEY].keys())[:10]}")

fc_in_dim = ckpt[STATE_KEY][fc_w_key].shape[1]
print(f"Using STATE_KEY={STATE_KEY} | fc_in_dim(from ckpt)={fc_in_dim}")

encoder = Encoder(len(src_itos), cfg["EMB_DIM"], cfg["HID_DIM"], src_stoi[PAD])
decoder = Decoder(len(tgt_itos), cfg["EMB_DIM"], cfg["HID_DIM"], tgt_stoi[PAD], fc_in_dim=fc_in_dim)

model = Seq2Seq(encoder, decoder)
model.load_state_dict(ckpt[STATE_KEY], strict=True)
model.eval()

def translate(sentence: str, max_len: int = 30) -> str:
    sentence = (sentence or "").strip()
    if not sentence:
        return ""

    tokens = [SOS] + tokenize_th(sentence) + [EOS]
    src_ids = [src_stoi.get(t, src_stoi[UNK]) for t in tokens]
    src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(1) 

    with torch.no_grad():
        enc_out, hidden = model.encoder(src)

    cur = torch.tensor([tgt_stoi[SOS]], dtype=torch.long)
    result = []

    for _ in range(max_len):
        with torch.no_grad():
            pred, hidden = model.decoder(cur, hidden, enc_out)

        nxt = int(pred.argmax(1).item())
        tok = tgt_itos[nxt]

        if tok == EOS:
            break

        result.append(tok)
        cur = torch.tensor([nxt], dtype=torch.long)

    return " ".join(result)


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [

        dbc.Row(
            dbc.Col(
                html.Div(
                    [
                        html.H2(
                            "Thai → English Machine Translation",
                            className="text-center fw-bold"
                        ),
                        html.P(
                            f"Checkpoint: {CKPT_PATH} | Model: {STATE_KEY.replace('_state_dict','').title()} Attention",
                            className="text-center text-muted"
                        ),
                    ],
                    className="my-4",
                )
            )
        ),

        dbc.Row(
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Input (Thai sentence)"),
                        dbc.CardBody(
                            [
                                dcc.Textarea(
                                    id="input-text",
                                    placeholder="พิมพ์ประโยคภาษาไทยที่นี่...",
                                    style={
                                        "width": "100%",
                                        "height": "120px",
                                        "fontSize": "16px",
                                    },
                                ),
                                html.Br(),
                                dbc.Button(
                                    "Translate",
                                    id="btn",
                                    color="primary",
                                    className="mt-2",
                                ),
                            ]
                        ),
                    ],
                    className="shadow-sm",
                ),
                width=12,
            )
        ),

        html.Br(),

        dbc.Row(
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Output (English translation)"),
                        dbc.CardBody(
                            dbc.Spinner(
                                html.Div(
                                    id="output",
                                    style={
                                        "fontSize": "18px",
                                        "whiteSpace": "pre-wrap",
                                        "minHeight": "60px",
                                    },
                                ),
                                color="primary",
                            )
                        ),
                    ],
                    className="shadow-sm",
                ),
                width=12,
            )
        ),

        html.Br(),

        dbc.Row(
            dbc.Col(
                html.Small(
                    "NLP Assignment 3 – Seq2Seq with Additive Attention (Dash Demo)",
                    className="text-muted",
                ),
                className="text-center mb-3",
            )
        ),
    ],
    fluid=True)


@app.callback(
    Output("output", "children"),
    Input("btn", "n_clicks"),
    State("input-text", "value")
)
def run_translate(n, text):
    if not n:
        return ""
    return translate(text)


if __name__ == "__main__":
    app.run(debug=True)

