"""
CheXReport AI — app.py  (FIXED)
================================
Bug fixes applied (see comments tagged FIX-1 … FIX-4):

  FIX-1  inp.value reset timing  ─ inp.value='' was being executed
         synchronously inside the 'change' handler, BEFORE the
         FileReader async callback fires.  On Safari/WebKit and certain
         Chromium builds this can release the File object reference
         prematurely, causing the read to silently produce no result and
         the upload zone to stay empty.
         → Moved inp.value='' to the FIRST line of reader.onload so the
           reset only happens after the base-64 data is safely captured.

  FIX-2  File-type validation rejected valid images  ─ the guard
         `file.type.startsWith('image/')` evaluates to false whenever
         file.type is an empty string.  file.type is '' in many
         OS / browser combos (macOS Finder drag-and-drop, Windows
         Explorer file picker, some mobile browsers, HF Spaces iframe
         context).  This silently aborts every upload attempt on those
         platforms while showing "Invalid file" in the status bar.
         → Added an extension-based fallback:
           /\.(png|jpg|jpeg)$/i.test(file.name)
           so any file whose name ends in .png / .jpg / .jpeg is
           accepted even when the MIME type is unavailable.

  FIX-3  Error display showed "Error: undefined"  ─ the catch block
         used err.message, but for non-Error throw values (plain strings,
         fetch AbortError, Gradio timeout, etc.) .message is undefined.
         → Changed to err.message || String(err) so the actual error is
           always visible to the user.

  FIX-4  Missing gradio version pin in requirements.txt  ─ server_functions
         on gr.HTML is a Gradio 5.x feature.  Without a version pin HF
         Spaces can install an older Gradio where the parameter is
         silently ignored, making server.run() undefined in JS and
         crashing the Generate step.
         → Add  gradio>=5.0  to requirements.txt  (separate file — not
           changed here, but noted).  The gr.HTML call below is correct
           for Gradio 5.x.

Dependencies (requirements.txt must contain):
  torch
  torchvision
  transformers
  gradio>=5.0          ← FIX-4: must be present and pinned
  Pillow
  huggingface_hub
  sacremoses
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from PIL import Image
import gradio as gr
import re, datetime, base64
from io import BytesIO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on: {device}')

# ── Architecture ───────────────────────────────────────────────────────────────

class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        densenet      = models.densenet121(weights=None)
        self.features = densenet.features
        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        f = self.features(x)
        f = torch.relu(f)
        f = self.avgpool(f)
        return f.flatten(1)

class ProjectionLayer(nn.Module):
    def __init__(self, vision_dim=1024, lm_hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, vision_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(vision_dim, lm_hidden_dim)
        )

    def forward(self, x):
        return self.projection(x).unsqueeze(1)

class CheXReportModel(nn.Module):
    def __init__(self, ve, pl, lm, tok):
        super().__init__()
        self.vision_encoder = ve
        self.projection     = pl
        self.lm             = lm
        self.tokenizer      = tok

# ── Load model ─────────────────────────────────────────────────────────────────

print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('microsoft/biogpt')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print('Loading BioGPT...')
biogpt = AutoModelForCausalLM.from_pretrained('microsoft/biogpt')

vision_encoder   = VisionEncoder()
projection_layer = ProjectionLayer(1024, biogpt.config.hidden_size)
model            = CheXReportModel(vision_encoder, projection_layer, biogpt, tokenizer).to(device)

print('Downloading weights...')
weights_path = hf_hub_download(
    repo_id   = 'muhammedpanchla/CheXReport',
    filename  = 'chexreport_best.pth',
    repo_type = 'model'
)
checkpoint = torch.load(weights_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f'Model ready — epoch {checkpoint["epoch"]} | val_loss {checkpoint["val_loss"]:.4f}')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── Inference ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def run(data_url: str) -> str:
    """
    Accepts a base64 data URL (data:image/...;base64,...) from the browser,
    runs the CheXReport pipeline, and returns the formatted report string.
    """
    try:
        if ',' in data_url:
            data_url = data_url.split(',', 1)[1]
        img_bytes = base64.b64decode(data_url)
        image     = Image.open(BytesIO(img_bytes)).convert('RGB')

        image_tensor    = transform(image).unsqueeze(0).to(device)
        vision_features = model.vision_encoder(image_tensor)
        visual_prefix   = model.projection(vision_features)

        bos_id        = tokenizer.bos_token_id or tokenizer.eos_token_id
        bos_tensor    = torch.tensor([[bos_id]], device=device)
        bos_embedding = model.lm.biogpt.embed_tokens(bos_tensor)
        combined      = torch.cat([visual_prefix, bos_embedding], dim=1)

        output_ids = model.lm.generate(
            inputs_embeds        = combined,
            max_new_tokens       = 150,
            num_beams            = 4,
            no_repeat_ngram_size = 3,
            repetition_penalty   = 1.3,
            early_stopping       = True,
            eos_token_id         = tokenizer.eos_token_id,
            pad_token_id         = tokenizer.pad_token_id,
        )

        raw = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        raw = re.sub(r'\bXXXX\b', '', raw)
        raw = re.sub(r'\s{2,}', ' ', raw).strip()
        raw = re.sub(r'\s+\.', '.', raw)
        raw = re.sub(r'\.\.+', '.', raw)

        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', raw) if s.strip()]
        if len(sentences) >= 3:
            findings   = ' '.join(sentences[:-1])
            impression = sentences[-1]
        else:
            findings   = ' '.join(sentences)
            impression = 'No discrete acute cardiopulmonary process identified. Clinical correlation recommended.'

        today = datetime.date.today().strftime('%d %b %Y')
        return (
            f"MODALITY : Chest X-Ray (PA/AP)\n"
            f"DATE     : {today}\n"
            f"{'─' * 52}\n\n"
            f"INDICATION\nChest X-ray submitted for AI-assisted evaluation.\n\n"
            f"COMPARISON\nNo prior studies available for comparison.\n\n"
            f"FINDINGS\n{findings}\n\n"
            f"IMPRESSION\n{impression}\n\n"
            f"{'─' * 52}\n"
            f"AI LIMITATION: BioGPT priors may introduce findings\n"
            f"not visible in the image. Verify with a radiologist."
        )
    except Exception as e:
        return f"ERROR: {str(e)}"


# ══════════════════════════════════════════════════════════════════════════════
#  HTML — full custom UI
# ══════════════════════════════════════════════════════════════════════════════

HTML = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Fraunces:ital,wght@0,700;1,700&family=JetBrains+Mono:wght@400;500&display=swap');

#cx *,#cx *::before,#cx *::after{box-sizing:border-box;margin:0;padding:0}
#cx{font-family:'Outfit',sans-serif;background:#f5f6f8;color:#111318;width:100%;overflow-x:hidden}

/* topbar */
#cx .r-topbar{height:3px;background:linear-gradient(90deg,#1a56db,#0e9a8c,#14b8a6)}

/* nav */
#cx .r-nav{background:rgba(255,255,255,.97);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);border-bottom:1px solid #e2e5eb;position:sticky;top:0;z-index:200}
#cx .r-nav-inner{max-width:1240px;margin:0 auto;padding:0 32px;height:58px;display:flex;align-items:center;justify-content:space-between}
#cx .r-logo{display:flex;align-items:center;gap:12px}
#cx .r-logo-name{font-family:'Fraunces',serif;font-size:16px;font-weight:700;color:#111318;letter-spacing:-.3px;line-height:1}
#cx .r-logo-sub{font-size:9px;font-weight:500;letter-spacing:2px;text-transform:uppercase;color:#8a93a6}
#cx .r-pill{display:flex;align-items:center;gap:6px;font-size:11px;font-weight:500;color:#454c5c;padding:5px 14px;border-radius:20px;border:1px solid #e2e5eb;background:#f5f6f8}
#cx .r-live{width:6px;height:6px;background:#16a34a;border-radius:50%;position:relative;flex-shrink:0}
#cx .r-live::after{content:'';position:absolute;inset:-3px;border-radius:50%;background:#16a34a;opacity:.25;animation:ripple 2s ease-out infinite}
@keyframes ripple{0%{transform:scale(1);opacity:.25}100%{transform:scale(2.8);opacity:0}}
#cx .r-nav-right{display:flex;align-items:center;gap:12px}
#cx .r-nav-link{font-size:12px;font-weight:500;letter-spacing:.5px;color:#8a93a6;text-decoration:none;transition:color .2s}
#cx .r-nav-link:hover{color:#111318}
#cx .r-nav-cta{font-size:12px;font-weight:600;padding:7px 16px;background:#111318;color:#fff;border:none;border-radius:8px;cursor:pointer;transition:background .2s}
#cx .r-nav-cta:hover{background:#454c5c}

/* wrap */
#cx .r-wrap{max-width:1240px;margin:0 auto;padding:0 32px}

/* hero */
#cx .r-hero{padding:64px 0 52px;display:grid;grid-template-columns:1fr 460px;gap:64px;align-items:center}
#cx .r-eyebrow{display:inline-flex;align-items:center;gap:8px;font-size:10px;font-weight:600;letter-spacing:3px;text-transform:uppercase;color:#0e9a8c;margin-bottom:20px}
#cx .r-eyebrow-dash{width:20px;height:1.5px;background:#0e9a8c}
#cx .r-h1{font-family:'Fraunces',serif;font-size:58px;line-height:1.03;letter-spacing:-2px;color:#111318;margin-bottom:20px;font-weight:700}
#cx .r-h1 em{font-style:italic;color:#0e9a8c}
#cx .r-desc{font-size:14px;line-height:1.85;color:#454c5c;max-width:440px;margin-bottom:36px;font-weight:300}
#cx .r-stats{display:flex;align-items:center;gap:24px}
#cx .r-stat-val{font-family:'JetBrains Mono',monospace;font-size:22px;font-weight:500;color:#111318;letter-spacing:-1px;line-height:1;margin-bottom:4px}
#cx .r-stat-lbl{font-size:9px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:#8a93a6}
#cx .r-hsep{width:1px;height:26px;background:#cdd1da}

/* x-ray hero card */
#cx .r-xcard{background:#0b0e14;border-radius:16px;overflow:hidden;box-shadow:0 24px 64px rgba(0,0,0,.32)}
#cx .r-xcard-hd{padding:12px 16px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid rgba(255,255,255,.05)}
#cx .r-xcard-title{font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:2px;text-transform:uppercase;color:rgba(255,255,255,.3)}
#cx .r-xdots{display:flex;gap:5px}
#cx .r-xdots span{width:8px;height:8px;border-radius:50%}
#cx .r-xdots span:nth-child(1){background:#ff5f57}
#cx .r-xdots span:nth-child(2){background:#febc2e}
#cx .r-xdots span:nth-child(3){background:#28c840}
#cx .r-xvp{position:relative;height:280px;background:#000;overflow:hidden;display:flex;align-items:center;justify-content:center}
#cx .r-xgrid{position:absolute;inset:0;background-image:linear-gradient(rgba(20,184,166,.05) 1px,transparent 1px),linear-gradient(90deg,rgba(20,184,166,.05) 1px,transparent 1px);background-size:28px 28px}
#cx .r-xray-body{width:170px;height:245px;position:relative;opacity:.6}
#cx .r-xr-spine{position:absolute;left:46%;top:4%;width:8%;height:92%;background:linear-gradient(180deg,rgba(255,255,255,.35),rgba(255,255,255,.18));border-radius:4px}
#cx .r-xr-heart{position:absolute;left:28%;top:22%;width:44%;height:52%;background:radial-gradient(ellipse at 45% 50%,rgba(255,255,255,.22) 0%,rgba(255,255,255,.06) 65%,transparent 100%);border-radius:50%}
#cx .r-xr-rib{position:absolute;left:5%;height:3px;border-radius:2px;background:rgba(255,255,255,.18)}
#cx .r-xr-rib:nth-child(1){top:8%;width:90%}
#cx .r-xr-rib:nth-child(2){top:20%;width:92%}
#cx .r-xr-rib:nth-child(3){top:32%;width:88%}
#cx .r-xr-rib:nth-child(4){top:44%;width:82%}
#cx .r-xr-rib:nth-child(5){top:56%;width:76%}
#cx .r-xr-rib:nth-child(6){top:68%;width:68%}
#cx .r-scanline{position:absolute;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,#14b8a6,transparent);box-shadow:0 0 10px #14b8a6;animation:scan 3s ease-in-out infinite;top:0}
@keyframes scan{0%{top:0;opacity:0}8%{opacity:.9}90%{opacity:.9}100%{top:100%;opacity:0}}
#cx .r-xcorner{position:absolute;width:14px;height:14px;border-color:rgba(20,184,166,.6);border-style:solid}
#cx .r-xcorner.tl{top:8px;left:8px;border-width:1.5px 0 0 1.5px}
#cx .r-xcorner.tr{top:8px;right:8px;border-width:1.5px 1.5px 0 0}
#cx .r-xcorner.bl{bottom:8px;left:8px;border-width:0 0 1.5px 1.5px}
#cx .r-xcorner.br{bottom:8px;right:8px;border-width:0 1.5px 1.5px 0}
#cx .r-xpip{position:absolute;top:10px;right:10px;background:rgba(0,0,0,.75);border:1px solid rgba(20,184,166,.25);border-radius:6px;padding:7px 10px;font-family:'JetBrains Mono',monospace;font-size:9px;color:#14b8a6;line-height:1.7}
#cx .r-xcard-ft{padding:11px 16px;border-top:1px solid rgba(255,255,255,.05);display:flex;align-items:center;justify-content:space-between}
#cx .r-xcard-ft-text{font-family:'JetBrains Mono',monospace;font-size:9px;color:rgba(255,255,255,.2)}
#cx .r-xbadge{font-size:9px;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;padding:4px 10px;border-radius:4px;background:rgba(22,163,74,.12);color:#4ade80;border:1px solid rgba(22,163,74,.2)}

/* arch strip */
#cx .r-arch{background:#fff;border-top:1px solid #e2e5eb;border-bottom:1px solid #e2e5eb}
#cx .r-arch-inner{max-width:1240px;margin:0 auto;display:flex}
#cx .r-step{flex:1;display:flex;align-items:center;gap:12px;padding:18px 20px;border-right:1px solid #e2e5eb;cursor:default;transition:background .2s}
#cx .r-step:last-child{border-right:none}
#cx .r-step:hover{background:#fafbfc}
#cx .r-step-num{width:24px;height:24px;border-radius:50%;background:#111318;color:#fff;font-size:10px;font-weight:700;display:flex;align-items:center;justify-content:center;flex-shrink:0;font-family:'JetBrains Mono',monospace}
#cx .r-step-name{font-size:12px;font-weight:600;color:#111318;margin-bottom:2px}
#cx .r-step-sub{font-size:10px;color:#8a93a6;font-family:'JetBrains Mono',monospace}

/* section */
#cx .r-section{padding:48px 0 0}
#cx .r-sec-head{display:flex;align-items:flex-end;justify-content:space-between;margin-bottom:24px}
#cx .r-sec-eye{font-size:10px;font-weight:600;letter-spacing:2.5px;text-transform:uppercase;color:#0e9a8c;margin-bottom:6px}
#cx .r-sec-title{font-family:'Fraunces',serif;font-size:32px;color:#111318;letter-spacing:-.5px;font-weight:700}
#cx .r-sec-right{font-family:'JetBrains Mono',monospace;font-size:11px;color:#8a93a6;text-align:right;line-height:1.7}

/* stat cards */
#cx .r-statrow{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:40px}
#cx .r-sc{background:#fff;border:1px solid #e2e5eb;border-radius:12px;padding:16px;transition:all .25s;position:relative;overflow:hidden;cursor:default}
#cx .r-sc:hover{border-color:#0e9a8c;box-shadow:0 0 0 3px rgba(14,154,140,.06)}
#cx .r-sc::after{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#1a56db,#0e9a8c);opacity:0;transition:opacity .3s}
#cx .r-sc:hover::after{opacity:1}
#cx .r-sc-val{font-family:'JetBrains Mono',monospace;font-size:16px;font-weight:500;color:#111318;letter-spacing:-.5px;margin-bottom:5px}
#cx .r-sc-lbl{font-size:9px;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;color:#8a93a6}

/* panels */
#cx .r-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}
#cx .r-panel{background:#fff;border:1px solid #e2e5eb;border-radius:14px;overflow:hidden;display:flex;flex-direction:column}
#cx .r-panel-hd{padding:13px 20px;border-bottom:1px solid #e2e5eb;background:#fafbfc;display:flex;align-items:center;justify-content:space-between;flex-shrink:0}
#cx .r-panel-left{display:flex;align-items:center;gap:8px}
#cx .r-pdot{width:6px;height:6px;border-radius:50%;background:#0e9a8c;box-shadow:0 0 6px #0e9a8c;flex-shrink:0}
#cx .r-pdot.dim{background:#cdd1da;box-shadow:none}
#cx .r-pdot.green{background:#16a34a;box-shadow:0 0 6px #16a34a}
#cx .r-panel-label{font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#454c5c}
#cx .r-panel-chip{font-family:'JetBrains Mono',monospace;font-size:10px;color:#8a93a6;background:#f5f6f8;border:1px solid #e2e5eb;padding:3px 8px;border-radius:4px}
#cx .r-panel-body{padding:16px;flex:1;display:flex;flex-direction:column}

#cx .r-upload{
  border:1.5px dashed #cdd1da;
  border-radius:10px;
  min-height:260px;
  display:flex;
  flex-direction:column;
  align-items:center;
  justify-content:center;
  gap:12px;
  cursor:pointer;
  transition:all .25s;
  background:#f5f6f8;
  position:relative;
  overflow:hidden;
  user-select:none;
}
#cx .r-upload:hover{border-color:#0e9a8c;background:#f0fdfb;box-shadow:0 0 0 4px rgba(14,154,140,.05)}
#cx .r-upload.dragover{border-color:#0e9a8c;background:#f0fdfb;box-shadow:0 0 0 4px rgba(14,154,140,.08)}
#cx .r-upload.has-image{border:none;background:#000;padding:0;min-height:260px}
#cx .r-upload.has-image img{width:100%;height:260px;object-fit:contain;display:block}

#cx #file-input{
  position:absolute;
  inset:0;
  width:100%;
  height:100%;
  opacity:0;
  cursor:pointer;
  z-index:10;
}
#cx #zone-placeholder{pointer-events:none}
#cx #zone-preview{pointer-events:none}

#cx .r-up-icon{width:48px;height:48px;border-radius:10px;background:#fff;border:1px solid #e2e5eb;display:flex;align-items:center;justify-content:center}
#cx .r-up-main{font-size:13px;font-weight:500;color:#454c5c;text-align:center}
#cx .r-up-sub{font-size:10px;color:#8a93a6;font-family:'JetBrains Mono',monospace;text-align:center}

/* generate button */
#cx .r-genbtn{width:100%;margin-top:12px;padding:12px 20px;background:#111318;border:none;border-radius:9px;color:#fff;font-family:'Outfit',sans-serif;font-size:13px;font-weight:600;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:8px;transition:all .2s;flex-shrink:0}
#cx .r-genbtn:hover:not(:disabled){background:#2d3340;transform:translateY(-1px);box-shadow:0 4px 16px rgba(0,0,0,.15)}
#cx .r-genbtn:disabled{opacity:.4;cursor:not-allowed;transform:none;box-shadow:none}
#cx .r-spinner{width:13px;height:13px;border:2px solid rgba(255,255,255,.25);border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite;flex-shrink:0}
@keyframes spin{to{transform:rotate(360deg)}}

/* status bar */
#cx .r-status{padding:9px 16px;background:#fafbfc;border-top:1px solid #e2e5eb;display:flex;align-items:center;justify-content:space-between;font-family:'JetBrains Mono',monospace;font-size:9px;color:#8a93a6;flex-shrink:0}
#cx .r-status-left{display:flex;align-items:center;gap:6px}
#cx .r-sdot{width:5px;height:5px;border-radius:50%;background:#16a34a;flex-shrink:0}
#cx .r-sdot.loading{background:#d97706;animation:blink 1s ease-in-out infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}

/* report output */
#cx .r-report-wrap{flex:1;border-radius:8px;border:1px solid #e2e5eb;background:#f5f6f8;position:relative;overflow:hidden;min-height:260px;display:flex;flex-direction:column}
#cx .r-empty{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px}
#cx .r-empty-icon{font-size:30px;opacity:.12}
#cx .r-empty-text{font-size:11px;color:#8a93a6;letter-spacing:.5px}
#cx .r-wave{position:absolute;bottom:0;left:0;right:0;height:36px;display:flex;align-items:flex-end;justify-content:center;gap:2px;padding:0 16px 8px;opacity:.12}
#cx .r-wave-bar{width:2px;background:#0e9a8c;border-radius:1px;animation:wv 1.5s ease-in-out infinite}
#cx .r-wave-bar:nth-child(2n){animation-delay:.2s}
#cx .r-wave-bar:nth-child(3n){animation-delay:.4s}
@keyframes wv{0%,100%{height:3px}50%{height:18px}}

#cx .r-report-content{display:none;width:100%;flex:1;min-height:260px;padding:16px;font-family:'JetBrains Mono',monospace;font-size:12px;line-height:2;color:#454c5c;background:transparent;border:none;outline:none;resize:none;white-space:pre-wrap;word-break:break-word}

#cx .r-scan{position:absolute;inset:0;display:none;flex-direction:column;align-items:center;justify-content:center;gap:16px;background:#f5f6f8;z-index:10}
#cx .r-scan.show{display:flex}
#cx .r-rings{width:52px;height:52px;position:relative}
#cx .r-ring1{position:absolute;inset:0;border:2px solid #e2e5eb;border-top-color:#0e9a8c;border-radius:50%;animation:spin 1.2s linear infinite}
#cx .r-ring2{position:absolute;inset:8px;border:1.5px solid #e2e5eb;border-top-color:#1a56db;border-radius:50%;animation:spin .8s linear infinite reverse}
#cx .r-scan-text{font-family:'JetBrains Mono',monospace;font-size:10px;color:#8a93a6;letter-spacing:2px;text-transform:uppercase}
#cx .r-steps{display:flex;flex-direction:column;gap:7px;width:170px}
#cx .r-rstep{display:flex;align-items:center;gap:8px;font-family:'JetBrains Mono',monospace;font-size:9px;color:#8a93a6;opacity:.35;transition:all .4s}
#cx .r-rstep.active{opacity:1;color:#0e9a8c}
#cx .r-rstep.done{opacity:.65;color:#16a34a}
#cx .r-rstep-dot{width:4px;height:4px;border-radius:50%;background:currentColor;flex-shrink:0}

/* download bar */
#cx .r-dlbar{display:none;flex-direction:column;gap:8px;margin-top:12px;flex-shrink:0}
#cx .r-dlbar.show{display:flex}
#cx .r-dlbtn{width:100%;padding:12px 20px;background:#111318;border:none;border-radius:9px;color:#fff;font-family:'Outfit',sans-serif;font-size:13px;font-weight:600;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:8px;transition:background .2s}
#cx .r-dlbtn:hover{background:#2d3340}
#cx .r-dlsub{display:flex;gap:8px}
#cx .r-dlsub-btn{flex:1;padding:8px;background:#fff;border:1px solid #e2e5eb;border-radius:9px;color:#454c5c;font-family:'Outfit',sans-serif;font-size:12px;font-weight:600;cursor:pointer;transition:all .2s}
#cx .r-dlsub-btn:hover{border-color:#0e9a8c;color:#0e9a8c}

/* warning */
#cx .r-warning{background:#fffbeb;border:1px solid #fde68a;border-radius:10px;padding:12px 16px;display:flex;gap:10px;font-size:12px;line-height:1.6;color:#78350f;margin:14px 0}

/* footer */
#cx footer{border-top:1px solid #e2e5eb;padding:24px 0;margin-top:4px}
#cx .r-foot-inner{display:flex;align-items:center;justify-content:space-between}
#cx .r-foot-left{font-size:12px;color:#8a93a6}
#cx .r-foot-left a{color:#0e9a8c;text-decoration:none}
#cx .r-foot-right{display:flex;gap:8px}
#cx .r-foot-chip{font-family:'JetBrains Mono',monospace;font-size:10px;color:#8a93a6;background:#fff;border:1px solid #e2e5eb;padding:4px 10px;border-radius:4px}
</style>

<div id="cx">
  <div class="r-topbar"></div>

  <nav class="r-nav">
    <div class="r-nav-inner">
      <div class="r-logo">
        <svg width="34" height="34" viewBox="0 0 36 36" fill="none">
          <rect width="36" height="36" rx="9" fill="#0f172a"/>
          <path d="M10 22c0-4 2-7 5-8.5C16 12.5 17 12 18 12s2 .5 3 1.5c3 1.5 5 4.5 5 8.5" stroke="#14b8a6" stroke-width="1.5" stroke-linecap="round"/>
          <path d="M7 19h4l2-4 3 8 2-6 2 4h3l2-3h4" stroke="white" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round" opacity=".9"/>
          <circle cx="18" cy="22" r="2.5" fill="none" stroke="#14b8a6" stroke-width="1.2"/>
        </svg>
        <div>
          <div class="r-logo-name">CheXReport</div>
          <div class="r-logo-sub">Medical AI &middot; Flowgenix</div>
        </div>
      </div>
      <div class="r-pill"><div class="r-live"></div>Model Active &middot; Epoch 9 &middot; Val 0.675</div>
      <div class="r-nav-right">
        <a href="https://www.linkedin.com/in/flowgenix-ai-b51517278" target="_blank" class="r-nav-link">Muhammed Panchla</a>
        <button class="r-nav-cta">Research Preview</button>
      </div>
    </div>
  </nav>

  <div class="r-wrap">
    <section class="r-hero">
      <div>
        <div class="r-eyebrow"><div class="r-eyebrow-dash"></div>Multimodal Medical Intelligence</div>
        <h1 class="r-h1">From X-Ray<br/>to Radiology<br/><em>Report.</em></h1>
        <p class="r-desc">Upload a chest X-ray and CheXReport generates a structured radiology findings report. Built on DenseNet121 and BioGPT &mdash; trained on 15 million biomedical abstracts from PubMed.</p>
        <div class="r-stats">
          <div><div class="r-stat-val">6,687</div><div class="r-stat-lbl">Training Scans</div></div>
          <div class="r-hsep"></div>
          <div><div class="r-stat-val">72M</div><div class="r-stat-lbl">Parameters</div></div>
          <div class="r-hsep"></div>
          <div><div class="r-stat-val">0.675</div><div class="r-stat-lbl">Val Loss</div></div>
          <div class="r-hsep"></div>
          <div><div class="r-stat-val">10 ep</div><div class="r-stat-lbl">Training</div></div>
        </div>
      </div>
      <div class="r-xcard">
        <div class="r-xcard-hd">
          <span class="r-xcard-title">PA Chest View &middot; Live Analysis</span>
          <div class="r-xdots"><span></span><span></span><span></span></div>
        </div>
        <div class="r-xvp">
          <div class="r-xgrid"></div>
          <div class="r-xray-body">
            <div class="r-xr-spine"></div><div class="r-xr-heart"></div>
            <div class="r-xr-rib"></div><div class="r-xr-rib"></div><div class="r-xr-rib"></div>
            <div class="r-xr-rib"></div><div class="r-xr-rib"></div><div class="r-xr-rib"></div>
          </div>
          <div class="r-scanline"></div>
          <div class="r-xcorner tl"></div><div class="r-xcorner tr"></div>
          <div class="r-xcorner bl"></div><div class="r-xcorner br"></div>
          <div class="r-xpip">RES&#160;&#160;512x512<br/>MODE&#160;DX/PA<br/>ENC&#160;&#160;DenseNet<br/>STATUS&#160;<span style="color:#4ade80">READY</span></div>
        </div>
        <div class="r-xcard-ft">
          <span class="r-xcard-ft-text">BioGPT &middot; Beam x4 &middot; IU X-Ray</span>
          <span class="r-xbadge">AI Ready</span>
        </div>
      </div>
    </section>
  </div>

  <div class="r-arch">
    <div class="r-arch-inner">
      <div class="r-step"><div class="r-step-num">01</div><div><div class="r-step-name">X-Ray Input</div><div class="r-step-sub">PNG / JPG</div></div></div>
      <div class="r-step"><div class="r-step-num">02</div><div><div class="r-step-name">DenseNet121</div><div class="r-step-sub">1024-dim</div></div></div>
      <div class="r-step"><div class="r-step-num">03</div><div><div class="r-step-name">Projection</div><div class="r-step-sub">LN-Linear-GELU</div></div></div>
      <div class="r-step"><div class="r-step-num">04</div><div><div class="r-step-name">BioGPT</div><div class="r-step-sub">Beam x4</div></div></div>
      <div class="r-step"><div class="r-step-num">05</div><div><div class="r-step-name">Report</div><div class="r-step-sub">Clinical text</div></div></div>
    </div>
  </div>

  <div class="r-wrap">
    <div class="r-section">

      <div class="r-sec-head">
        <div>
          <div class="r-sec-eye">Live Inference</div>
          <div class="r-sec-title">Upload &amp; Generate</div>
        </div>
        <div class="r-sec-right">DenseNet121 &middot; BioGPT &middot; PyTorch<br/>IU X-Ray &middot; CPU</div>
      </div>

      <div class="r-statrow">
        <div class="r-sc"><div class="r-sc-val">DenseNet</div><div class="r-sc-lbl">Vision Encoder</div></div>
        <div class="r-sc"><div class="r-sc-val">BioGPT</div><div class="r-sc-lbl">Language Model</div></div>
        <div class="r-sc"><div class="r-sc-val">IU X-Ray</div><div class="r-sc-lbl">Dataset</div></div>
        <div class="r-sc"><div class="r-sc-val">Beam x4</div><div class="r-sc-lbl">Decoding</div></div>
        <div class="r-sc"><div class="r-sc-val">0.675</div><div class="r-sc-lbl">Best Val Loss</div></div>
      </div>

      <div class="r-grid">
        <!-- LEFT: INPUT PANEL -->
        <div class="r-panel">
          <div class="r-panel-hd">
            <div class="r-panel-left">
              <div class="r-pdot" id="inp-dot"></div>
              <span class="r-panel-label">Input &mdash; Chest X-Ray</span>
            </div>
            <span class="r-panel-chip">224&times;224 &middot; Grayscale</span>
          </div>
          <div class="r-panel-body">
            <div class="r-upload" id="upload-zone">
              <input
                type="file"
                id="file-input"
                accept="image/png,image/jpeg,image/jpg"
              />
              <div id="zone-placeholder" style="display:flex;flex-direction:column;align-items:center;gap:12px;">
                <div class="r-up-icon">
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#8a93a6" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                    <polyline points="17 8 12 3 7 8"/>
                    <line x1="12" y1="3" x2="12" y2="15"/>
                  </svg>
                </div>
                <div>
                  <div class="r-up-main">Click or drop chest X-ray here</div>
                  <div class="r-up-sub">PNG &middot; JPG &middot; JPEG</div>
                </div>
              </div>
              <img
                id="zone-preview"
                src=""
                alt="X-Ray preview"
                style="display:none;width:100%;height:260px;object-fit:contain;border-radius:8px;"
              />
            </div><!-- /upload-zone -->

            <button class="r-genbtn" id="gen-btn" disabled>
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><polygon points="13,2 3,14 12,14 11,22 21,10 12,10"/></svg>
              Generate Radiology Report
            </button>
          </div>
          <div class="r-status">
            <div class="r-status-left"><div class="r-sdot" id="inp-sdot"></div><span id="inp-stxt">Model ready &middot; val_loss 0.6757 &middot; epoch 9</span></div>
            <span>PyTorch &middot; CPU</span>
          </div>
        </div>

        <!-- RIGHT: OUTPUT PANEL -->
        <div class="r-panel">
          <div class="r-panel-hd">
            <div class="r-panel-left">
              <div class="r-pdot dim" id="out-dot"></div>
              <span class="r-panel-label">Generated Radiology Report</span>
            </div>
            <span class="r-panel-chip" id="out-chip">awaiting scan</span>
          </div>
          <div class="r-panel-body">
            <div class="r-report-wrap">
              <div class="r-empty" id="empty-state">
                <div class="r-empty-icon">&#x1FAC1;</div>
                <div class="r-empty-text">Report will appear here</div>
                <div class="r-wave">
                  <div class="r-wave-bar" style="height:3px"></div>
                  <div class="r-wave-bar" style="height:6px"></div>
                  <div class="r-wave-bar" style="height:9px"></div>
                  <div class="r-wave-bar" style="height:6px"></div>
                  <div class="r-wave-bar" style="height:4px"></div>
                  <div class="r-wave-bar" style="height:8px"></div>
                  <div class="r-wave-bar" style="height:5px"></div>
                  <div class="r-wave-bar" style="height:10px"></div>
                  <div class="r-wave-bar" style="height:7px"></div>
                  <div class="r-wave-bar" style="height:4px"></div>
                </div>
              </div>
              <div class="r-scan" id="scan-state">
                <div class="r-rings"><div class="r-ring1"></div><div class="r-ring2"></div></div>
                <div class="r-scan-text">Analysing scan</div>
                <div class="r-steps">
                  <div class="r-rstep" id="rs1"><div class="r-rstep-dot"></div>Vision encoding</div>
                  <div class="r-rstep" id="rs2"><div class="r-rstep-dot"></div>Projection layer</div>
                  <div class="r-rstep" id="rs3"><div class="r-rstep-dot"></div>BioGPT decoding</div>
                </div>
              </div>
              <textarea class="r-report-content" id="report-content" readonly></textarea>
            </div>
            <div class="r-dlbar" id="dl-bar">
              <button class="r-dlbtn" id="dl-txt">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                Download TXT
              </button>
              <div class="r-dlsub">
                <button class="r-dlsub-btn" id="copy-report">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                  Copy Report
                </button>
                <button class="r-dlsub-btn" id="copy-json">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                  Copy JSON
                </button>
              </div>
            </div>
          </div>
          <div class="r-status">
            <div class="r-status-left">
              <div class="r-sdot" style="background:#cdd1da" id="out-sdot"></div>
              <span id="out-stxt">Waiting for upload...</span>
            </div>
            <span id="out-wc">&mdash;</span>
          </div>
        </div>
      </div>

      <div class="r-warning">
        <span>&#x26A0;&#xFE0F;</span>
        <div><strong>Research use only.</strong> Not a certified medical device. All outputs must be reviewed by a qualified radiologist before any clinical decision is made.</div>
      </div>

      <footer>
        <div class="r-foot-inner">
          <div class="r-foot-left">Built by <strong>Muhammed Panchla</strong> &middot; <a href="https://www.linkedin.com/in/flowgenix-ai-b51517278" target="_blank">Flowgenix AI</a> &middot; DenseNet121 &middot; BioGPT &middot; IU X-Ray &middot; PyTorch</div>
          <div class="r-foot-right"><span class="r-foot-chip">v1.0</span><span class="r-foot-chip">2026</span></div>
        </div>
      </footer>

    </div>
  </div>
</div>
"""

# ══════════════════════════════════════════════════════════════════════════════
#  JAVASCRIPT  (FIXED — see FIX-1, FIX-2, FIX-3 inline)
# ══════════════════════════════════════════════════════════════════════════════

JS = """
(function() {
  function q(id) { return element.querySelector('#' + id); }

  var dataUrl   = null;
  var patientID = 'PT-' + Math.random().toString(36).substr(2,8).toUpperCase();
  var studyDate = new Date().toLocaleDateString('en-GB', {day:'2-digit', month:'short', year:'numeric'});

  function init() {
    var zone = q('upload-zone');
    var inp  = q('file-input');
    var btn  = q('gen-btn');
    if (!zone || !inp || !btn) { setTimeout(init, 80); return; }

    var placeholder = q('zone-placeholder');
    var preview     = q('zone-preview');

    // ── File loading ─────────────────────────────────────────────────────
    function loadFile(file) {
      /*
       * FIX-2: The original guard was:
       *   if (!file || !file.type.startsWith('image/'))
       *
       * file.type is an empty string '' in many environments:
       *   • macOS Finder / Safari drag-and-drop
       *   • Windows file picker in Chrome/Edge
       *   • HuggingFace Spaces iframe context on some browsers
       *   • Mobile browsers (iOS Safari, Android Chrome)
       *
       * ''.startsWith('image/') === false, so every upload attempt was
       * silently rejected, leaving the zone empty and the Generate button
       * disabled.
       *
       * Fix: Accept the file if EITHER the MIME type starts with 'image/'
       * OR the filename extension is .png / .jpg / .jpeg.
       */
      var validByMime = file && file.type.startsWith('image/');
      var validByExt  = file && /\\.(png|jpg|jpeg)$/i.test(file.name);
      if (!file || (!validByMime && !validByExt)) {
        q('inp-stxt').textContent = 'Invalid file \u00B7 please choose a PNG or JPG';
        return;
      }

      var reader = new FileReader();

      reader.onload = function(e) {
        /*
         * FIX-1: inp.value='' was being called synchronously in the
         * 'change' handler BEFORE this async callback fires.
         *
         * On Safari / WebKit (including HuggingFace Spaces via Safari on
         * macOS/iOS), resetting the input value before the FileReader
         * completes can release the underlying File object reference,
         * causing reader.result to be null or the load event to never fire.
         * The net effect: the image never appears, the Generate button
         * stays disabled, and the status bar never updates.
         *
         * Fix: Reset inp.value HERE — inside onload — so it only runs
         * AFTER the base-64 data is safely captured in dataUrl.
         * This also preserves the intended behaviour of allowing the same
         * file to be re-selected after a successful read.
         */
        inp.value = '';   // ← FIX-1: moved from change handler to here

        dataUrl = e.target.result;
        preview.src = dataUrl;
        preview.style.display = 'block';
        placeholder.style.display = 'none';
        zone.classList.add('has-image');
        btn.disabled = false;
        q('inp-stxt').textContent = 'Image loaded \u00B7 ready to generate';
        q('out-stxt').textContent = 'Image received \u00B7 click Generate to run inference';
      };

      reader.onerror = function() {
        q('inp-stxt').textContent = 'Error reading file \u00B7 try again';
      };

      reader.readAsDataURL(file);
    }

    // ── File input change ─────────────────────────────────────────────────
    inp.addEventListener('change', function(e) {
      var f = e.target.files && e.target.files[0];
      if (f) loadFile(f);
      /*
       * FIX-1 continued: inp.value='' has been removed from here and
       * moved into reader.onload above so it runs only after the file
       * has been fully read.
       */
    });

    // ── Drag and drop ──────────────────────────────────────────────────────
    zone.addEventListener('dragover', function(e) {
      e.preventDefault();
      e.stopPropagation();
      zone.classList.add('dragover');
    });
    zone.addEventListener('dragleave', function(e) {
      if (!zone.contains(e.relatedTarget)) zone.classList.remove('dragover');
    });
    zone.addEventListener('drop', function(e) {
      e.preventDefault();
      e.stopPropagation();
      zone.classList.remove('dragover');
      var f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (f) loadFile(f);
    });

    // ── Step animation ─────────────────────────────────────────────────────
    function stepAnim() {
      var steps = ['rs1','rs2','rs3'], i = 0;
      var iv = setInterval(function() {
        if (i > 0) q(steps[i-1]).className = 'r-rstep done';
        if (i < steps.length) { q(steps[i]).className = 'r-rstep active'; i++; }
        else clearInterval(iv);
      }, 1300);
    }

    // ── Generate ───────────────────────────────────────────────────────────
    btn.addEventListener('click', async function() {
      if (!dataUrl) return;
      btn.disabled = true;
      btn.innerHTML = '<div class="r-spinner"></div> Analysing...';
      q('inp-sdot').className = 'r-sdot loading';
      q('inp-stxt').textContent = 'Running inference...';
      q('empty-state').style.display = 'none';
      q('scan-state').classList.add('show');
      q('report-content').style.display = 'none';
      q('out-chip').textContent = 'generating...';
      q('out-stxt').textContent = 'Vision encoder \u2192 projection \u2192 BioGPT...';
      q('out-sdot').style.background = '#d97706';
      q('dl-bar').classList.remove('show');
      ['rs1','rs2','rs3'].forEach(function(s) { q(s).className = 'r-rstep'; });
      stepAnim();

      try {
        var report = await server.run(dataUrl);
        q('scan-state').classList.remove('show');
        q('report-content').style.display = 'block';
        q('report-content').style.color = '#454c5c';
        q('out-dot').className = 'r-pdot green';
        q('out-sdot').style.background = '#16a34a';
        // Typewriter effect
        var words = report.split(' '), idx = 0;
        q('report-content').value = '';
        function tw() {
          if (idx < words.length) {
            q('report-content').value += (idx === 0 ? '' : ' ') + words[idx++];
            setTimeout(tw, 22);
          } else {
            q('out-chip').textContent = 'complete';
            q('out-stxt').textContent = 'Done \u00B7 ' + words.length + ' words';
            q('out-wc').textContent = words.length + ' words';
            q('dl-bar').classList.add('show');
          }
        }
        tw();
      } catch(err) {
        /*
         * FIX-3: The original code used err.message which is undefined
         * for non-Error throw values (Gradio timeout objects, plain
         * strings, fetch AbortErrors).  This caused the catch block to
         * display "Error: undefined", hiding the real failure reason.
         *
         * Fix: Use err.message || String(err) so any thrown value is
         * shown as a readable string.
         */
        var msg = (err && err.message) ? err.message : String(err);
        q('scan-state').classList.remove('show');
        q('report-content').style.display = 'block';
        q('report-content').style.color = '#dc2626';
        q('report-content').value = 'Error: ' + msg + '\\n\\nPlease try again.';
        q('out-chip').textContent = 'error';
        q('out-stxt').textContent = 'Failed';
        q('out-sdot').style.background = '#dc2626';
      }

      btn.disabled = false;
      btn.innerHTML = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><polygon points="13,2 3,14 12,14 11,22 21,10 12,10"/></svg> Generate Radiology Report';
      q('inp-sdot').className = 'r-sdot';
      q('inp-stxt').textContent = 'Model ready \u00B7 val_loss 0.6757 \u00B7 epoch 9';
    });

    // ── Downloads & clipboard ──────────────────────────────────────────────
    function getReport() { return (q('report-content') || {}).value || ''; }

    function parseSections(r) {
      var sec = {INDICATION:'', COMPARISON:'', FINDINGS:'', IMPRESSION:''}, cur = '';
      r.split('\\n').forEach(function(l) {
        var t = l.trim();
        if (sec.hasOwnProperty(t)) { cur = t; }
        else if (cur && t && !t.startsWith('\u2500') && !t.startsWith('AI') && !t.startsWith('MODALITY') && !t.startsWith('DATE')) {
          sec[cur] += (sec[cur] ? ' ' : '') + t;
        }
      });
      return sec;
    }

    function flashBtn(el, msg) {
      var orig = el.innerHTML;
      el.innerHTML = msg;
      el.style.borderColor = '#16a34a';
      el.style.color = '#16a34a';
      setTimeout(function() {
        el.innerHTML = orig;
        el.style.borderColor = '';
        el.style.color = '';
      }, 1800);
    }

    // Download TXT
    var dlTxt = q('dl-txt');
    if (dlTxt) dlTxt.addEventListener('click', function() {
      var r = getReport(); if (!r) return;
      var content =
        'CHEXREPORT AI \u2014 RADIOLOGY REPORT\\n' +
        '\u2500'.repeat(48) + '\\n' +
        'Patient ID : ' + patientID + '\\n' +
        'Study Date : ' + studyDate + '\\n' +
        'Modality   : Chest X-Ray (PA/AP)\\n' +
        'Model      : DenseNet121 + BioGPT v1.0\\n' +
        '\u2500'.repeat(48) + '\\n\\n' +
        r +
        '\\n\\nAI DISCLAIMER: Research use only. Not a certified medical device.';
      var blob = new Blob([content], {type:'text/plain;charset=utf-8'});
      var url  = URL.createObjectURL(blob);
      var a    = document.createElement('a');
      a.href     = url;
      a.download = 'chexreport_' + studyDate.replace(/ /g,'_') + '.txt';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(function(){ URL.revokeObjectURL(url); }, 1000);
    });

    // Copy Report (plain text)
    var copyReport = q('copy-report');
    if (copyReport) copyReport.addEventListener('click', function() {
      var r = getReport(); if (!r) return;
      var copied = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><polyline points="20 6 9 17 4 12"/></svg> Copied!';
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(r)
          .then(function(){ flashBtn(copyReport, copied); })
          .catch(function(){ fallbackCopy(r, copyReport, copied); });
      } else {
        fallbackCopy(r, copyReport, copied);
      }
    });

    // Copy JSON (structured)
    var copyJson = q('copy-json');
    if (copyJson) copyJson.addEventListener('click', function() {
      var r = getReport(); if (!r) return;
      var sec  = parseSections(r);
      var json = JSON.stringify({
        generated_by : 'CheXReport AI v1.0',
        model        : 'DenseNet121+BioGPT',
        patient_id   : patientID,
        study_date   : studyDate,
        modality     : 'Chest X-Ray (PA/AP)',
        indication   : sec.INDICATION,
        comparison   : sec.COMPARISON,
        findings     : sec.FINDINGS,
        impression   : sec.IMPRESSION,
        disclaimer   : 'Research use only. Not a certified medical device.'
      }, null, 2);
      var copied = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><polyline points="20 6 9 17 4 12"/></svg> Copied!';
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(json)
          .then(function(){ flashBtn(copyJson, copied); })
          .catch(function(){ fallbackCopy(json, copyJson, copied); });
      } else {
        fallbackCopy(json, copyJson, copied);
      }
    });

    function fallbackCopy(text, btn, successMsg) {
      var ta = document.createElement('textarea');
      ta.value = text;
      ta.style.cssText = 'position:fixed;top:0;left:0;opacity:0;pointer-events:none;';
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      try {
        document.execCommand('copy');
        flashBtn(btn, successMsg);
      } catch(e) {
        flashBtn(btn, 'Copy failed');
      }
      document.body.removeChild(ta);
    }

  } // end init()

  init();
})();
"""

# ── Gradio app ─────────────────────────────────────────────────────────────────
# FIX-4 REMINDER: requirements.txt must include  gradio>=5.0
# server_functions is a Gradio 5.x feature. Without a version pin, HuggingFace
# Spaces may install an older Gradio that silently ignores this parameter,
# leaving server.run() undefined in JS and crashing the Generate step.
# Add the following line to requirements.txt:
#   gradio>=5.0

RESET_CSS = """
body, .gradio-container, .gradio-container > .main {
    background: #f5f6f8 !important;
    margin: 0 !important;
    padding: 0 !important;
    max-width: 100% !important;
    min-height: unset !important;
}
footer.svelte-1rjryqp, footer { display: none !important; }
"""

with gr.Blocks(title="CheXReport AI — Flowgenix", css=RESET_CSS) as demo:
    gr.HTML(
        value             = HTML,
        js_on_load        = JS,
        server_functions  = [run],
        apply_default_css = False,
        sanitize_html     = False,
    )

if __name__ == "__main__":
    demo.launch()
