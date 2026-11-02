import os
from PIL import Image
import streamlit as st
import timm
import torch
import torch.nn as nn
from torchvision import transforms

# ===== Model Configuration =====
ARCH = 'convnext_small.fb_in22k'
CLS = ['ai_edited', 'fully_ai', 'real']
SZ = 256
MODEL_PATH = 'best_model.pth'


# ===== Architecture =====
class FreqBlock(nn.Module):

  def __init__(self, out_c=256):
    super().__init__()
    self.srm = nn.Conv2d(3, 12, 3, padding=1, bias=False)
    nn.init.kaiming_normal_(self.srm.weight)

    self.net = nn.Sequential(
        nn.Conv2d(18, 64, 3, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.GELU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1),
        nn.BatchNorm2d(128),
        nn.GELU(),
        nn.Conv2d(128, out_c, 3, stride=2, padding=1),
        nn.BatchNorm2d(out_c),
        nn.GELU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )

  def forward(self, x):
    xf = x.float()
    ft = torch.fft.fftshift(torch.fft.fft2(xf, norm='ortho'))
    amp = torch.log(torch.abs(ft) + 1e-6)
    ph = torch.angle(ft)
    amp = (amp - amp.mean(dim=(-2, -1), keepdim=True)) / (
        amp.std(dim=(-2, -1), keepdim=True) + 1e-6
    )
    ph = ph / 3.141592653589793

    s = self.srm(x)
    cat = torch.cat([s, amp.to(x.dtype), ph.to(x.dtype)], dim=1)
    return self.net(cat)


class HybridModel(nn.Module):

  def __init__(self, arch=ARCH, n_cls=3, pre=False):
    super().__init__()
    self.base = timm.create_model(arch, pretrained=pre, num_classes=0)
    dim1 = self.base.num_features
    dim2 = 256
    self.freq = FreqBlock(dim2)
    tot = dim1 + dim2

    self.gate = nn.Sequential(
        nn.Linear(tot, tot // 4),
        nn.GELU(),
        nn.Linear(tot // 4, tot),
        nn.Sigmoid(),
    )
    self.head = nn.Sequential(
        nn.LayerNorm(tot),
        nn.Dropout(0.3),
        nn.Linear(tot, 512),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(512, n_cls),
    )

  def forward(self, x):
    f1 = self.base(x)
    f2 = self.freq(x)
    f = torch.cat([f1, f2], dim=1)
    f = f * self.gate(f)
    return self.head(f)


# ===== Load Model =====
@st.cache_resource
def load_trained_model():
  dev = torch.device('cpu')
  model = HybridModel(ARCH, n_cls=len(CLS), pre=False)

  if os.path.exists(MODEL_PATH):
    weights = torch.load(MODEL_PATH, map_location=dev)

    if isinstance(weights, dict):
      if 'model_state_dict' in weights:
        weights = weights['model_state_dict']
      elif 'state_dict' in weights:
        weights = weights['state_dict']
      elif 'model' in weights:
        weights = weights['model']
      elif 'net' in weights:
        weights = weights['net']

    cleaned = {str(k).replace('module.', ''): v for k, v in weights.items()}
    model.load_state_dict(cleaned, strict=False)

  model.to(dev)
  model.eval()
  return model


# ===== Preprocessing =====
val_tf = transforms.Compose([
    transforms.Resize((SZ, SZ)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ===== Streamlit UI =====
st.set_page_config(page_title='AI Photo Detector', page_icon='🔍')
st.title('🔍 AI vs Real Picture Detector')
st.write(
    'Upload a picture to check whether it is **Real**, **AI Edited**, or'
    ' **Fully AI Generated**.'
)

uploaded_file = st.file_uploader(
    'Choose an image...', type=['jpg', 'jpeg', 'png', 'webp']
)

if uploaded_file is not None:
  img = Image.open(uploaded_file).convert('RGB')
  st.image(img, caption='Uploaded Image', use_container_width=True)

  with st.spinner('Analyzing image...'):
    model = load_trained_model()
    inp = val_tf(img).unsqueeze(0)
    with torch.no_grad():
      out = model(inp)
      probs = torch.softmax(out, dim=1).squeeze(0)

    preds = {CLS[i]: float(probs[i]) for i in range(len(CLS))}
    best_cls = max(preds, key=preds.get)

  st.subheader(f'Prediction: **{best_cls.upper()}** ({preds[best_cls]*100:.2f}%)')

  st.write('Confidence Breakdown:')
  for c, p in preds.items():
    st.write(f'- **{c}**: {p*100:.2f}%')
    st.progress(p)
