import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model.mnist_cnn import CNN

# ===== 1. 모델 로드 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load("saved_model.pth", map_location=device))
model.eval()

# ===== 2. 이미지 업로드 =====
st.title("MNIST 손글씨 숫자 예측")
uploaded_file = st.file_uploader("손글씨 이미지를 업로드하세요 (28x28 흑백 권장)", type=["png","jpg","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")  # 흑백 변환
    st.image(image, caption='업로드된 이미지', use_column_width=True)

    # ===== 3. 전처리 =====
    transform = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    # ===== 4. 예측 =====
    with torch.no_grad():
        output = model(input_tensor)
        pred = output.argmax(dim=1, keepdim=True).item()
    
    st.success(f"예측 숫자: {pred}")
