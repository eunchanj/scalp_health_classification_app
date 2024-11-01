import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import sys

# 페이지 설정 (Streamlit 명령어 중 첫 번째 위치)
st.set_page_config(page_title="두피 건강 모니터링", page_icon=":bar_chart:", layout="centered")

# 실행 파일의 위치를 기준으로 경로 설정
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
else:
    application_path = os.path.dirname(__file__)

MODEL_PATH = os.path.join(application_path, 'model')  # 모델 디렉토리 경로

# 모델 로드 함수
@st.cache_resource  # Streamlit의 캐싱 기능을 사용하여 로드를 최적화
def load_model():
    try:
        # SavedModel 포맷의 모델을 불러옵니다.
        model = tf.saved_model.load(MODEL_PATH)
        inference_func = model.signatures["serving_default"]  # 기본 서명으로 모델 함수 설정
        st.write("모델이 성공적으로 로드되었습니다.")  # 디버깅 정보 출력
        return inference_func
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {e}")

# 모델 불러오기
model = load_model()

# 제목
st.title("두피 건강 모니터링")

# 설명
st.write("두피 이미지를 업로드하여 건강 상태를 분석합니다.")

# 파일 업로드
uploaded_file = st.file_uploader("두피 이미지 선택", type=["jpg", "jpeg", "png", "bmp", "gif"])

# 이미지 업로드 확인
if uploaded_file is not None:
    # 이미지 열기 및 표시
    image = Image.open(uploaded_file)
    st.image(image, caption="업로드된 이미지", use_column_width=True)

    # 이미지 전처리
    def preprocess_image(image):
        image = image.resize((224, 224))  # MobileNet 입력 크기
        image = np.array(image) / 255.0  # 정규화
        image = image.astype(np.float32)  # float32로 변환
        image = np.expand_dims(image, axis=0)  # 배치 차원 추가
        return image

    img_tensor = preprocess_image(image)

    # 디버깅용으로 이미지 텐서 출력
    st.write("이미지 텐서:", img_tensor.shape, img_tensor.dtype)

    # 예측 버튼
    if st.button("분류 시작"):
        with st.spinner("분류 중..."):
            try:
                # 예측 수행
                predictions = model(tf.constant(img_tensor))  # 모델에 입력
                
                # 모델 출력 디버깅
                st.write("모델 예측 결과:", predictions)

                # 출력 레이어 이름 찾기 (모델 출력 구조 확인)
                output_name = list(predictions.keys())  # 첫 번째 출력 이름 사용
                print(output_name)
                st.write("출력 레이어 이름:", output_name)

                # 확률 값 추출
                probability = 1 - predictions[output_name][0].numpy().item()  # 첫 번째 예측 확률 추출
                st.write("추출된 확률:", probability)

                # 분류 기준 설정
                if probability >= 0.6:
                    prediction = "위험해요"
                    color = "red"
                else:
                    prediction = "안전해요"
                    color = "green"

                # 결과 표시
                st.markdown(f"**결과:** <span style='color:{color}'>{prediction}</span>", unsafe_allow_html=True)
                st.write(f"**확률:** {probability:.2f}")

            except Exception as e:
                st.error(f"분류 중 오류 발생: {e}")
