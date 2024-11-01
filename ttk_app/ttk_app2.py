import torch
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import threading
import sys
import os

# 실행 파일의 위치를 기준으로 경로 설정
if getattr(sys, 'frozen', False):
    # 실행 파일로 패키징된 경우
    application_path = os.path.dirname(sys.executable)
else:
    # 스크립트로 실행되는 경우
    application_path = os.path.dirname(__file__)

MODEL_PATH = os.path.join(application_path, 'model.pt')  # 모델 파일의 경로

# 모델 로드
model = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
model.eval()

class ImageClassifierApp:
    def __init__(self, master):
        self.master = master
        master.title("두피 건강 모니터링")
        master.geometry("400x700")  # 창 크기 조정 (높이 증가)
        master.configure(bg='#FFFFFF')

        # 창 크기 조정 불가 (선택 사항)
        master.resizable(False, False)

        # 스타일 설정
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # 기본 버튼 스타일
        self.style.configure('TButton', font=('Helvetica', 12), background='#4285F4', foreground='white', padding=10)
        self.style.map('TButton', background=[('active', '#3367D6')])

        # 레이블 스타일
        self.style.configure('TLabel', background='#FFFFFF', font=('Helvetica', 12))
        self.style.configure('Result.TLabel', font=('Helvetica', 14, 'bold'))

        # Hover 스타일 추가
        self.style.configure('Hover.TButton', background='#3367D6')

        # 메인 프레임 생성 및 배경색 설정
        self.main_frame = tk.Frame(master, bg='#FFFFFF')
        self.main_frame.pack(expand=True, fill='both')

        # 로고 이미지 추가 (옵션)
        logo_path = os.path.join(application_path, 'logo.png')
        if os.path.exists(logo_path):
            logo_image = Image.open(logo_path)
            logo_image = logo_image.resize((142, 202), Image.LANCZOS)
            self.logo_photo = ImageTk.PhotoImage(logo_image)
            self.logo_label = ttk.Label(self.main_frame, image=self.logo_photo, background='#FFFFFF')
            self.logo_label.pack(pady=(20, 10))

        # 위젯 생성
        self.label = ttk.Label(self.main_frame, text="두피 건강 모니터링 프로토타입")
        self.label.pack(pady=10)

        self.classify_button = ttk.Button(self.main_frame, text="두피 이미지 선택", command=self.open_image, style='TButton')
        self.classify_button.pack(pady=5)

        self.image_label = ttk.Label(self.main_frame, background='#FFFFFF')
        self.image_label.pack(pady=10)

        # 결과 표시 프레임
        self.result_frame = tk.Frame(self.main_frame, bg='#FFFFFF')
        self.result_frame.pack(pady=10)

        self.result_label = ttk.Label(self.result_frame, text="", style='Result.TLabel')
        self.result_label.pack()

        # 위험도 표시 프레임
        self.scale_frame = tk.Frame(self.main_frame, bg='#FFFFFF')
        self.scale_frame.pack(pady=10)

        # 위험도 초기화
        self.update_scale_frame()

        # 진행 상황 표시기
        self.progress = ttk.Progressbar(self.main_frame, orient='horizontal', mode='indeterminate')
        self.progress.pack(pady=10)

        # 이미지 경로 초기화
        self.image_path = None

        # 이벤트 바인딩 (마우스 오버 효과)
        self.classify_button.bind("<Enter>", self.on_enter)
        self.classify_button.bind("<Leave>", self.on_leave)

    def on_enter(self, e):
        self.classify_button.config(style='Hover.TButton')

    def on_leave(self, e):
        self.classify_button.config(style='TButton')

    def open_image(self):
        self.image_path = filedialog.askopenfilename(
            title="이미지 선택",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")]
        )
        if self.image_path:
            # 선택한 이미지 표시
            img = Image.open(self.image_path)
            img.thumbnail((200, 200))  # 이미지 크기 조정
            self.photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.photo)

            # GUI 멈춤 방지를 위해 새로운 스레드에서 실행
            threading.Thread(target=self.classify_image).start()

    def classify_image(self):
        try:
            # 진행 표시기 시작
            self.progress.start()
            # 결과 레이블 초기화
            self.result_label.config(text="분류 중...", foreground='black')

            # 위험도 초기화
            self.update_scale_frame()

            # 이미지 전처리
            img = Image.open(self.image_path).convert('RGB')
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
            img_tensor = preprocess(img).unsqueeze(0)

            # 예측 수행
            with torch.no_grad():
                outputs = model(img_tensor)[0][1]
                probability = outputs.item()  # 확률 값 추출

                # 확률 값 제한 (0과 1 사이로 설정)
                probability = min(max(probability, 0), 1)

                # 분류 기준 적용
                if probability <= 0.2:
                    prediction = '매우안전'
                    color = 'blue'
                elif probability <= 0.6:
                    prediction = '안전'
                    color = 'green'
                elif probability <= 0.8:
                    prediction = '위험'
                    color = 'orange'
                else:
                    prediction = '매우위험'
                    color = 'red'

                # 결과 레이블에 예측 결과 표시
                self.result_label.config(
                    text=f"결과: {prediction}",
                    foreground=color
                )

                # 위험도 표시 업데이트
                self.update_scale_frame()

        except Exception as e:
            # 예외 처리
            messagebox.showerror("오류", str(e))
            self.result_label.config(text="")
            self.update_scale_frame(reset=True)
        finally:
            # 진행 표시기 중지
            self.progress.stop()

    def update_scale_frame(self, reset=False):
        # scale_frame 내용 초기화
        for widget in self.scale_frame.winfo_children():
            widget.destroy()

        if reset:
            # 초기 위험도 표시
            scale_text = "위험도: "
            scale_label = ttk.Label(self.scale_frame, text=scale_text, background='#FFFFFF', font=('Helvetica', 12))
            scale_label.pack(side='left')

            grades = [
                ('매우위험', 'red'),
                ('위험', 'orange'),
                ('안전', 'green'),
                ('매우안전', 'blue')
            ]

            for i, (grade, color) in enumerate(grades):
                grade_label = ttk.Label(self.scale_frame, text=grade, foreground=color, background='#FFFFFF', font=('Helvetica', 12))
                grade_label.pack(side='left')
                if i < len(grades) -1:
                    dash_label = ttk.Label(self.scale_frame, text="-", background='#FFFFFF', font=('Helvetica', 12))
                    dash_label.pack(side='left')
            return

        # 예측 결과에 따른 위험도 표시
        prediction = self.result_label.cget("text").replace("결과: ", "")
        prediction_color = self.result_label.cget("foreground")

        scale_text = "위험도: "
        scale_label = ttk.Label(self.scale_frame, text=scale_text, background='#FFFFFF', font=('Helvetica', 12))
        scale_label.pack(side='left')

        grades = [
            ('매우위험', 'red'),
            ('위험', 'orange'),
            ('안전', 'green'),
            ('매우안전', 'blue')
        ]

        for i, (grade, color) in enumerate(grades):
            if grade == prediction:
                # 현재 예측된 등급 강조
                grade_label = ttk.Label(self.scale_frame, text=grade, foreground=color, background='#FFFFFF', font=('Helvetica', 12, 'bold'))
            else:
                grade_label = ttk.Label(self.scale_frame, text=grade, foreground=color, background='#FFFFFF', font=('Helvetica', 12))
            grade_label.pack(side='left')
            if i < len(grades) -1:
                dash_label = ttk.Label(self.scale_frame, text="-", background='#FFFFFF', font=('Helvetica', 12))
                dash_label.pack(side='left')

def main():
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
