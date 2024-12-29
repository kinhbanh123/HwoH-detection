import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
current_dir = os.getcwd()

# Hàm kiểm tra hộp A nằm trong hộp B bằng cách tìm tâm của hộp A có nằm trong hộp B không
def is_inside(box_a, box_b):
    x_min_a, y_min_a, x_max_a, y_max_a = box_a
    x_min_b, y_min_b, x_max_b, y_max_b = box_b

    center_a_x = (x_min_a + x_max_a) / 2
    center_a_y = (y_min_a + y_max_a) / 2

    return x_min_b <= center_a_x <= x_max_b and y_min_b <= center_a_y <= y_max_b

# Hàm xử lý ảnh với trạng thái checkbox
def refresh_image():
    if not hasattr(refresh_image, "image_path") or not refresh_image.image_path:
        return  # Không làm gì nếu chưa có ảnh được chọn
    process_image(refresh_image.image_path)

# Hàm xử lý ảnh (chỉnh sửa để chấp nhận đường dẫn ảnh đầu vào)
def process_image(image_path=None):
    global img_result, img_display, total_people, people_without_full_clothing

    if image_path is None:  # Nếu không cung cấp đường dẫn, hiển thị hộp thoại chọn ảnh
        image_path = filedialog.askopenfilename(title="Chọn ảnh", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not image_path:
            return
        refresh_image.image_path = image_path  # Lưu đường dẫn ảnh để làm mới sau này

    # Cập nhật trạng thái
    loading_label.configure(image=loading_gif)  # Hiển thị GIF xử lý
    loading_label.image = loading_gif  # Giữ lại ảnh GIF
    app.update()

    # Ẩn dòng chữ ban đầu
    image_label.configure(text="")
    app.update()

    # Xử lý YOLO
    if current_dir[-3:] == "src": 
        yolov8_model = YOLO(f"{current_dir[:-3]}models\\last.pt")  
    else:
        yolov8_model = YOLO(f"{current_dir}\\models\\last.pt")
    results = yolov8_model([image_path])
    img = cv2.imread(image_path)

    for result in results:
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu()
        conf = boxes.conf.cpu()
        cls = boxes.cls.cpu()

        masks = {
            0: (cls == 0).nonzero(as_tuple=True)[0],
            1: (cls == 1).nonzero(as_tuple=True)[0],
            2: (cls == 2).nonzero(as_tuple=True)[0]
        }

        total_people = 0
        people_without_full_clothing = 0
        for person_idx in masks[2]:
            if conf[person_idx] < 0.5:
                continue

            total_people += 1
            person_box = xyxy[person_idx]
            contains_hat = any(is_inside(xyxy[hat_idx], person_box) for hat_idx in masks[1])
            contains_shirt = any(is_inside(xyxy[shirt_idx], person_box) for shirt_idx in masks[0])

            color = (0, 255, 0) if contains_hat and contains_shirt else (0, 0, 255)
            label = "An toan" if contains_hat and contains_shirt else "Khong an toan"
            if not (contains_hat and contains_shirt):
                people_without_full_clothing += 1

            x_min, y_min, x_max, y_max = map(int, person_box)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.rectangle(img, (x_min, y_min + (y_max - y_min) // 2 - 10), (x_min + 100, y_min + (y_max - y_min) // 2 + 10), color, -1)
            cv2.putText(img, label, (x_min + 5, y_min + (y_max - y_min) // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if show_boxes_var.get():
            for hat_idx in masks[1]:  # Mũ
                hat_box = xyxy[hat_idx]
                x_min, y_min, x_max, y_max = map(int, hat_box)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)  # Vàng
                cv2.rectangle(img, (x_min, y_min + (y_max - y_min) // 2 - 10), (x_min + 50, y_min + (y_max - y_min) // 2 + 10), (0, 255, 255), -1)
                cv2.putText(img, "Mu", (x_min + 5, y_min + (y_max - y_min) // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            for shirt_idx in masks[0]:  # Áo
                shirt_box = xyxy[shirt_idx]
                x_min, y_min, x_max, y_max = map(int, shirt_box)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Xanh dương
                cv2.rectangle(img, (x_min, y_min + (y_max - y_min) // 2 - 10), (x_min + 50, y_min + (y_max - y_min) // 2 + 10), (255, 0, 0), -1)
                cv2.putText(img, "Ao", (x_min + 5, y_min + (y_max - y_min) // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Kết quả cuối
    img_result = img
    img_display = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

    # Hiển thị kết quả
    image_label.configure(image=img_display)
    image_label.image = img_display
    loading_label.configure(image="")  # Tắt GIF khi hoàn thành

    result_label.configure(text=f"Tổng số người: {total_people}\nSố người không đủ mũ và áo: {people_without_full_clothing}")

# Hàm lưu ảnh
def save_image():
    if img_result is None:
        return
    save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
    if save_path:
        cv2.imwrite(save_path, img_result)
        print(f"Đã lưu ảnh tại: {save_path}")

# Khởi tạo giao diện
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("YOLOv8 Detection App")
app.geometry("1200x800")

# Frame chính
main_frame = ctk.CTkFrame(app)
main_frame.pack(fill="both", expand=True, padx=20, pady=20)

# Bố cục
left_frame = ctk.CTkFrame(main_frame, width=600, corner_radius=10)
right_frame = ctk.CTkFrame(main_frame, width=250, corner_radius=10)

left_frame.pack(side="left", fill="both", expand=True, padx=10)
right_frame.pack(side="right", fill="y", padx=10)

# Hiển thị ảnh
image_label = ctk.CTkLabel(left_frame, text="Chọn một ảnh để hiển thị kết quả", font=("Arial", 16), fg_color="gray")
image_label.pack(fill="both", expand=True, padx=10, pady=10)

# Hiển thị trạng thái xử lý
if current_dir[-3:] == "src": 
    loading_gif = Image.open(f"{current_dir[:-3]}loading.gif")
else:    
    loading_gif = Image.open("loading.gif")
loading_gif = loading_gif.resize((50, 50))  # Thay đổi kích thước GIF
loading_gif = ImageTk.PhotoImage(loading_gif)  # Chuyển đổi GIF sang ImageTk
loading_label = ctk.CTkLabel(left_frame, text="")
loading_label.pack(pady=20)

# Nút chọn và lưu
btn_select = ctk.CTkButton(right_frame, text="Chọn Ảnh", command=lambda: process_image(None), font=("Arial", 16), width=200)
btn_select.pack(pady=20)

btn_save = ctk.CTkButton(right_frame, text="Lưu Kết Quả", command=save_image, font=("Arial", 16), width=200)
btn_save.pack(pady=20)

# Nút tick hiển thị box
show_boxes_var = ctk.BooleanVar(value=False)
checkbox_show_boxes = ctk.CTkCheckBox(right_frame, text="Hiển thị box", variable=show_boxes_var, font=("Arial", 16))
checkbox_show_boxes.pack(pady=20)

# Liên kết nút tick với làm mới ảnh
show_boxes_var.trace_add("write", lambda *args: refresh_image())

# Hiển thị kết quả
result_label = ctk.CTkLabel(right_frame, text="", font=("Arial", 14))
result_label.pack(pady=20)

app.mainloop()
