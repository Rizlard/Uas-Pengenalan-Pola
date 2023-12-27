import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Fungsi untuk memuat dataset dan mengonversi gambar ke dalam larik numpy
def load_images(folder_path, label):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        img = Image.open(os.path.join(folder_path, filename))
        img = img.resize((100, 100))  # Sesuaikan ukuran gambar jika perlu
        img_array = np.array(img)
        images.append(img_array.flatten())  # Mengubah matriks gambar menjadi vektor
        labels.append(label)
    return images, labels

# Fungsi untuk memprediksi kelas dari gambar yang diinput oleh pengguna
def predict_user_image():
    global panel, result_label
    image_path = filedialog.askopenfilename(initialdir="/", title="Pilih Gambar", filetypes=(("Image files", "*.jpg *.jpeg *.png *.gif"), ("All files", "*.*")))
    if image_path:
        img = Image.open(image_path)
        img = img.resize((100, 100))  # Sesuaikan ukuran gambar jika perlu
        img = ImageTk.PhotoImage(img)

        # Hapus gambar dan hasil prediksi sebelumnya
        panel.pack_forget()
        result_label.pack_forget()

        # Tampilkan gambar yang dipilih
        panel = tk.Label(root, image=img)
        panel.image = img
        panel.pack()

        img_array = np.array(Image.open(image_path).resize((100, 100))).flatten().reshape(1, -1)
        predicted_class = model.predict(img_array)
        class_names = ['Scissors', 'Rock', 'Paper']
        result_label = tk.Label(root, text=f"Model memprediksi gambar termasuk ke kelas: {class_names[predicted_class[0]]}")
        result_label.pack()

# Direktori yang berisi gambar gunting, batu, dan kertas (pastikan direktori sesuai)
scissors_path = 'D:/Kuliah/Pengenalan Pola/UAS/archive 2/scissors/'
rock_path = 'D:/Kuliah/Pengenalan Pola/UAS/archive 2/rock/'
paper_path = 'D:/Kuliah/Pengenalan Pola/UAS/archive 2/paper/'

# Memuat dan mengonversi gambar menjadi larik numpy
scissors_images, scissors_labels = load_images(scissors_path, 0)
rock_images, rock_labels = load_images(rock_path, 1)
paper_images, paper_labels = load_images(paper_path, 2)

# Menggabungkan gambar dan label dari setiap kelas
images = scissors_images + rock_images + paper_images
labels = scissors_labels + rock_labels + paper_labels

# Pisahkan data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Inisialisasi dan melatih model MLP
model = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Evaluasi model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi model: {accuracy * 100:.2f}%")

# Membuat GUI
root = tk.Tk()
root.title("Prediksi Gambar")
root.geometry("400x350")

# Tombol untuk memilih gambar
browse_button = tk.Button(root, text="Pilih Gambar", command=predict_user_image)
browse_button.pack()

# Panel untuk menampilkan gambar yang dipilih
panel = tk.Label(root)
panel.pack()

# Label untuk menampilkan hasil prediksi
result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()
