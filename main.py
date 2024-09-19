
import sys
import os
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QFileDialog , QPushButton,QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap, QImage , QPainter, QPen,QPalette , QPen
from PyQt5.QtCore import Qt, QTimer , pyqtSlot , QFile , QTextStream, QPoint, QEvent, QDate
from interface import Ui_MainWindow
from PIL import Image 
import shutil 
import tensorflow as tf
import numpy as np
import os
from PyQt5 import QtCore
import qrcode
import time
from Custom_Widgets.Widgets import * 
import cv2
from PIL.ImageQt import ImageQt
from PyQt5.QtPrintSupport import QPrinter, QPrintDialog
from pyzbar import pyzbar 


 
###############################################classe video capture######################################################

class VideoCapture:
    def __init__(self, video_source):
        self.video = cv2.VideoCapture(video_source)
        self.current_frame = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def start_capture(self):
        self.timer.start(30)  # Update frame every 30 milliseconds

    def stop_capture(self):
        self.timer.stop() 

    


    def update_frame(self):
        ret, frame = self.video.read()
        if ret:
            self.current_frame = frame
            frame = cv2.resize(self.current_frame, (640, 480))  # Resize frame to match label size

            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply image processing techniques to enhance quality
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # Convert enhanced image back to BGR color
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

            img = QImage(enhanced_bgr, enhanced_bgr.shape[1], enhanced_bgr.shape[0], QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(img)
            window.ui.scan_video_label.setPixmap(pixmap.scaled(window.ui.scan_video_label.size(), Qt.KeepAspectRatio))

            window.current_captured_frame = enhanced_bgr  # Store the enhanced frame

            window.ui.scan_detect_the_disease_btn.setEnabled(True)
            window.ui.scan_save_your_prediction_btn.setEnabled(True)

    def save_captured_frame(self):
        if self.current_frame is not None:
            # Apply the same image processing techniques to the captured frame
            gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

            # Save the enhanced captured frame as an image in the "uploaded_images" folder with a unique name
            image_name = f"captured_image_{int(time.time())}.jpg"
            image_path = os.path.join("uploaded_images", image_name)
            cv2.imwrite(image_path, enhanced_bgr)

            return image_path
        else:
            return None



###############################################classe video capture######################################################


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.clickPosition = None

        self.ui.icon_only_widget.hide()
        self.ui.stackedWidget.setCurrentIndex(0)
        self.ui.home_btn_2.setCheckable(True)
        loadJsonStyle(self , self.ui)
        #########################interface detected #####################################################

        self.ui.upload_image_btn.clicked.connect(self.upload_image)
        self.ui.detect_the_disease_btn.clicked.connect(self.detect_disease)
        self.ui.save_the_result_btn.clicked.connect(self.save_result)

        # Charger le modèle CNN pré-entraîné
        self.model = tf.keras.models.load_model('modele/modele.h5')

        # Désactiver le bouton "Detect Disease" et "Save Result" par défaut
        # self.ui.detect_the_disease_btn.setEnabled(False)
        # self.ui.save_the_result_btn.setEnabled(False)

        # Variables pour stocker les informations
        self.uploaded_image_path = None
        self.result_kidney_disease = None
        self.result_precision = None
        self.result_accuracy = None
        self.save_name = None
        self.save_firstname = None
        self.save_date_naissance = None
        self.save_address = None
        self.scanned_image_path = None

        #########################interface detected ###################################################




        #############################interface scan ###################################################

        self.video_capture = VideoCapture(0)  # Initialize video capture with camera index 0 (default camera)

        self.current_captured_frame = None

        self.ui.scan_start_scan_btn.clicked.connect(self.start_scan)
        self.ui.scan_detect_the_disease_btn.clicked.connect(self.scan_detect_disease)
        self.ui.scan_save_your_prediction_btn.clicked.connect(self.save_your_prediction)
        self.ui.scan_save_renalScanAI_predicion_btn.clicked.connect(self.save_renalScanAI_prediction)

        self.ui.scan_detect_the_disease_btn.setEnabled(False)
        self.ui.scan_save_your_prediction_btn.setEnabled(False)

        # # Load the pre-trained model
        # self.model = tf.keras.models.load_model("modele/modele.h5")
        self.classes = ['Renal cyst', 'Healthy kidney', 'Kidney stone', 'Renal tumor']

        #############################interface sance ###################################################
        #############################interface visualize ###################################################
        #Connect the drag and drop event
        self.setAcceptDrops(True)

        # Create a label to display the image
        self.image_label = QLabel(self.ui.visualize_image)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.ui.visualize_image.setLayout(QVBoxLayout())
        self.ui.visualize_image.layout().addWidget(self.image_label)

        # Create a label to display the generated QR code
        self.qrcode_label = QLabel(self.ui.visualize_qrcode_label)
        self.qrcode_label.setAlignment(Qt.AlignCenter)
        self.ui.visualize_qrcode_label.setLayout(QVBoxLayout())
        self.ui.visualize_qrcode_label.layout().addWidget(self.qrcode_label)

        # Variables for drawing functionality
        self.drawing = False
        self.last_point = QPoint()
        self.current_drawing_points = []  # List to store the drawing points for the current drawing
        self.drawing_objects = []  # List to store the completed drawing objects

        # Variables for zoom functionality
        self.zoom_factor = 1.0
        self.zoom_step = 0.1
        self.image_label.installEventFilter(self)

        # Variables for image movement functionality
        self.is_mouse_middle_button_pressed = False
        self.last_mouse_pos = QPoint()

        # Variable to store the current image path
        self.visualize_curent_image_path = None
        self.image_visualized_destination = None
        # Connect button click event to the corresponding function
        self.ui.visualize_save_btn.clicked.connect(self.save_image_and_generate_qrcode)

        self.ui.visualize_qrcode_label.mouseDoubleClickEvent = self.check_image_and_print_visualize_qrcode_label

        self.ui.qrcode_label.mouseDoubleClickEvent = self.check_image_and_print_qrcode_label

        self.ui.scan_qrcode_left_label.mouseDoubleClickEvent = self.check_image_and_print_scan_qrcode_left_label

        self.ui.scan_qrcode_right_label.mouseDoubleClickEvent = self.check_image_and_print_scan_qrcode_right_label


        #############################interface visualize ###################################################

        ##################Connexions des boutons aux méthodes correspondantes ###########################
        self.ui.search_btn.clicked.connect(self.on_search_btn_clicked)

        self.ui.home_btn_1.clicked.connect(self.on_home_btn_1_toggle)
        self.ui.home_btn_2.clicked.connect(self.on_home_btn_2_toggle)

        self.ui.detection_btn_1.clicked.connect(self.on_detection_btn_1_toggle)
        self.ui.detection_btn_2.clicked.connect(self.on_detection_btn_2_toggle)

        self.ui.scan_image_btn_1.clicked.connect(self.on_scan_image_btn_1_toggle)
        self.ui.scan_image_btn_2.clicked.connect(self.on_scan_image_btn_2_toggle)

        self.ui.visualize_btn_1.clicked.connect(self.on_visualize_btn_1_toggle)
        self.ui.visualize_btn_2.clicked.connect(self.on_visualize_btn_2_toggle)

        self.ui.list_detection_1.clicked.connect(self.on_list_detection_btn_1_toggle)
        self.ui.list_detection_2.clicked.connect(self.on_list_detection_btn_2_toggle)

        self.ui.scan_qrcode_btn_1.clicked.connect(self.on_scan_qrcode_btn_1_toggle)
        self.ui.scan_qrcode_btn_2.clicked.connect(self.on_scan_qrcode_btn_2_toggle)

        self.ui.info_btn_1.clicked.connect(self.on_info_btn_2_toggle)
        self.ui.info_btn_2.clicked.connect(self.on_info_btn_2_toggle)

        self.ui.help_btn_1.clicked.connect(self.on_help_btn_1_toggle)
        self.ui.help_btn_2.clicked.connect(self.on_help_btn_2_toggle)

        self.ui.stackedWidget.currentChanged.connect(self.on_stackedWidget_currentChanged)


#####################Connexions des boutons aux méthodes correspondantes ###########################
    #####################qr code scan###########################
        self.ui.qr_code_label_2.mousePressEvent = self.qr_code_start_video_capture
        self.ui.qr_scan_btn.clicked.connect(self.qr_code_scan_image)
        self.ui.qr_retour_to_scan_btn.clicked.connect(self.go_to_scan_screen)

        # Désactiver le bouton "qr_scan_btn" au démarrage
        self.ui.qr_scan_btn.setEnabled(False)
    #####################qr code scan ###########################

    #####################qr code scan fucntion ###########################
    def qr_code_start_video_capture(self, event):
        # Démarrer la capture vidéo en temps réel
        self.qr_code_video_capture = cv2.VideoCapture(0)
        self.qr_code_timer = QTimer()
        self.qr_code_timer.timeout.connect(self.qr_code_capture_frame)
        self.qr_code_timer.start(1)

        # Arrêter la capture vidéo et capturer l'image après 3 secondes
        QTimer.singleShot(3000, self.qr_code_capture_image)

    def qr_code_capture_frame(self):
        ret, frame = self.qr_code_video_capture.read()

        if ret:
            # Redimensionner le frame pour avoir la même taille que le label
            frame_resized = cv2.resize(frame, (self.ui.qr_code_label_2.width(), self.ui.qr_code_label_2.height()))

            # Convertir le frame en format QImage
            rgb_image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Afficher l'image dans le QLabel
            self.ui.qr_code_label_2.setPixmap(QPixmap.fromImage(q_image))

            # Activer le bouton "qr_scan_btn" lorsque des images sont capturées
            self.ui.qr_scan_btn.setEnabled(True)
        else:
            # Désactiver le bouton "qr_scan_btn" si aucune image n'est capturée
            self.ui.qr_scan_btn.setEnabled(False)

    def qr_code_capture_image(self):
        # Arrêter la capture vidéo
        self.qr_code_timer.stop()

        # Capturer la dernière image de la vidéo
        ret, frame = self.qr_code_video_capture.read()

        if ret:
            # Redimensionner le frame pour avoir la même taille que le label
            frame_resized = cv2.resize(frame, (self.ui.qr_code_label_2.width(), self.ui.qr_code_label_2.height()))

            # Convertir le frame en format QImage
            rgb_image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Afficher l'image capturée dans le QLabel
            self.ui.qr_code_label_2.setPixmap(QPixmap.fromImage(q_image))
        # # Libérer la webcam
        self.qr_code_video_capture.release()

        # if ret:
        #     # Redimensionner le frame pour correspondre à la taille du label
        #     frame = self.qr_code_resize_frame(frame)

        #     # Convertir le frame en format QImage
        #     rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     h, w, ch = rgb_image.shape
        #     bytes_per_line = ch * w
        #     q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        #     # Afficher l'image capturée dans le QLabel
        #     self.ui.qr_code_label_2.setPixmap(QPixmap.fromImage(q_image))

    def qr_code_scan_image(self):
        # Obtenir l'image capturée du QLabel
        pixmap = self.ui.qr_code_label_2.pixmap()

        # Convertir le QPixmap en QImage
        image = pixmap.toImage()

        # Convertir l'image en format OpenCV
        width = image.width()
        height = image.height()
        img_format = QImage.Format_RGB888 if image.hasAlphaChannel() else QImage.Format_RGB32
        ptr = image.constBits()
        ptr.setsize(image.byteCount())
        numpy_array = np.array(ptr).reshape((height, width, -1)).copy()

        # Convertir l'image en niveaux de gris
        gray_image = cv2.cvtColor(numpy_array, cv2.COLOR_BGR2GRAY)

        # Scanner l'image pour les codes QR
        results = pyzbar.decode(gray_image)

        # Afficher les résultats dans la console
        if len(results) > 0:

            for result in results:
                # Récupérer les données du résultat du code QR
                result_data = result.data.decode("utf-8") 
                # Extraire les informations du contenu du code QR
                nom = ""
                prenom = ""
                date_naissance = ""
                adresse = ""
                maladie_renale = ""
                precision = ""
                image_path = ""

                lines = result_data.split("\n")
                for line in lines:
                    if line.startswith("Nom:"):
                        nom = line.split(":")[1].strip()
                    elif line.startswith("Prenom:"):
                        prenom = line.split(":")[1].strip()
                    elif line.startswith("Date de naissance:"):
                        date_naissance = line.split(":")[1].strip()
                    elif line.startswith("Adresse:"):
                        adresse = line.split(":")[1].strip()
                    elif line.startswith("Maladie renale:"):
                        maladie_renale = line.split(":")[1].strip()
                    elif line.startswith("Precision:"):
                        precision = line.split(":")[1].strip()
                    elif line.startswith("Image:"):
                        image_path = line.split(":")[1].strip()
                # Vérifier si la valeur de maladie rénale est vide

                if not maladie_renale:
                    maladie_renale = "Not specified"
                print(maladie_renale)

                # Afficher les informations dans les labels
                self.ui.qr_name_label.setText(nom)
                self.ui.qr_firstname_label.setText(prenom)
                self.ui.qr_date_naissance_label.setText(date_naissance)
                self.ui.qr_address_label.setText(adresse)
                self.ui.qr_result_kidney_disease_label.setText(maladie_renale)

                # Afficher la valeur de précision en pourcentage si c'est un nombre
                if precision:
                    try:
                        precision_float = float(precision)
                        precision_percent = precision_float * 100
                        precision_text = f"{precision_percent:.2f}%"
                    except ValueError:
                        # Si la valeur n'est pas un nombre, afficher le résultat tel quel
                        precision_text = precision
                else:
                    precision_text = ""

                self.ui.qr_result_precision_label.setText(precision_text)
                if '%' in precision:
                    precision = precision.replace('%', '')
                    precision = float(precision) / 100

                self.ui.qr_result_accuracy_label.setText(str(precision))

                # Chargement et affichage de l'image
                if os.path.isfile(image_path):
                    image = QPixmap(image_path)
                    self.ui.qr_scan_label.setPixmap(image)
                else:
                    self.ui.qr_scan_label.setText("Image path not found")

            # Passer à l'index 0 de QStackedWidget
            self.ui.stackedWidget.setCurrentIndex(9)

        else:
            # Afficher une boîte de dialogue indiquant qu'il n'y a pas de résultat
            QMessageBox.information(self, "No results", "No QR code was found in the image.")

        # Désactiver le bouton qr_scan_btn si aucune image n'est capturée
        self.ui.qr_scan_btn.setEnabled(len(results) > 0)

    def go_to_scan_screen(self):
        # Revenir à l'écran de scan
        self.ui.stackedWidget.setCurrentIndex(5)
        self.ui.qr_scan_btn.setEnabled(False)

    #####################qr code scan function ###########################



################################ Function for changing menu page##################################

    ## Function for searching
    def on_search_btn_clicked(self):
        self.ui.stackedWidget.setCurrentIndex(8)
        search_text = self.ui.search_input.text().strip()
        if search_text : 
            self.ui.label_13.setText(search_text)
    ##Function for changing page to user page
    ## change QPushButton Checkable status when stackedWidget index changed
    def on_stackedWidget_currentChanged(self, index):
        btn_list = self.ui.icon_only_widget.findChildren(QPushButton) \
                    + self.ui.full_menu_widget.findChildren(QPushButton)
 
        for btn in btn_list:
            if index in [8, 9]:
                btn.setAutoExclusive(False)
                btn.setChecked(False)
            else : 
                btn.setAutoExclusive(True)

    def on_home_btn_1_toggle(self):
        self.ui.stackedWidget.setCurrentIndex(0)
    def on_home_btn_2_toggle(self):
        self.ui.stackedWidget.setCurrentIndex(0)


    def on_detection_btn_1_toggle(self):
        self.ui.stackedWidget.setCurrentIndex(1)
    def on_detection_btn_2_toggle(self):
        self.ui.stackedWidget.setCurrentIndex(1)


    def on_scan_image_btn_1_toggle(self):
        self.ui.stackedWidget.setCurrentIndex(2)
    def on_scan_image_btn_2_toggle(self):
        self.ui.stackedWidget.setCurrentIndex(2)

    def on_visualize_btn_1_toggle(self):
        self.ui.stackedWidget.setCurrentIndex(3)
    def on_visualize_btn_2_toggle(self):
        self.ui.stackedWidget.setCurrentIndex(3)

    def on_list_detection_btn_1_toggle(self):
        self.ui.stackedWidget.setCurrentIndex(4)
    def on_list_detection_btn_2_toggle(self):
        self.ui.stackedWidget.setCurrentIndex(4)


    def on_scan_qrcode_btn_1_toggle(self):
        self.ui.stackedWidget.setCurrentIndex(5)
    def on_scan_qrcode_btn_2_toggle(self):
        self.ui.stackedWidget.setCurrentIndex(5)

    def on_info_btn_1_toggle(self):
        self.ui.stackedWidget.setCurrentIndex(6)
    def on_info_btn_2_toggle(self):
        self.ui.stackedWidget.setCurrentIndex(6)

    def on_help_btn_1_toggle(self):
        self.ui.stackedWidget.setCurrentIndex(7)
    def on_help_btn_2_toggle(self):
        self.ui.stackedWidget.setCurrentIndex(7)

################################ Function for changing menu page##################################


################################ Function for detect menu page##################################

    def upload_image(self):
        # Ouverture de la boîte de dialogue pour sélectionner un fichier
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(None, "Upload Image", "", "Image Files (*.png *.jpg *.jpeg *.JPEG *.JPG)")

        # Vérification si un fichier a été sélectionné
        if file_path:
            # Chargement de l'image avec Pillow
            image = Image.open(file_path)
            image = image.convert("RGBA")

            # Redimensionnement de l'image sans perte de qualité
            label_size = self.ui.upload_detect_image_label.size()
            image = image.resize((label_size.width(), label_size.height()), Image.LANCZOS)

            # Correction de la rotation automatique
            image = image.rotate(self.get_image_rotation(file_path), expand=True)

            # Enregistrement de l'image dans le dossier "uploaded_images" avec un nom unique
            image_name = self.generate_unique_name(".png")
            save_path = os.path.join("uploaded_images", image_name)
            image.save(save_path)

            # Affichage de l'image dans le label
            qt_image = QPixmap.fromImage(QImage(image.tobytes(), image.size[0], image.size[1], QImage.Format_RGBA8888))
            self.ui.upload_detect_image_label.setPixmap(qt_image)

            # Activer le bouton "Detect Disease"
            self.ui.detect_the_disease_btn.setEnabled(True)

            # Stocker le chemin de l'image chargée
            self.uploaded_image_path = save_path

            # Désactiver le bouton "Save Result"
            self.ui.save_the_result_btn.setEnabled(False)

            print("Image saved:", save_path)
        else:
            # Pas d'image sélectionnée, désactiver le bouton "Detect Disease"
            self.ui.detect_the_disease_btn.setEnabled(False)

    def detect_disease(self):
        # Vérification si une image est présente dans le label
        if self.ui.upload_detect_image_label.pixmap() is None:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("warning")
            msg_box.setText("No image has been loaded.")
            msg_box.setStyleSheet("QMessageBox { background-color: #1f232a; }"
                                   "QMessageBox QLabel { color: white; }"
                                   "QMessageBox QPushButton { color: white; background-color: #2e3440; }")
            msg_box.setIcon(QMessageBox.Information)
            msg_box.exec_()

            return

        # Affichage de l'effet de scanner
        self.ui.upload_detect_image_label.setText("Scanning...")
        self.ui.upload_detect_image_label.repaint()

        # Attente de quelques secondes pour simuler le temps de scan
        QTimer.singleShot(1500, self.perform_detection)

    def perform_detection(self):
        if self.uploaded_image_path is None:
            print("No uploaded image available. Please upload an image first.")
            return

        # Récupération du chemin de l'image précédemment téléchargée
        image_path = self.uploaded_image_path

        # Chargement de l'image avec Pillow
        image = Image.open(image_path)
        image = image.convert("RGBA")

        # Conversion de l'image en tableau numpy
        width, height = image.size
        buffer = image.tobytes()
        arr = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 4))


        # Conversion de l'image en format RGB
        image_rgb = arr[:, :, :3]

        # Redimensionnement de l'image aux dimensions attendues par le modèle
        resized_image = Image.fromarray(image_rgb)
        resized_image = resized_image.resize((224, 224))

        # Prétraitement de l'image pour la passer au modèle
        img = np.array(resized_image) / 255.0
        img = np.expand_dims(img, axis=0)

        # Faire la prédiction
        predictions = self.model.predict(img)

        # Obtenir les noms des classes
        class_names = ['Renal cyst', 'Healthy kidney', 'Kidney stone', 'Renal tumor']

        # Obtenir la classe prédite
        predicted_class = class_names[np.argmax(predictions)]

        # Obtenir la précision du modèle
        accuracy = np.max(predictions)

        # Affichage des résultats
        self.result_kidney_disease = predicted_class
        self.result_precision = accuracy
        self.result_accuracy = accuracy
        self.ui.result_kidney_disease_label.setText(predicted_class)
        self.ui.result_precision_label.setText("{:.2%}".format(accuracy))
        self.ui.result_accuracy_label.setText(str(accuracy))

        # Mise à jour de self.scanned_image_path avec le chemin de l'image scannée
        self.scanned_image_path = image_path

        # Planifier l'affichage de l'image dans le label après un délai
        QTimer.singleShot(0, self.show_scanned_image)

        # Activer le bouton "Save Result"
        self.ui.save_the_result_btn.setEnabled(True)
    
    def show_scanned_image(self):
        if self.scanned_image_path:
            # Chargement de l'image scannée avec Pillow
            image = Image.open(self.scanned_image_path)
            image = image.convert("RGBA")

            # Redimensionnement de l'image sans perte de qualité
            label_size = self.ui.upload_detect_image_label.size()
            image = image.resize((label_size.width(), label_size.height()), Image.LANCZOS)

            # Conversion de l'image en format QImage
            qt_image = QImage(image.tobytes(), image.size[0], image.size[1], QImage.Format_RGBA8888)

            # Affichage de l'image dans le label
            self.ui.upload_detect_image_label.setPixmap(QPixmap.fromImage(qt_image))


    def save_result(self):
        # Vérification si une image est présente dans le label
        if self.ui.upload_detect_image_label.pixmap() is None:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("warning")
            msg_box.setText("No image has been loaded.")
            msg_box.setStyleSheet("QMessageBox { background-color: #1f232a; }"
                                   "QMessageBox QLabel { color: white; }"
                                   "QMessageBox QPushButton { color: white; background-color: #2e3440; }")
            msg_box.setIcon(QMessageBox.Information)
            msg_box.exec_()
            return

        # Vérification des champs obligatoires
        if (
                self.ui.save_name_input.text().strip() == ""
                or self.ui.save_firstname_input.text().strip() == ""
                or self.ui.save_date_naissance_input.text().strip() == ""
                or self.ui.save_address_input.text().strip() == ""
                or self.ui.result_kidney_disease_label.text().strip() == ""
                or self.ui.result_precision_label.text().strip() == ""
                or self.ui.result_accuracy_label.text().strip() == ""
        ):
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Error")
            msg_box.setText("Please fill in all fields.")
            msg_box.setStyleSheet("QMessageBox { background-color: #1f232a; }"
                                   "QMessageBox QLabel { color: white; }"
                                   "QMessageBox QPushButton { color: white; background-color: #2e3440; }")
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.exec_()
            return

        # Récupération des informations
        self.save_name = self.ui.save_name_input.text().strip()
        self.save_firstname = self.ui.save_firstname_input.text().strip()
        self.save_date_naissance = self.ui.save_date_naissance_input.text().strip()
        self.save_address = self.ui.save_address_input.text().strip()
        self.result_kidney_disease = self.ui.result_kidney_disease_label.text().strip()
        self.result_precision = self.ui.result_precision_label.text().strip()
        self.result_accuracy = self.ui.result_accuracy_label.text().strip()

        # Récupération du chemin de l'image
        image_path = "uploaded_images"

        # Obtention du dernier fichier dans le dossier "uploaded_images"
        latest_file = max([os.path.join(image_path, f) for f in os.listdir(image_path)], key=os.path.getctime)

        # Enregistrer l'image dans le dossier "image_saved"
        image_name = self.ui.save_name_input.text().strip()+"_"+self.generate_unique_name(".png")
        save_path = os.path.join("saved_images", image_name)
        shutil.copy(latest_file, save_path)
        print("Image saved:", save_path)

        # Génération du contenu du QR Code
        qr_code_content = (
                f"Nom: {self.save_name}\n"
                f"Prenom: {self.save_firstname}\n"
                f"Date de naissance: {self.save_date_naissance}\n"
                f"Addresse: {self.save_address}\n"
                f"Maladie renale: {self.result_kidney_disease}\n"
                f"Precision: {self.result_precision}\n"
                f"Accuracy: {self.result_accuracy}\n"
                f"Image: {save_path}"
        )

        # Génération du QR Code
        qr_code = qrcode.make(qr_code_content)

        # Enregistrement du QR Code avec un nom unique mais explicite
        qr_code_name = self.ui.save_name_input.text().strip()+"_"+self.generate_unique_name(".png")
        qr_code_path = os.path.join("qr_code_images", qr_code_name)
        qr_code.save(qr_code_path)

        # Affichage du QR Code dans le label
        qt_image = QPixmap(qr_code_path)
        self.ui.qrcode_label.setPixmap(qt_image)
        self.ui.save_name_input.clear()
        self.ui.save_firstname_input.clear()
        self.ui.save_address_input.clear()

        # Affichage du message de succès
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Success")
        msg_box.setText("The result has been saved successfully.")
        msg_box.setStyleSheet("QMessageBox { background-color: #1f232a; }"
                               "QMessageBox QLabel { color: white; }"
                               "QMessageBox QPushButton { color: white; background-color: #2e3440; }")
        msg_box.setIcon(QMessageBox.Information)
        msg_box.exec_()

        print("QR Code saved:", qr_code_path)

    def get_image_rotation(self, image_path):
        exif_orientation_tag = 0x0112
        try:
            with Image.open(image_path) as image:
                exif = image._getexif()
                if exif is not None:
                    orientation = exif.get(exif_orientation_tag, 1)
                    if orientation == 3:
                        return 180
                    elif orientation == 6:
                        return 270
                    elif orientation == 8:
                        return 90
        except Exception as e:
            print("Error getting image rotation:", e)
        return 0

    def generate_unique_name(self, extension):
        timestamp = str(int(time.time()))
        random_number = str(np.random.randint(1000, 9999))
        return timestamp + "_" + random_number + extension




################################ Function for detect menu page##################################
##################################Funtion for scan page###############################################


    def start_scan(self):
        self.video_capture.start_capture()
        QTimer.singleShot(4000, self.capture_frame)  # Capture frame after 4 seconds

    def capture_frame(self):
        self.video_capture.stop_capture()
        
        self.current_captured_frame = self.video_capture.current_frame
        if self.current_captured_frame is not None:
            self.current_captured_frame = cv2.resize(self.current_captured_frame, (640, 480))  # Resize frame to match label size

            # Convert frame to grayscale
            gray = cv2.cvtColor(self.current_captured_frame, cv2.COLOR_BGR2GRAY)

            # Apply image processing techniques to enhance quality
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # Convert enhanced image back to BGR color
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

            img = QImage(enhanced_bgr, enhanced_bgr.shape[1], enhanced_bgr.shape[0], QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(img)
            self.ui.scan_video_label.setPixmap(pixmap.scaled(self.ui.scan_video_label.size(), Qt.KeepAspectRatio))

            self.ui.scan_detect_the_disease_btn.setEnabled(True)
            self.ui.scan_save_your_prediction_btn.setEnabled(True)
        
            

    def scan_detect_disease(self):
        if self.current_captured_frame is not None:
            # Perform disease detection on the captured frame

            # Apply the same image processing techniques to the captured frame
            gray = cv2.cvtColor(self.current_captured_frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

            # Save the enhanced captured frame as an image in the "uploaded_images" folder with a unique name
            image_name = f"captured_image_{int(time.time())}.jpg"
            image_path = os.path.join("uploaded_images", image_name)
            cv2.imwrite(image_path, enhanced_bgr)

            # Load the image and preprocess it for prediction
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            img_array /= 255.0

            # Make predictions using the loaded model
            predictions = self.model.predict(img_array)
            prediction_index = np.argmax(predictions[0])
            disease_prediction = self.classes[prediction_index]
            disease_precision = predictions[0][prediction_index]

            # Update the GUI labels with the prediction and precision
            self.ui.scan_result_kidney_disease_prediction_label.setText(disease_prediction)
            self.ui.scan_result_kidney_precision_label.setText("{:.2%}".format(disease_precision))

            self.ui.scan_save_your_prediction_btn.setEnabled(True)
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Success")
            msg_box.setText("The image has been processed and saved successfully.")
            msg_box.setStyleSheet("QMessageBox { background-color: #1f232a; }"
                                "QMessageBox QLabel { color: #fff; }"
                                "QMessageBox QPushButton { color: #fff; background-color: #2e3440; }")
            msg_box.setIcon(QMessageBox.Information)
            msg_box.exec_()

        else:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Error")
            msg_box.setText("Please scan an image first.")
            msg_box.setStyleSheet("QMessageBox { background-color: #1f232a; }"
                                "QMessageBox QLabel { color: #fff; }"
                                "QMessageBox QPushButton { color: #fff; background-color: #2e3440; }")
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.exec_()


    def save_your_prediction(self):
        if self.current_captured_frame is not None:
            # Check if all input fields are completed
            name = self.ui.scan_name_input.text()
            firstname = self.ui.scan_firstname_input.text()
            date = self.ui.scan_date_edit_input.date().toString(Qt.ISODate)
            address = self.ui.scan_address_input.text()
            own_prediction = self.ui.scan_own_predict_input.text()
            own_precision = self.ui.scan_own_precision_input.text()
            

            if name and firstname and date and address and own_prediction and own_precision:
                # Save the captured frame as an image in the "uploaded_images" folder with a unique name
                image_path = self.video_capture.save_captured_frame()

                if image_path is not None:
                    # Generate QR code image
                    qr_data = f"Nom: {name}\nPrenom: {firstname}\nDate de naissance: {date}\nAddresse: {address}\nMaladie renale: {own_prediction}\nPrecision: {own_precision}\nAccuracy:{own_precision}"
                    qr_code = qrcode.make(qr_data)

                    # Save QR code image in the "qr_code_images" folder with a unique name
                    qr_code_name = f"{name}_{int(time.time())}.png"
                    qr_code_path = os.path.join("qr_code_images", qr_code_name)
                    # qr_code.save(qr_code_path)

                    # Move the captured frame image to the "saved_images" folder with the same name as the QR code
                    saved_image_path = os.path.join("saved_images", qr_code_name)
                    shutil.move(image_path, saved_image_path)
                    
                    # Update the QR code to include the path of the saved image
                    updated_qr_data = f"{qr_data}\nImage: {saved_image_path}"
                    updated_qr_code = qrcode.make(updated_qr_data)
                    updated_qr_code_path = os.path.join("qr_code_images", f"doctor_predict_{qr_code_name}")
                    updated_qr_code.save(updated_qr_code_path)

                    # Display QR code image in the label
                    qr_code_pixmap = QPixmap(updated_qr_code_path)
                    self.ui.scan_qrcode_left_label.setPixmap(qr_code_pixmap.scaled(
                        self.ui.scan_qrcode_left_label.size(), Qt.KeepAspectRatio))
                    msg_box = QMessageBox(self)
                    msg_box.setWindowTitle("Success")
                    msg_box.setText("The image, QR code, and the save file have been saved successfully.")
                    msg_box.setStyleSheet("QMessageBox { background-color: #1f232a; }"
                                        "QMessageBox QLabel { color: #fff; }"
                                        "QMessageBox QPushButton { color: #fff; background-color: #2e3440; }")
                    msg_box.setIcon(QMessageBox.Information)
                    msg_box.exec_()

                    self.ui.scan_name_input.clear()
                    self.ui.scan_firstname_input.clear()
                    self.ui.scan_address_input.clear()
                    self.ui.scan_own_predict_input.clear()
                    self.ui.scan_own_precision_input.clear()
                    self.ui.scan_date_edit_input.clear()

                else:
                    msg_box = QMessageBox(self)
                    msg_box.setWindowTitle("Error")
                    msg_box.setText("No image captured.")
                    msg_box.setStyleSheet("QMessageBox { background-color: #1f232a; }"
                                        "QMessageBox QLabel { color: #fff; }"
                                        "QMessageBox QPushButton { color: #fff; background-color: #2e3440; }")
                    msg_box.setIcon(QMessageBox.Critical)
                    msg_box.exec_()

            else:
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("Error")
                msg_box.setText("Please fill in all fields.")
                msg_box.setStyleSheet("QMessageBox { background-color: #1f232a; }"
                                    "QMessageBox QLabel { color: #fff; }"
                                    "QMessageBox QPushButton { color: #fff; background-color: #2e3440; }")
                msg_box.setIcon(QMessageBox.Critical)
                msg_box.exec_()

        else:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Error")
            msg_box.setText("Please scan an image first.")
            msg_box.setStyleSheet("QMessageBox { background-color: #1f232a; }"
                                "QMessageBox QLabel { color: #fff; }"
                                "QMessageBox QPushButton { color: #fff; background-color: #2e3440; }")
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.exec_()


    def save_renalScanAI_prediction(self):
        if self.current_captured_frame is not None:
            # Check if all required input fields are completed
            name = self.ui.scan_name_input.text()
            firstname = self.ui.scan_firstname_input.text()
            date = self.ui.scan_date_edit_input.date().toString(Qt.ISODate)
            address = self.ui.scan_address_input.text()
            self.result_precision = self.ui.scan_result_kidney_precision_label.text().strip()
            self.result_kidney_disease = self.ui.scan_result_kidney_disease_prediction_label.text().strip() 

            if name and firstname and date and address:
                # Save the captured frame as an image in the "uploaded_images" folder with a unique name
                image_path = self.video_capture.save_captured_frame()

                if image_path is not None:
                    # Generate QR code image
                    qr_data = f"Nom: {name}\nPrenom: {firstname}\nDate de naissance: {date}\nAddresse: {address}\nMaladie renale: {self.result_kidney_disease}\nPrecision: {self.result_precision}\nAccuracy: {self.result_precision} "
                    qr_code = qrcode.make(qr_data)

                    # Save QR code image in the "qr_code_images" folder with a unique name
                    qr_code_name = f"{name}_{int(time.time())}.png"
                    qr_code_path = os.path.join("qr_code_images", qr_code_name)
                    # qr_code.save(qr_code_path)

                    # Move the captured frame image to the "saved_images" folder with the same name as the QR code
                    saved_image_path = os.path.join("saved_images", qr_code_name)
                    shutil.move(image_path, saved_image_path)

                    # Update the QR code to include the path of the saved image
                    updated_qr_data = f"{qr_data}\nImage: {saved_image_path}"
                    updated_qr_code = qrcode.make(updated_qr_data)
                    updated_qr_code_path = os.path.join("qr_code_images", f"renalscanAI_predict_{qr_code_name}")
                    updated_qr_code.save(updated_qr_code_path)

                    # Display QR code image in the label
                    qr_code_pixmap = QPixmap(updated_qr_code_path)
                    self.ui.scan_qrcode_right_label.setPixmap(qr_code_pixmap.scaled(
                        self.ui.scan_qrcode_right_label.size(), Qt.KeepAspectRatio))
                    msg_box = QMessageBox(self)
                    msg_box.setWindowTitle("Success")
                    msg_box.setText("The image, QR code, and save file have been saved successfully.")
                    msg_box.setStyleSheet("QMessageBox { background-color: #1f232a; }"
                                        "QMessageBox QLabel { color: #fff; }"
                                        "QMessageBox QPushButton { color: #fff; background-color: #2e3440; }")
                    msg_box.setIcon(QMessageBox.Information)
                    msg_box.exec_()

                    self.ui.scan_name_input.clear()
                    self.ui.scan_firstname_input.clear()
                    self.ui.scan_address_input.clear()
                    self.ui.scan_date_edit_input.clear()
                else:
                    msg_box = QMessageBox(self)
                    msg_box.setWindowTitle("Error")
                    msg_box.setText("No image captured.")
                    msg_box.setStyleSheet("QMessageBox { background-color: #1f232a; }"
                                        "QMessageBox QLabel { color: #fff; }"
                                        "QMessageBox QPushButton { color: #fff; background-color: #2e3440; }")
                    msg_box.setIcon(QMessageBox.Critical)
                    msg_box.exec_()

            else:
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("Error")
                msg_box.setText("Please complete all fields.")
                msg_box.setStyleSheet("QMessageBox { background-color: #1f232a; }"
                                    "QMessageBox QLabel { color: #fff; }"
                                    "QMessageBox QPushButton { color: #fff; background-color: #2e3440; }")
                msg_box.setIcon(QMessageBox.Critical)
                msg_box.exec_()

        else:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Error")
            msg_box.setText("Please scan an image first.")
            msg_box.setStyleSheet("QMessageBox { background-color: #1f232a; }"
                                "QMessageBox QLabel { color: #fff; }"
                                "QMessageBox QPushButton { color: #fff; background-color: #2e3440; }")
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.exec_()



# ##################################Funtion for scan page###############################################

# ##################################Funtion visualize image###############################################

    def eventFilter(self, obj, event):
        if obj == self.image_label and event.type() == QEvent.Wheel:
            # Check if the mouse is over the image
            image_rect = self.image_label.rect()
            if image_rect.contains(event.pos()):
                # Update the zoom factor based on the mouse wheel rotation direction
                num_degrees = event.angleDelta().y() / 8  # Angle in degrees (multiples of 8)
                num_steps = num_degrees / 15  # Each 15 degrees step corresponds to an angleDelta of 120
                self.zoom_factor += num_steps * self.zoom_step
                self.zoom_factor = max(0.1, self.zoom_factor)  # Limit minimum zoom to 10%
                self.zoom_factor = min(2.0, self.zoom_factor)  # Limit maximum zoom to 200%

                # Update the displayed image with the new zoom
                self.update_image_with_zoom()

                # Indicate that the event has been handled
                return True

        return super(MainWindow, self).eventFilter(obj, event)

    def update_image_with_zoom(self):
        file_path = self.visualize_curent_image_path  # Path to the currently displayed image
        if file_path:
            image = Image.open(file_path)
            width = int(self.image_label.width() * self.zoom_factor)
            height = int(self.image_label.height() * self.zoom_factor)
            image = image.resize((width, height), Image.ANTIALIAS)
            qimage = ImageQt(image.convert("RGBA"))
            pixmap = QPixmap.fromImage(qimage)

            # Create a new pixmap and draw the stored drawing objects on it
            new_pixmap = QPixmap(pixmap.size())
            new_pixmap.fill(Qt.transparent)
            painter = QPainter(new_pixmap)
            painter.drawPixmap(0, 0, pixmap)

            # Draw the completed drawing objects on the new pixmap
            for drawing_object in self.drawing_objects:
                pen = drawing_object["pen"]
                points = drawing_object["points"]
                scaled_points = [QPoint(int(p.x() * self.zoom_factor), int(p.y() * self.zoom_factor)) for p in points]
                for i in range(len(scaled_points) - 1):
                    current_pos = scaled_points[i]
                    next_pos = scaled_points[i + 1]
                    painter.setPen(pen)
                    painter.drawLine(current_pos, next_pos)

            # Draw the current drawing points in red
            if self.drawing and len(self.current_drawing_points) > 1:
                pen = QPen(Qt.blue, 2, Qt.SolidLine)
                for i in range(len(self.current_drawing_points) - 1):
                    current_pos = self.current_drawing_points[i]
                    next_pos = self.current_drawing_points[i + 1]
                    scaled_current_pos = QPoint(int(current_pos.x() * self.zoom_factor), int(current_pos.y() * self.zoom_factor))
                    scaled_next_pos = QPoint(int(next_pos.x() * self.zoom_factor), int(next_pos.y() * self.zoom_factor))
                    painter.setPen(pen)
                    painter.drawLine(scaled_current_pos, scaled_next_pos)

            painter.end()

            self.image_label.setPixmap(new_pixmap)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if os.path.isfile(file_path):
                    if self.ui.visualize_image.rect().contains(self.ui.visualize_image.mapFromGlobal(event.pos())):
                        self.clear_drawing_objects()  # Clear drawing objects
                        self.display_image(file_path)
                        # self.save_image(file_path)
                    else:
                        msg_box = QMessageBox(self)
                        msg_box.setWindowTitle("Warning")
                        msg_box.setText("Please drop the image inside the 'visualize_image' label.")
                        msg_box.setIcon(QMessageBox.Warning)

                        msg_box.setStyleSheet("""
                            QMessageBox {
                                background-color: #16191d;
                            }
                            QMessageBox QLabel {
                                color: white;
                            }
                        """)

                        msg_box.exec_()
                else:
                    msg_box = QMessageBox(self)
                    msg_box.setWindowTitle("Warning")
                    msg_box.setText("Invalid file: {}".format(file_path))
                    msg_box.setIcon(QMessageBox.Warning)

                    msg_box.setStyleSheet("""
                        QMessageBox {
                            background-color: #16191d;
                        }
                        QMessageBox QLabel {
                            color: white;
                        }
                    """)

                    msg_box.exec_()
        else:
            event.ignore()

    def display_image(self, file_path):
        self.visualize_curent_image_path = file_path  # Save the path of the currently displayed image
        self.zoom_factor = 1.0  # Reset the zoom factor to 1.0
        self.current_drawing_points = []  # Clear the current drawing points
        self.update_image_with_zoom()

    

    def visualise_generate_unique_name(self):
        timestamp = int(time.time())
        return "image_"+str(timestamp)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.visualize_curent_image_path is not None:
            self.drawing = True
            self.last_point = self.image_label.mapFromGlobal(event.globalPos())
            self.current_drawing_points = []  # Clear the current drawing points
        elif event.button() == Qt.MiddleButton:
            self.is_mouse_middle_button_pressed = True
            self.last_mouse_pos = event.globalPos()
        
        if event.button() == Qt.LeftButton:
            self.clickPosition = event.globalPos()  # Enregistrez la position du clic

        event.accept()

    def mouseMoveEvent(self, event):
        if self.drawing:
            current_pos = self.image_label.mapFromGlobal(event.globalPos())
            self.current_drawing_points.append(current_pos)  # Store the current position
            self.update_image_with_zoom()
        elif self.is_mouse_middle_button_pressed:
            delta = event.globalPos() - self.last_mouse_pos
            self.image_label.move(self.image_label.x() + delta.x(), self.image_label.y() + delta.y())
            self.last_mouse_pos = event.globalPos()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.visualize_curent_image_path is not None:
            self.drawing = False
            if len(self.current_drawing_points) > 1:
                # Store the current drawing points and pen in the drawing objects list
                pen = QPen(Qt.red, 2, Qt.SolidLine)
                self.drawing_objects.append({"pen": pen, "points": self.current_drawing_points})
            self.current_drawing_points = []  # Clear the current drawing points
            self.update_image_with_zoom()
        elif event.button() == Qt.MiddleButton:
            self.is_mouse_middle_button_pressed = False

    def clear_drawing_objects(self):
        self.drawing_objects = []  # Clear the drawing objects list

    def validate_input_fields(self):
        # Check if an image is present in the visualize_image label
        if self.visualize_curent_image_path is None:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Warning")
            msg_box.setText("No image has been loaded.")
            msg_box.setIcon(QMessageBox.Warning)

            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: #16191d;
                }
                QMessageBox QLabel {
                    color: white;
                }
            """)

            msg_box.exec_()

            return False

        # Check if all the input fields are filled
        visualize_kidney_disease_prediction_input = self.ui.visualize_kidney_diesease_prediction_input.text()
        visualize_kidney_precision_input = self.ui.visualize_kidney_precision_input.text()
        visualize_name_input = self.ui.visualize_name_input.text()
        visualize_firstname_input = self.ui.visualize_firstname_input.text()
        visualize_date_naissance_input = self.ui.visualize_date_naissance_input.text()
        visualize_address_input = self.ui.visualize_address_input.text()

        if (
            visualize_kidney_disease_prediction_input == ""
            or visualize_kidney_precision_input == ""
            or visualize_name_input == ""
            or visualize_firstname_input == ""
            or visualize_date_naissance_input == ""
            or visualize_address_input == ""
        ):
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Warning")
            msg_box.setText("Please fill in all the fields.")
            msg_box.setIcon(QMessageBox.Warning)

            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: #16191d;
                }
                QMessageBox QLabel {
                    color: white;
                }
            """)

            msg_box.exec_()

            return False

        return True


    def visualize_generate_qr_code(self):
        # Get the data from the input fields
        visualize_kidney_disease_prediction = self.ui.visualize_kidney_diesease_prediction_input.text()
        visualize_kidney_precision = self.ui.visualize_kidney_precision_input.text()
        visualize_name = self.ui.visualize_name_input.text()
        visualize_firstname = self.ui.visualize_firstname_input.text()
        visualize_date_naissance = self.ui.visualize_date_naissance_input.text()
        visualize_address = self.ui.visualize_address_input.text()
        visualize_text_edit = self.ui.visualize_text_edit_input.toPlainText()

       
        data = f"Nom: {visualize_name}\nPrenom: {visualize_firstname}\nDate de naissance: {visualize_date_naissance}\nAddresse: {visualize_address}\nMaladie renale: {visualize_kidney_disease_prediction}\nPrecision: {visualize_kidney_precision}\nAccuracy: {visualize_kidney_precision}\nImage: {self.image_visualized_destination}\ntext_edit: {visualize_text_edit}"
                 
        # Generate the QR code with the data
        qr_code = qrcode.make(data)

        # Generate a unique filename for the QR code image
        timestamp = int(time.time())
        qr_code_filename = f"qr_code_{visualize_name}_{timestamp}.png"

        # Save the QR code image
        qr_code_filepath = os.path.join("qr_code_images", qr_code_filename)
        qr_code.save(qr_code_filepath)

        return qr_code_filepath

    
    def save_image_after_saved(self):
        # Check if an image is present in the visualize_image label
        if self.visualize_curent_image_path is None:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Warning")
            msg_box.setText("No image selected.")
            msg_box.setIcon(QMessageBox.Warning)

            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: #16191d;
                }
                QMessageBox QLabel {
                    color: white;
                }
            """)

            msg_box.exec_()
            return

        # Get the pixmap from the image_label
        pixmap = self.image_label.pixmap()

        # Check if the pixmap is available
        if pixmap is None or pixmap.isNull():
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Warning")
            msg_box.setText("No image found.")
            msg_box.setIcon(QMessageBox.Warning)

            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: #16191d;
                }
                QMessageBox QLabel {
                    color: white;
                }
            """)

            msg_box.exec_()
            return

        # Convert the pixmap to QImage
        image = pixmap.toImage()

        # Generate a unique filename for the saved image
        visualize_name = self.ui.visualize_name_input.text()
        timestamp = int(time.time())
        image_filename = f"image_visualise_{visualize_name}_{timestamp}.png"

        # Set the destination path for saving the image
        self.image_visualized_destination = os.path.join("image_visualise", image_filename)

        # Save the image
        image.save(self.image_visualized_destination)

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Success")
        msg_box.setText("Data saved successfully.")
        msg_box.setIcon(QMessageBox.Information)

        msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: #16191d;
                }
                QMessageBox QLabel {
                    color: white;
                }
            """)

        msg_box.exec_()



    def save_image_and_generate_qrcode(self):
        if self.validate_input_fields():
            # Save the visualized image
            self.save_image_after_saved()

            # Generate and display the QR code
            qr_code_filepath = self.visualize_generate_qr_code()
            qr_code_image = QImage(qr_code_filepath)

            # Scale the QR code image to fit the size of visualize_qrcode_label
            scaled_qrcode_image = qr_code_image.scaled(
                self.ui.visualize_qrcode_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

            self.ui.visualize_qrcode_label.setPixmap(QPixmap.fromImage(scaled_qrcode_image))

            self.ui.visualize_kidney_diesease_prediction_input.clear()
            self.ui.visualize_kidney_precision_input.clear()
            self.ui.visualize_name_input.clear()
            self.ui.visualize_firstname_input.clear()
            self.ui.visualize_date_naissance_input.clear()
            self.ui.visualize_address_input.clear()
            self.ui.visualize_text_edit_input.clear()

    

    
    def check_image_and_print_visualize_qrcode_label(self, event):
        # Vérifier si une image est présente dans le label visualize_qrcode_label
        if self.ui.visualize_qrcode_label.pixmap() is not None:
            # Demander à l'utilisateur de choisir le dossier de sauvegarde
            save_dir = QFileDialog.getExistingDirectory(self, "Select folder")
            if save_dir:
                # Obtenir le nom de fichier personnalisé
                filename, _ = QFileDialog.getSaveFileName(self, "Save image", save_dir, "Images (*.png)")
                if filename:
                    # Sauvegarder l'image dans le dossier sélectionné avec le nom de fichier personnalisé
                    pixmap = self.ui.visualize_qrcode_label.pixmap()
                    image = pixmap.toImage()
                    image.save(filename)
                    msg_box = QMessageBox(self)
                    msg_box.setWindowTitle("Success")
                    msg_box.setText("Data saved successfully.")
                    msg_box.setIcon(QMessageBox.Information)

                    msg_box.setStyleSheet("""
                        QMessageBox {
                            background-color: #16191d;
                        }
                        QMessageBox QLabel {
                            color: white;
                        }
                    """)

                    msg_box.exec_()
                else:
                    msg_box = QMessageBox(self)
                    msg_box.setWindowTitle("Warning")
                    msg_box.setText("No save directory selected.")
                    msg_box.setIcon(QMessageBox.Warning)

                    msg_box.setStyleSheet("""
                        QMessageBox {
                            background-color: #16191d;
                        }
                        QMessageBox QLabel {
                            color: white;
                        }
                    """)

                    msg_box.exec_()
            else:
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("Warning")
                msg_box.setText("No save directory selected.")
                msg_box.setIcon(QMessageBox.Warning)

                msg_box.setStyleSheet("""
                    QMessageBox {
                        background-color: #16191d;
                    }
                    QMessageBox QLabel {
                        color: white;
                    }
                """)

                msg_box.exec_()
        else:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Error")
            msg_box.setText("No image found.")
            msg_box.setIcon(QMessageBox.Warning)

            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: #16191d;
                }
                QMessageBox QLabel {
                    color: white;
                }
            """)

            msg_box.exec_()


    
    def check_image_and_print_qrcode_label(self, event):
        # Vérifier si une image est présente dans le label visualize_qrcode_label
        if self.ui.qrcode_label.pixmap() is not None:
            # Demander à l'utilisateur de choisir le dossier de sauvegarde
            save_dir = QFileDialog.getExistingDirectory(self, "Select folder")
            if save_dir:
                # Obtenir le nom de fichier personnalisé
                filename, _ = QFileDialog.getSaveFileName(self, "Save image", save_dir, "Images (*.png)")
                if filename:
                    # Sauvegarder l'image dans le dossier sélectionné avec le nom de fichier personnalisé
                    pixmap = self.ui.qrcode_label.pixmap()
                    image = pixmap.toImage()
                    image.save(filename)
                    msg_box = QMessageBox(self)
                    msg_box.setWindowTitle("Success")
                    msg_box.setText("Data saved successfully.")
                    msg_box.setIcon(QMessageBox.Information)

                    msg_box.setStyleSheet("""
                        QMessageBox {
                            background-color: #16191d;
                        }
                        QMessageBox QLabel {
                            color: white;
                        }
                    """)

                    msg_box.exec_()
                else:
                    msg_box = QMessageBox(self)
                    msg_box.setWindowTitle("Warning")
                    msg_box.setText("No save directory selected.")
                    msg_box.setIcon(QMessageBox.Warning)

                    msg_box.setStyleSheet("""
                        QMessageBox {
                            background-color: #16191d;
                        }
                        QMessageBox QLabel {
                            color: white;
                        }
                    """)

                    msg_box.exec_()
            else:
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("Warning")
                msg_box.setText("No save directory selected.")
                msg_box.setIcon(QMessageBox.Warning)

                msg_box.setStyleSheet("""
                    QMessageBox {
                        background-color: #16191d;
                    }
                    QMessageBox QLabel {
                        color: white;
                    }
                """)

                msg_box.exec_()
        else:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Error")
            msg_box.setText("No image found.")
            msg_box.setIcon(QMessageBox.Warning)

            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: #16191d;
                }
                QMessageBox QLabel {
                    color: white;
                }
            """)

            msg_box.exec_()


    def check_image_and_print_scan_qrcode_left_label(self, event):
        # Vérifier si une image est présente dans le label visualize_qrcode_label
        if self.ui.qrcode_label.pixmap() is not None:
            # Demander à l'utilisateur de choisir le dossier de sauvegarde
            save_dir = QFileDialog.getExistingDirectory(self, "Select folder")
            if save_dir:
                # Obtenir le nom de fichier personnalisé
                filename, _ = QFileDialog.getSaveFileName(self, "Save image", save_dir, "Images (*.png)")
                if filename:
                    # Sauvegarder l'image dans le dossier sélectionné avec le nom de fichier personnalisé
                    pixmap = self.ui.scan_qrcode_left_label.pixmap()
                    image = pixmap.toImage()
                    image.save(filename)
                    msg_box = QMessageBox(self)
                    msg_box.setWindowTitle("Success")
                    msg_box.setText("Data saved successfully.")
                    msg_box.setIcon(QMessageBox.Information)

                    msg_box.setStyleSheet("""
                        QMessageBox {
                            background-color: #16191d;
                        }
                        QMessageBox QLabel {
                            color: white;
                        }
                    """)

                    msg_box.exec_()
                else:
                    msg_box = QMessageBox(self)
                    msg_box.setWindowTitle("Warning")
                    msg_box.setText("No save directory selected.")
                    msg_box.setIcon(QMessageBox.Warning)

                    msg_box.setStyleSheet("""
                        QMessageBox {
                            background-color: #16191d;
                        }
                        QMessageBox QLabel {
                            color: white;
                        }
                    """)

                    msg_box.exec_()
            else:
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("Warning")
                msg_box.setText("No save directory selected.")
                msg_box.setIcon(QMessageBox.Warning)

                msg_box.setStyleSheet("""
                    QMessageBox {
                        background-color: #16191d;
                    }
                    QMessageBox QLabel {
                        color: white;
                    }
                """)

                msg_box.exec_()
        else:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Error")
            msg_box.setText("No image found.")
            msg_box.setIcon(QMessageBox.Warning)

            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: #16191d;
                }
                QMessageBox QLabel {
                    color: white;
                }
            """)

            msg_box.exec_()


    def check_image_and_print_scan_qrcode_right_label(self, event):
        # Vérifier si une image est présente dans le label visualize_qrcode_label
        if self.ui.qrcode_label.pixmap() is not None:
            # Demander à l'utilisateur de choisir le dossier de sauvegarde
            save_dir = QFileDialog.getExistingDirectory(self, "Select folder")
            if save_dir:
                # Obtenir le nom de fichier personnalisé
                filename, _ = QFileDialog.getSaveFileName(self, "Save image", save_dir, "Images (*.png)")
                if filename:
                    # Sauvegarder l'image dans le dossier sélectionné avec le nom de fichier personnalisé
                    pixmap = self.ui.scan_qrcode_right_label.pixmap()
                    image = pixmap.toImage()
                    image.save(filename)
                    msg_box = QMessageBox(self)
                    msg_box.setWindowTitle("Success")
                    msg_box.setText("Data saved successfully.")
                    msg_box.setIcon(QMessageBox.Information)

                    msg_box.setStyleSheet("""
                        QMessageBox {
                            background-color: #16191d;
                        }
                        QMessageBox QLabel {
                            color: white;
                        }
                    """)

                    msg_box.exec_()
                else:
                    msg_box = QMessageBox(self)
                    msg_box.setWindowTitle("Warning")
                    msg_box.setText("No save directory selected.")
                    msg_box.setIcon(QMessageBox.Warning)

                    msg_box.setStyleSheet("""
                        QMessageBox {
                            background-color: #16191d;
                        }
                        QMessageBox QLabel {
                            color: white;
                        }
                    """)

                    msg_box.exec_()
            else:
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("Warning")
                msg_box.setText("No save directory selected.")
                msg_box.setIcon(QMessageBox.Warning)

                msg_box.setStyleSheet("""
                    QMessageBox {
                        background-color: #16191d;
                    }
                    QMessageBox QLabel {
                        color: white;
                    }
                """)

                msg_box.exec_()
        else:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Error")
            msg_box.setText("No image found.")
            msg_box.setIcon(QMessageBox.Warning)

            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: #16191d;
                }
                QMessageBox QLabel {
                    color: white;
                }
            """)

            msg_box.exec_()




##################################Funtion visualize image###############################################

if __name__ == "__main__":
    app= QApplication(sys.argv)
    ## loading style file
    with open("style.qss","r") as style_file:
        style_str = style_file.read()
    app.setStyleSheet(style_str)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())