import cv2
from pyzbar import pyzbar

def scan_qr_code(image_path):
    # Charger l'image
    image = cv2.imread(image_path)

    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Détecter les codes QR dans l'image
    barcodes = pyzbar.decode(gray)

    # Vérifier si des codes QR ont été détectés
    if len(barcodes) == 0:
        return None

    # Extraire les données du code QR
    qr_code_data = barcodes[0].data.decode("utf-8")

    return qr_code_data

# Exemple d'utilisation
qr_code_image_path = "qr_code_images/qr_code_v_1687585190.png"
result = scan_qr_code(qr_code_image_path)

if result is not None:
    print("QR Code data:", result)
else:
    print("Aucun QR Code détecté.")
