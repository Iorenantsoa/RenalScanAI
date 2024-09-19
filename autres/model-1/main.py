from PIL import Image

def inverse_colors(image_path, output_path):
    # Ouvrir l'image
    image = Image.open(image_path)

    # Obtenir les dimensions de l'image
    width, height = image.size

    # Parcourir tous les pixels de l'image
    for x in range(width):
        for y in range(height):
            # Obtenir la couleur du pixel
            pixel = image.getpixel((x, y))

            # Inverser la couleur du pixel
            inverted_pixel = (255 - pixel[0], 255 - pixel[1], 255 - pixel[2])

            # Mettre à jour le pixel dans l'image
            image.putpixel((x, y), inverted_pixel)

    # Sauvegarder l'image modifiée
    image.save(output_path)

# Exemple d'utilisation
image_path = "meth fit.PNG"
output_path = "meth fi .png"

inverse_colors(image_path, output_path)
