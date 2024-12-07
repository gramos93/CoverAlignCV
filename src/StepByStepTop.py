import cv2
import numpy as np

from DetectTop import VUE_DESSUS_PATH

# Définir la troisième zone de recherche (x, y, largeur, hauteur)
zones_pour_image_de_dessus = [
    (130, 500, 90, 700),
    (190, 1100, 800, 100),
    (500, 550, 125, 125)
 ]  # Ajout de la troisième zone


# Fonction pour afficher l'image
def afficher_image(titre, image):
    hauteur, largeur = image.shape[:2]
    if hauteur > 800 or largeur > 1200:
        facteur_redimensionnement = min(800 / hauteur, 1200 / largeur)
        image = cv2.resize(image, (int(largeur * facteur_redimensionnement), int(hauteur * facteur_redimensionnement)))
    cv2.imshow(titre, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detecter_les_zones_dans_image_top(image_path, zones_recherche):
    # Charger et verifier le chargement de l'image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erreur lors du chargement de l'image à l'emplacement: {image_path}")
        return

    image_zones = image.copy()

    # Tracer les rectangles pour les zones_pour_image_de_cote de recherche
    for (x, y, w, h) in zones_recherche:
        cv2.rectangle(image_zones, (x, y), (x + w, y + h), (0, 255, 0), 2)
    afficher_image("Zones de recherche", image_zones)

    # Initialisation
    ligne_verticale = None
    ligne_horizontale = None
    centre_cercle = None

    for i, (x, y, w, h) in enumerate(zones_recherche):
        sous_image = image[y:y + h, x:x + w]
        sous_image_gris = cv2.cvtColor(sous_image, cv2.COLOR_BGR2GRAY)
        afficher_image(f"Image en niveaux de gris (Zone {i+1})", sous_image_gris)

        if i < 2:  # Détection des lignes
            sous_image_floue = cv2.GaussianBlur(sous_image_gris, (5, 5), 0)
            grad_x = cv2.Sobel(sous_image_floue, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(sous_image_floue, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = cv2.magnitude(grad_x, grad_y)
            gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
            gradient_magnitude = np.uint8(gradient_magnitude)
            _, edges = cv2.threshold(gradient_magnitude, 50, 255, cv2.THRESH_BINARY)
            lignes = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
            if lignes is not None:
                for ligne in lignes:
                    x1, y1, x2, y2 = ligne[0]
                    if i == 0 and abs(x1 - x2) < 10:
                        ligne_verticale = (x1 + x, y1 + y, x2 + x, y2 + y)
                    elif i == 1 and abs(y1 - y2) < 10:
                        ligne_horizontale = (x1 + x, y1 + y, x2 + x, y2 + y)

        elif i == 2:  # Détection du cercle
            cercles = cv2.HoughCircles(sous_image_gris, cv2.HOUGH_GRADIENT,
                                       dp=1.2, minDist=20, param1=50, param2=20, minRadius=1, maxRadius=12)
            if cercles is not None:
                cercles = np.uint16(np.around(cercles))
                x_c, y_c, r = cercles[0][0]  # Utiliser le premier cercle détecté
                centre_cercle = (x_c + x, y_c + y)
                cv2.circle(image_zones, centre_cercle, r, (0, 255, 255), 2)

    # Si les lignes et le cercle sont détectés, calculer les coordonnées relatives
    if ligne_verticale and ligne_horizontale and centre_cercle:
        x1_v, y1_v, x2_v, y2_v = ligne_verticale
        x1_h, y1_h, x2_h, y2_h = ligne_horizontale
        intersection = (x1_v, y1_h)
        coord_relative = (centre_cercle[0] - intersection[0], centre_cercle[1] - intersection[1])

        # Tracer les lignes et l'intersection
        cv2.line(image_zones, (x1_v, y1_v), intersection, (255, 0, 255), 3)
        cv2.line(image_zones, (x2_v, y2_v), intersection, (255, 0, 255), 3)
        cv2.line(image_zones, (x1_h, y1_h), intersection, (255, 0, 255), 3)
        cv2.line(image_zones, (x2_h, y2_h), intersection, (255, 0, 255), 3)

        print(f"Les coordonnées du point d'intersection sont : {intersection}")
        print(f"Coordonnées du centre du cercle : {centre_cercle}")
        print(f"Coordonnées relatives au point d'intersection : {coord_relative}")
        cv2.circle(image_zones, intersection, 5, (255, 0, 0), -1)

    afficher_image("Resultat final", image_zones)


if __name__ == '__main__':
    detecter_les_zones_dans_image_top(VUE_DESSUS_PATH, zones_pour_image_de_dessus)