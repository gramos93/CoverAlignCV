from pathlib import Path
import cv2
import numpy as np


INPUT_PATH = r"./input"
VUE_DESSUS_PATH =  f"{INPUT_PATH}/Vue_Dessus.jpg"
VUE_COTE_PATH = f"{INPUT_PATH}/Vue_Cote.jpg"


# Charger l'image
image_path = VUE_DESSUS_PATH
image = cv2.imread(image_path)

# Vérification du chargement de l'image
if image is None:
    print(f"Erreur lors du chargement de l'image a l'emplacement: {image_path}")
else:
    # Définir les zones_pour_image_de_cote de recherche
    zones_recherche = [(130, 500, 90, 700), (190, 1100, 800, 100), (500, 550, 125, 125)]  # Ajout de la troisième zone
    image_zones = image.copy()

    # Initialisation
    ligne_verticale = None
    ligne_horizontale = None
    centre_cercle = None

    # Traiter chaque zone
    for i, (x, y, w, h) in enumerate(zones_recherche):
        sous_image = image[y:y + h, x:x + w]
        sous_image_gris = cv2.cvtColor(sous_image, cv2.COLOR_BGR2GRAY)

        if i < 2:  # Détection des lignes
            sous_image_floue = cv2.GaussianBlur(sous_image_gris, (5, 5), 0)
            edges = cv2.Canny(sous_image_floue, 50, 150)
            lignes = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
            if lignes is not None:
                for ligne in lignes:
                    x1, y1, x2, y2 = ligne[0]
                    if i == 0 and abs(x1 - x2) < 10:  # Ligne verticale
                        ligne_verticale = (x1 + x, y1 + y, x2 + x, y2 + y)
                    elif i == 1 and abs(y1 - y2) < 10:  # Ligne horizontale
                        ligne_horizontale = (x1 + x, y1 + y, x2 + x, y2 + y)

        elif i == 2:  # Détection du cercle
            cercles = cv2.HoughCircles(sous_image_gris, cv2.HOUGH_GRADIENT,
                                       dp=1.2, minDist=20, param1=50, param2=20, minRadius=1, maxRadius=12)
            if cercles is not None:
                cercles = np.uint16(np.around(cercles))
                x_c, y_c, r = cercles[0][0]  # Utiliser le premier cercle détecté
                centre_cercle = (x_c + x, y_c + y)
                cv2.circle(image_zones, centre_cercle, r, (0, 255, 255), 2)

    # Calcul des résultats finaux
    if ligne_verticale and ligne_horizontale and centre_cercle:
        x1_v, y1_v, x2_v, y2_v = ligne_verticale
        x1_h, y1_h, x2_h, y2_h = ligne_horizontale
        intersection = (x1_v, y1_h)
        coord_relative = (centre_cercle[0] - intersection[0], centre_cercle[1] - intersection[1])

        # Tracer les résultats sur l'image
        cv2.line(image_zones, (x1_v, y1_v), (x2_v, y2_v), (255, 0, 255), 3)
        cv2.line(image_zones, (x1_h, y1_h), (x2_h, y2_h), (255, 0, 255), 3)
        cv2.circle(image_zones, intersection, 5, (255, 0, 0), -1)

        # Afficher les résultats finaux
        print(f"Intersection : {intersection}")
        print(f"Centre du cercle : {centre_cercle}")
        print(f"Coordonnees relatives : {coord_relative}")

        # Afficher l'image finale
        hauteur, largeur = image_zones.shape[:2]
        if hauteur > 800 or largeur > 1200:
            facteur_redimensionnement = min(800 / hauteur, 1200 / largeur)
            image_zones = cv2.resize(image_zones, (int(largeur * facteur_redimensionnement), int(hauteur * facteur_redimensionnement)))
        cv2.imshow("Resultat final", image_zones)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
