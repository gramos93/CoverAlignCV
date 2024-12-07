import cv2
import numpy as np

from DetectTop import INPUT_PATH, VUE_COTE_PATH

SIDE_A_PATH =  f"{INPUT_PATH}/Side_A.png"

# Définir les zones_pour_image_de_cote de recherche inclinées (x, y, largeur, hauteur, angle de rotation)
zones_pour_image_de_cote = [
    (780, 720, 50, 600, -82),   # Zone pour la ligne verticale
    (1100, 1000, 200, 90, -15)  # Zone pour la ligne horizontale
]


# Fonction pour afficher l'image
def afficher_image(titre, image):
    hauteur, largeur = image.shape[:2]
    if hauteur > 800 or largeur > 1200:
        facteur_redimensionnement = min(800 / hauteur, 1200 / largeur)
        image = cv2.resize(image, (int(largeur * facteur_redimensionnement), int(hauteur * facteur_redimensionnement)))
    cv2.imshow(titre, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Fonction pour calculer les sommets d'un rectangle après rotation
def calculer_sommets(x, y, largeur, hauteur, angle):
    angle_rad = np.deg2rad(angle)
    centre_x, centre_y = x + largeur / 2, y + hauteur / 2
    coins = np.array([
        [x, y],
        [x + largeur, y],
        [x + largeur, y + hauteur],
        [x, y + hauteur]
    ], dtype=np.float64)
    # Translation pour mettre le centre à l'origine
    coins -= [centre_x, centre_y]
    # Rotation
    matrice_rotation = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    coins = coins @ matrice_rotation.T
    # Translation inverse
    coins += [centre_x, centre_y]
    return coins.astype(int)


# Fonction pour pivoter une sous-image
def pivoter_sous_image(image, angle):
    hauteur, largeur = image.shape[:2]
    centre = (largeur // 2, hauteur // 2)
    matrice_rotation = cv2.getRotationMatrix2D(centre, angle, 1.0)
    return cv2.warpAffine(image, matrice_rotation, (largeur, hauteur))


# Fonction pour extraire une sous-image inclinée
def extraire_sous_image_inclinee(image, sommets):
    masque = np.zeros_like(image, dtype=np.uint8)
    cv2.fillPoly(masque, [sommets], (255, 255, 255))
    image_masquee = cv2.bitwise_and(image, masque)
    x, y, largeur, hauteur = cv2.boundingRect(sommets)
    sous_image = image_masquee[y:y + hauteur, x:x + largeur]
    return sous_image


# Fonction pour calculer la longueur d'une ligne
def longueur_ligne(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def detecter_les_zones_dans_image_side(image_path, zones_recherche):
    # Charger et verifier le chargement de l'image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erreur lors du chargement de l'image à l'emplacement: {SIDE_A_PATH}")
        return

    image_zones = image.copy()

    # Tracer les zones_pour_image_de_cote inclinées sur l'image
    for (x, y, largeur, hauteur, angle) in zones_recherche:
        sommets = calculer_sommets(x, y, largeur, hauteur, angle)
        cv2.polylines(image_zones, [sommets], isClosed=True, color=(0, 255, 0), thickness=2)

    afficher_image("Zones de recherche inclinees", image_zones)

    # Détection et dessin de la ligne la plus longue
    for i, (x, y, largeur, hauteur, angle) in enumerate(zones_recherche):
        # Calculer les sommets de la zone inclinée
        sommets = calculer_sommets(x, y, largeur, hauteur, angle)

        # Extraire les zones_pour_image_de_cote inclinées et pivoter la sous-image
        sous_image = extraire_sous_image_inclinee(image, sommets)
        sous_image_pivotee = pivoter_sous_image(sous_image, angle)
        afficher_image(f"Sous-image pivotee (Zone {i + 1})", sous_image_pivotee)

        # Convertir en niveaux de gris
        sous_image_gris = cv2.cvtColor(sous_image_pivotee, cv2.COLOR_BGR2GRAY)

        # Détection des contours et des lignes
        sous_image_floue = cv2.GaussianBlur(sous_image_gris, (5, 5), 0)
        edges = cv2.Canny(sous_image_floue, 50, 150)
        afficher_image(f"Contours detectes (Zone {i + 1})", edges)

        lignes = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
        if lignes is not None:
            # Initialiser les variables pour la ligne la plus longue
            ligne_max = None
            longueur_max = 0

            # Parcourir toutes les lignes détectées
            for ligne in lignes:
                x1, y1, x2, y2 = ligne[0]

                # Calculer la longueur de la ligne
                longueur = longueur_ligne(x1, y1, x2, y2)

                # Vérifier si cette ligne est plus longue que la précédente
                if longueur > longueur_max:
                    longueur_max = longueur
                    ligne_max = (x1, y1, x2, y2)

            # Si une ligne a été trouvée, la dessiner
            if ligne_max is not None:
                x1, y1, x2, y2 = ligne_max
                cv2.line(sous_image_pivotee, (x1, y1), (x2, y2), (255, 0, 0), 2)
            afficher_image(f"Ligne la plus longue (Zone {i + 1})", sous_image_pivotee)


if __name__ == '__main__':
    detecter_les_zones_dans_image_side(VUE_COTE_PATH, zones_pour_image_de_cote)