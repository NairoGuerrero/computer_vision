"""
MÓDULO DE GENERACIÓN DE IMÁGENES PANORÁMICAS MEDIANTE STITCHING

Este módulo permite la unión de múltiples imágenes utilizando técnicas de visión por computadora.
Incluye la detección de puntos clave, la coincidencia de características y la estimación de homografía
para generar una imagen panorámica a partir de varias imágenes de entrada.

Autores:
    - Nairo Guerrero Marquez
    - Fabian Esteban Quintero
    - Juan Esteban Osorio Montoya
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


class Imagestiching:
    """
    Clase para realizar la unión de imágenes mediante la técnica de stitching.
    """

    def __init__(self, images: tuple):
        """
        Inicializa la clase con las imágenes a procesar.

        Args:
            images (tuple): Tupla con las rutas de las imágenes a unir.
        """
        # Carga las imágenes desde las rutas proporcionadas
        self.images = [cv2.imread(img) for img in images]
        # Crea un detector de características ORB
        self.descriptor = cv2.ORB.create()
        # Crea un comparador de características usando el algoritmo BFMatcher con norma Hamming
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def processimages(self, image):
        """
        Convierte la imagen a escala de grises y RGB.

        Args:
            image (numpy.ndarray): Imagen en formato BGR.

        Returns:
            tuple: Imagen en escala de grises y en formato RGB.
        """
        # Convierte la imagen de BGR a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Convierte la imagen de BGR a RGB para visualización
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return gray, rgb

    def keypoints(self, image):
        """
        Detecta los puntos clave y extrae los descriptores de la imagen.

        Args:
            image (numpy.ndarray): Imagen en escala de grises.

        Returns:
            tuple: Lista de puntos clave y matriz de descriptores.
        """
        # Detecta los puntos clave y extrae los descriptores
        keypoints, features = self.descriptor.detectAndCompute(image, None)
        return keypoints, features

    def matchfeatures(self, featuresimg1, featuresimg2):
        """
        Encuentra las mejores coincidencias entre los descriptores de dos imágenes.

        Args:
            featuresimg1 (numpy.ndarray): Descriptores de la primera imagen.
            featuresimg2 (numpy.ndarray): Descriptores de la segunda imagen.

        Returns:
            list: Lista de las mejores coincidencias ordenadas.
        """
        # Encuentra las coincidencias entre los descriptores de ambas imágenes
        best = self.matcher.match(featuresimg1, featuresimg2)
        # Ordena las coincidencias según la distancia (menor es mejor)
        ordered = sorted(best, key=lambda x: x.distance)
        return ordered

    def showmatches(self, img1, keypoints1, img2, keypoints2, matches):
        """
        Muestra las coincidencias entre dos imágenes.

        Args:
            img1 (numpy.ndarray): Primera imagen.
            keypoints1 (list): Puntos clave de la primera imagen.
            img2 (numpy.ndarray): Segunda imagen.
            keypoints2 (list): Puntos clave de la segunda imagen.
            matches (list): Lista de coincidencias entre las imágenes.
        """
        # Convierte las imágenes a RGB para visualización
        _, rgb1 = self.processimages(img1)
        _, rgb2 = self.processimages(img2)
        # Dibuja las primeras 100 coincidencias
        imagenesmapeadas = cv2.drawMatches(rgb1, keypoints1, rgb2, keypoints2, matches[:100], None,
                                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # Muestra la imagen resultante con las coincidencias
        plt.figure(figsize=(20, 10))
        plt.axis('off')
        plt.imshow(imagenesmapeadas)
        plt.show()

    def homography(self, keypointsimg1, keypointsimg2, matches, umbral):
        """
        Calcula la homografía entre dos imágenes.

        Args:
            keypointsimg1 (list): Puntos clave de la primera imagen.
            keypointsimg2 (list): Puntos clave de la segunda imagen.
            matches (list): Lista de coincidencias entre las imágenes.
            umbral (float): Umbral para la estimación de RANSAC.

        Returns:
            numpy.ndarray: Matriz de transformación homográfica.
        """
        # Extrae las coordenadas de los puntos clave
        keypointsimg1 = np.float32([keypoint.pt for keypoint in keypointsimg1])
        keypointsimg2 = np.float32([keypoint.pt for keypoint in keypointsimg2])

        if len(matches) > 4:
            # Obtiene los puntos de coincidencia
            points_train = np.float32([keypointsimg1[m.queryIdx] for m in matches])
            points_query = np.float32([keypointsimg2[m.trainIdx] for m in matches])
            # Calcula la homografía usando RANSAC
            H, status = cv2.findHomography(points_train, points_query, cv2.RANSAC, umbral)
            return H
        else:
            raise ValueError("Hay menos de 4 puntos que se relacionan")

    def panorama(self):
        """
        Crea una imagen panorámica a partir de las imágenes de entrada.

        Returns:
            numpy.ndarray: Imagen panorámica resultante.
        """
        # Obtiene las imágenes cargadas
        img1, img2, img3 = self.images
        # Convierte las imágenes a escala de grises
        img1gray, _ = self.processimages(img1)
        img2gray, _ = self.processimages(img2)
        img3gray, _ = self.processimages(img3)

        # Extrae los puntos clave y descriptores
        keypoints1, features1 = self.keypoints(img1gray)
        keypoints2, features2 = self.keypoints(img2gray)
        keypoints3, features3 = self.keypoints(img3gray)

        # Encuentra coincidencias entre imágenes consecutivas
        matches12 = self.matchfeatures(features1, features2)
        matches23 = self.matchfeatures(features2, features3)

        # Muestra las coincidencias encontradas
        self.showmatches(img1, keypoints1, img2, keypoints2, matches12)
        self.showmatches(img2, keypoints2, img3, keypoints3, matches23)

        # Calcula las homografías
        H12 = self.homography(keypoints1, keypoints2, matches12, 5.0)
        H23 = self.homography(keypoints2, keypoints3, matches23, 5.0)

        # Define el tamaño del lienzo
        width = img1.shape[1] + img2.shape[1] + img3.shape[1]
        height = max(img1.shape[0], img2.shape[0], img3.shape[0])

        # Aplica la transformación de perspectiva
        panorama12 = cv2.warpPerspective(img1, H12, (width, height))
        panorama12[0:img2.shape[0], 0:img2.shape[1]] = img2
        panorama123 = cv2.warpPerspective(panorama12, H23, (width, height))
        panorama123[0:img3.shape[0], 0:img3.shape[1]] = img3

        return panorama123


if __name__ == "__main__":
    """
    Bloque principal que ejecuta el script.
    """
    images = ['img3.jpg', 'img2.jpg', 'img1.jpg']  # Lista con las rutas de las imágenes a unir
    stitcher = Imagestiching(images)  # Instancia de la clase Imagestiching
    panorama = stitcher.panorama()  # Genera la imagen panorámica
    cv2.imwrite('panorama.jpg', panorama)  # Guarda la imagen panorámica en un archivo
    cv2.imshow('Panorama', panorama)  # Muestra la imagen panorámica en una ventana
    cv2.waitKey(0)  # Espera una tecla para cerrar la ventana
    cv2.destroyAllWindows()  # Cierra todas las ventanas abiertas de OpenCV
