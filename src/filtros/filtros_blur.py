import cv2
import numpy as np


def filtro_gaussiano(
    img: np.ndarray,
    kernel_width: int = 5,
    kernel_height: int = 5,
    sigma: float = 0
) -> np.ndarray:
    """
    Aplica filtro Gaussiano (blur).

    Args:
        img: Imagem de entrada
        kernel_width: Largura do kernel (deve ser ímpar)
        kernel_height: Altura do kernel (deve ser ímpar)
        sigma: Desvio padrão (0 = calculado automaticamente)

    Returns:
        Imagem com filtro aplicado
    """
    # Garantir que os valores são ímpares
    if kernel_width % 2 == 0:
        kernel_width += 1
    if kernel_height % 2 == 0:
        kernel_height += 1

    return cv2.GaussianBlur(img, (kernel_width, kernel_height), sigma)


def filtro_bilateral(
    img: np.ndarray,
    d: int = 9,
    sigma_cor: int = 75,
    sigma_espaco: int = 75
) -> np.ndarray:
    """
    Aplica filtro Bilateral (preserva bordas).

    Args:
        img: Imagem de entrada
        d: Diâmetro da vizinhança de pixels
        sigma_cor: Filtro sigma no espaço de cor
        sigma_espaco: Filtro sigma no espaço de coordenadas

    Returns:
        Imagem com filtro aplicado
    """
    return cv2.bilateralFilter(img, d, sigma_cor, sigma_espaco)


def filtro_media(
    img: np.ndarray,
    kernel_width: int = 3,
    kernel_height: int = 3
) -> np.ndarray:
    """
    Aplica filtro de Média.

    Args:
        img: Imagem de entrada
        kernel_width: Largura do kernel
        kernel_height: Altura do kernel

    Returns:
        Imagem com filtro aplicado
    """
    return cv2.blur(img, (kernel_width, kernel_height))


def filtro_mediana(img: np.ndarray, tamanho: int = 3) -> np.ndarray:
    """
    Aplica filtro de Mediana (remove ruído sal e pimenta).

    Args:
        img: Imagem de entrada
        tamanho: Tamanho do kernel (deve ser ímpar)

    Returns:
        Imagem com filtro aplicado
    """
    # Garantir que o tamanho é ímpar
    if tamanho % 2 == 0:
        tamanho += 1

    return cv2.medianBlur(img, tamanho)


# Configurações de níveis para cada filtro
NIVEIS_GAUSSIANO = {
    1: {"kernel_width": 5, "kernel_height": 5, "sigma": 0},
    2: {"kernel_width": 15, "kernel_height": 15, "sigma": 0},
    3: {"kernel_width": 35, "kernel_height": 35, "sigma": 0}
}

NIVEIS_BILATERAL = {
    1: {"d": 9, "sigma_cor": 25, "sigma_espaco": 25},
    2: {"d": 15, "sigma_cor": 50, "sigma_espaco": 50},
    3: {"d": 25, "sigma_cor": 75, "sigma_espaco": 75}
}

NIVEIS_MEDIA = {
    1: {"kernel_width": 3, "kernel_height": 3},
    2: {"kernel_width": 7, "kernel_height": 7},
    3: {"kernel_width": 15, "kernel_height": 15}
}

NIVEIS_MEDIANA = {
    1: {"tamanho": 3},
    2: {"tamanho": 7},
    3: {"tamanho": 15}
}


def aplicar_gaussiano_nivel(img: np.ndarray, nivel: int) -> np.ndarray:
    """Aplica Gaussiano com configuração pré-definida por nível."""
    params = NIVEIS_GAUSSIANO.get(nivel, NIVEIS_GAUSSIANO[2])
    return filtro_gaussiano(img, **params)


def aplicar_bilateral_nivel(img: np.ndarray, nivel: int) -> np.ndarray:
    """Aplica Bilateral com configuração pré-definida por nível."""
    params = NIVEIS_BILATERAL.get(nivel, NIVEIS_BILATERAL[2])
    return filtro_bilateral(img, **params)


def aplicar_media_nivel(img: np.ndarray, nivel: int) -> np.ndarray:
    """Aplica Média com configuração pré-definida por nível."""
    params = NIVEIS_MEDIA.get(nivel, NIVEIS_MEDIA[2])
    return filtro_media(img, **params)


def aplicar_mediana_nivel(img: np.ndarray, nivel: int) -> np.ndarray:
    """Aplica Mediana com configuração pré-definida por nível."""
    params = NIVEIS_MEDIANA.get(nivel, NIVEIS_MEDIANA[2])
    return filtro_mediana(img, **params)
