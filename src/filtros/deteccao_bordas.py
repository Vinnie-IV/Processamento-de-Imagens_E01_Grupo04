import cv2
import numpy as np
from skimage import filters
from .utilitarios import converter_para_cinza
from fastapi import HTTPException
from validacao import validar_intervalo, validar_tamanho_abertura_canny, validar_ordem_limiares_canny


def borda_sobel(img: np.ndarray) -> np.ndarray:
    """
    Aplica detector de bordas Sobel.

    Args:
        img: Imagem de entrada (BGR ou escala de cinza)

    Returns:
        Imagem com bordas detectadas

    Raises:
        HTTPException: Se houver erro no processamento
    """
    try:
        # Converter para escala de cinza se necessário
        img_gray = converter_para_cinza(img)

        # Normalizar para float64 (necessário para scikit-image)
        img_normalizada = img_gray.astype(np.float64) / 255.0

        # Aplicar Sobel em X e Y
        sobel_x = filters.sobel_h(img_normalizada)
        sobel_y = filters.sobel_v(img_normalizada)

        # Calcular magnitude
        borda = np.hypot(sobel_x, sobel_y)

        # Verificar divisão por zero
        max_val = borda.max()
        if max_val == 0:
            return np.zeros_like(img_gray)

        # Normalizar para 0-255
        borda = (borda / max_val) * 255
        return borda.astype(np.uint8)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao aplicar filtro Sobel: {str(e)}"
        )


def borda_roberts(img: np.ndarray) -> np.ndarray:
    """
    Aplica detector de bordas Roberts.

    Args:
        img: Imagem de entrada (BGR ou escala de cinza)

    Returns:
        Imagem com bordas detectadas

    Raises:
        HTTPException: Se houver erro no processamento
    """
    try:
        # Converter para escala de cinza se necessário
        img_gray = converter_para_cinza(img)

        # Normalizar para float64
        img_normalizada = img_gray.astype(np.float64) / 255.0

        # Aplicar Roberts
        roberts_borda = filters.roberts(img_normalizada)

        # Verificar divisão por zero
        max_val = roberts_borda.max()
        if max_val == 0:
            return np.zeros_like(img_gray)

        # Normalizar para 0-255
        roberts_borda = (roberts_borda / max_val) * 255
        return roberts_borda.astype(np.uint8)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao aplicar filtro Roberts: {str(e)}"
        )


def borda_canny(
    img: np.ndarray,
    limiar1: int = 100,
    limiar2: int = 200,
    tamanho_abertura: int = 3,
    aplicar_blur: bool = True
) -> np.ndarray:
    """
    Aplica detector de bordas Canny.

    Args:
        img: Imagem de entrada (BGR ou escala de cinza)
        limiar1: Primeiro limiar para histerese
        limiar2: Segundo limiar para histerese
        tamanho_abertura: Tamanho da abertura para operador Sobel
        aplicar_blur: Se True, aplica blur gaussiano antes da detecção

    Returns:
        Imagem com bordas detectadas

    Raises:
        HTTPException: Se os parâmetros forem inválidos ou houver erro no OpenCV
    """
    try:
        # Validar parâmetros
        validar_intervalo(limiar1, 0, 255, "limiar1")
        validar_intervalo(limiar2, 0, 255, "limiar2")
        validar_tamanho_abertura_canny(tamanho_abertura)
        validar_ordem_limiares_canny(limiar1, limiar2)

        # Converter para escala de cinza se necessário
        img_gray = converter_para_cinza(img)

        # Garantir que está em uint8
        if img_gray.dtype != np.uint8:
            if img_gray.dtype in [np.float32, np.float64]:
                img_gray = (img_gray * 255).astype(np.uint8)
            else:
                img_gray = img_gray.astype(np.uint8)

        # Aplicar blur se solicitado
        if aplicar_blur:
            img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

        # Aplicar Canny
        edges = cv2.Canny(img_gray, limiar1, limiar2, apertureSize=tamanho_abertura)

        return edges

    except HTTPException:
        raise
    except cv2.error as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro no OpenCV ao aplicar detector Canny: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao aplicar detector Canny: {str(e)}"
        )


# Configurações de níveis para Canny
NIVEIS_CANNY = {
    1: {"limiar1": 50, "limiar2": 150, "tamanho_abertura": 3, "aplicar_blur": True},
    2: {"limiar1": 100, "limiar2": 200, "tamanho_abertura": 3, "aplicar_blur": True},
    3: {"limiar1": 150, "limiar2": 250, "tamanho_abertura": 3, "aplicar_blur": True}
}


def aplicar_canny_nivel(img: np.ndarray, nivel: int) -> np.ndarray:
    """
    Aplica Canny com configuração pré-definida por nível.

    Args:
        img: Imagem de entrada
        nivel: Nível de intensidade (1, 2 ou 3)

    Returns:
        Imagem com bordas detectadas
    """
    params = NIVEIS_CANNY.get(nivel, NIVEIS_CANNY[2])
    return borda_canny(img, **params)
