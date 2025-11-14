"""
Classes de exceção customizadas para tratamento de erros na API.
"""


class ImageProcessingException(Exception):
    """Exceção base para erros de processamento de imagem."""

    def __init__(self, message: str, detail: dict = None):
        self.message = message
        self.detail = detail or {}
        super().__init__(self.message)


class InvalidParameterException(ImageProcessingException):
    """Exceção para parâmetros inválidos de filtros."""
    pass


class ImageDecodingException(ImageProcessingException):
    """Exceção para erros ao decodificar imagem."""
    pass


class FileProcessingException(ImageProcessingException):
    """Exceção para erros em operações de arquivo."""
    pass


class OpenCVException(ImageProcessingException):
    """Exceção para erros do OpenCV."""
    pass
