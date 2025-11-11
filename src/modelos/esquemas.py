from pydantic import BaseModel, Field
from typing import Optional


class RespostaFiltroJSON(BaseModel):
    """Modelo de resposta para endpoints que retornam JSON com base64"""
    imagem_original: str = Field(..., description="Imagem original em base64")
    imagem_filtrada: str = Field(..., description="Imagem filtrada em base64")
    tempo_ms: float = Field(..., description="Tempo de processamento em milissegundos")
    filtro: str = Field(..., description="Nome do filtro aplicado")
    nivel: Optional[int] = Field(None, description="Nível de intensidade do filtro (1, 2 ou 3)")
    parametros: Optional[dict] = Field(None, description="Parâmetros customizados utilizados")


class ParametrosGaussiano(BaseModel):
    """Parâmetros customizados para filtro Gaussiano"""
    kernel_width: int = Field(5, ge=1, description="Largura do kernel (deve ser ímpar)")
    kernel_height: int = Field(5, ge=1, description="Altura do kernel (deve ser ímpar)")
    sigma: float = Field(0, ge=0, description="Desvio padrão (0 = calculado automaticamente)")


class ParametrosBilateral(BaseModel):
    """Parâmetros customizados para filtro Bilateral"""
    d: int = Field(9, ge=1, description="Diâmetro da vizinhança de pixels")
    sigma_cor: int = Field(75, ge=1, description="Filtro sigma no espaço de cor")
    sigma_espaco: int = Field(75, ge=1, description="Filtro sigma no espaço de coordenadas")


class ParametrosMedia(BaseModel):
    """Parâmetros customizados para filtro de Média"""
    kernel_width: int = Field(3, ge=1, description="Largura do kernel")
    kernel_height: int = Field(3, ge=1, description="Altura do kernel")


class ParametrosMediana(BaseModel):
    """Parâmetros customizados para filtro de Mediana"""
    tamanho: int = Field(3, ge=1, description="Tamanho do kernel (deve ser ímpar)")


class ParametrosCanny(BaseModel):
    """Parâmetros customizados para detector de bordas Canny"""
    limiar1: int = Field(100, ge=0, le=255, description="Primeiro limiar para histerese")
    limiar2: int = Field(200, ge=0, le=255, description="Segundo limiar para histerese")
    tamanho_abertura: int = Field(3, ge=3, le=7, description="Tamanho da abertura Sobel")
    aplicar_blur: bool = Field(True, description="Aplicar blur gaussiano antes da detecção")
