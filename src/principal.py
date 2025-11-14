from fastapi import FastAPI, UploadFile, File, HTTPException, Path, BackgroundTasks, Form, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from enum import Enum
import time
import logging

from modelos.esquemas import RespostaFiltroJSON
from filtros.utilitarios import (
    processar_imagem_upload,
    imagem_para_base64,
    criar_zip_resposta,
    limpar_arquivo_temporario
)
from filtros.deteccao_bordas import (
    borda_sobel,
    borda_roberts,
    aplicar_canny_nivel
)
from filtros.filtros_blur import (
    aplicar_gaussiano_nivel,
    aplicar_bilateral_nivel,
    aplicar_media_nivel,
    aplicar_mediana_nivel,
    filtro_gaussiano,
    filtro_bilateral,
    filtro_media,
    filtro_mediana,
    NIVEIS_GAUSSIANO,
    NIVEIS_BILATERAL,
    NIVEIS_MEDIA,
    NIVEIS_MEDIANA
)
from filtros.deteccao_bordas import borda_canny, NIVEIS_CANNY

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Enum para níveis de filtro
class NivelFiltro(int, Enum):
    """Níveis de intensidade do filtro"""
    BAIXO = 1
    NORMAL = 2
    FORTE = 3


# Enum para formatos de saída de imagem
class FormatoImagem(str, Enum):
    """Formatos de saída para imagens"""
    PNG = "png"
    JPEG = "jpeg"
    JPG = "jpg"


# Inicializar FastAPI
app = FastAPI(
    title="API de Filtros de Imagem",
    description="API para aplicar filtros de processamento de imagem com diferentes níveis de intensidade",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# EXCEPTION HANDLERS - Tratamento de Erros Global
# ============================================================================

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(_request, exc):
    """Handler para HTTPExceptions com mensagens em português."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"mensagem": exc.detail}
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request, exc):
    """Handler para erros de validação do Pydantic com mensagens em português."""
    errors = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error["loc"])
        msg = error["msg"]
        errors.append(f"{field}: {msg}")

    # Retornar mensagem consolidada
    mensagem_detalhada = "Erro de validação: " + "; ".join(errors)
    return JSONResponse(
        status_code=422,
        content={"mensagem": mensagem_detalhada}
    )


@app.exception_handler(Exception)
async def generic_exception_handler(_request, exc):
    """Handler para exceções não tratadas."""
    logger.error(f"Erro não tratado: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"mensagem": f"Erro interno do servidor: {str(exc)}"}
    )


# ============================================================================
# ENDPOINTS CUSTOMIZADOS (devem vir ANTES dos endpoints com {nivel})
# ============================================================================

# ENDPOINTS GAUSSIANO CUSTOMIZADO

@app.post("/filtros/gaussiano/customizado", response_model=RespostaFiltroJSON, tags=["Filtros Customizados"])
async def gaussiano_customizado(
    arquivo: UploadFile = File(..., description="Imagem para processar"),
    kernel_width: int = Form(5, description="Largura do kernel (deve ser ímpar)"),
    kernel_height: int = Form(5, description="Altura do kernel (deve ser ímpar)"),
    sigma: float = Form(0, description="Desvio padrão (0 = calculado automaticamente)"),
    formato: FormatoImagem = Query(FormatoImagem.PNG, description="Formato de saída da imagem")
):
    """Aplica filtro Gaussiano com parâmetros customizados (retorna JSON)."""
    inicio = time.time()

    img_original = await processar_imagem_upload(arquivo)
    img_filtrada = filtro_gaussiano(
        img_original,
        kernel_width,
        kernel_height,
        sigma
    )

    tempo_ms = (time.time() - inicio) * 1000
    logger.info(f"Filtro Gaussiano customizado gerado em {tempo_ms:.2f} ms")

    parametros = {
        "kernel_width": kernel_width,
        "kernel_height": kernel_height,
        "sigma": sigma
    }

    return RespostaFiltroJSON(
        imagem_original=imagem_para_base64(img_original, formato.value),
        imagem_filtrada=imagem_para_base64(img_filtrada, formato.value),
        tempo_ms=tempo_ms,
        filtro="gaussiano",
        parametros=parametros
    )


@app.post("/filtros/gaussiano/customizado/download", tags=["Filtros Customizados"])
async def gaussiano_customizado_download(
    background_tasks: BackgroundTasks,
    arquivo: UploadFile = File(..., description="Imagem para processar"),
    kernel_width: int = Form(5, description="Largura do kernel (deve ser ímpar)"),
    kernel_height: int = Form(5, description="Altura do kernel (deve ser ímpar)"),
    sigma: float = Form(0, description="Desvio padrão (0 = calculado automaticamente)"),
    formato: FormatoImagem = Query(FormatoImagem.PNG, description="Formato de saída da imagem")
):
    """Aplica filtro Gaussiano customizado e retorna ZIP."""
    inicio = time.time()

    img_original = await processar_imagem_upload(arquivo)
    img_filtrada = filtro_gaussiano(
        img_original,
        kernel_width,
        kernel_height,
        sigma
    )

    tempo_ms = (time.time() - inicio) * 1000
    logger.info(f"Filtro Gaussiano customizado gerado em {tempo_ms:.2f} ms")

    parametros = {
        "kernel_width": kernel_width,
        "kernel_height": kernel_height,
        "sigma": sigma
    }

    metadados = {
        "tempo_ms": tempo_ms,
        "filtro": "gaussiano",
        "parametros": parametros,
        "descricao": "Filtro Gaussiano com parâmetros customizados"
    }

    caminho_zip = criar_zip_resposta(img_original, img_filtrada, metadados, "gaussiano_customizado", formato=formato.value)
    background_tasks.add_task(limpar_arquivo_temporario, caminho_zip)

    return FileResponse(
        caminho_zip,
        media_type="application/zip",
        filename="filtro_gaussiano_customizado.zip"
    )


# ENDPOINTS BILATERAL CUSTOMIZADO

@app.post("/filtros/bilateral/customizado", response_model=RespostaFiltroJSON, tags=["Filtros Customizados"])
async def bilateral_customizado(
    arquivo: UploadFile = File(..., description="Imagem para processar"),
    d: int = Form(9, description="Diâmetro da vizinhança de pixels"),
    sigma_cor: int = Form(75, description="Filtro sigma no espaço de cor"),
    sigma_espaco: int = Form(75, description="Filtro sigma no espaço de coordenadas"),
    formato: FormatoImagem = Query(FormatoImagem.PNG, description="Formato de saída da imagem")
):
    """Aplica filtro Bilateral com parâmetros customizados (retorna JSON)."""
    inicio = time.time()

    img_original = await processar_imagem_upload(arquivo)
    img_filtrada = filtro_bilateral(
        img_original,
        d,
        sigma_cor,
        sigma_espaco
    )

    tempo_ms = (time.time() - inicio) * 1000
    logger.info(f"Filtro Bilateral customizado gerado em {tempo_ms:.2f} ms")

    parametros = {
        "d": d,
        "sigma_cor": sigma_cor,
        "sigma_espaco": sigma_espaco
    }

    return RespostaFiltroJSON(
        imagem_original=imagem_para_base64(img_original, formato.value),
        imagem_filtrada=imagem_para_base64(img_filtrada, formato.value),
        tempo_ms=tempo_ms,
        filtro="bilateral",
        parametros=parametros
    )


@app.post("/filtros/bilateral/customizado/download", tags=["Filtros Customizados"])
async def bilateral_customizado_download(
    background_tasks: BackgroundTasks,
    arquivo: UploadFile = File(..., description="Imagem para processar"),
    d: int = Form(9, description="Diâmetro da vizinhança de pixels"),
    sigma_cor: int = Form(75, description="Filtro sigma no espaço de cor"),
    sigma_espaco: int = Form(75, description="Filtro sigma no espaço de coordenadas"),
    formato: FormatoImagem = Query(FormatoImagem.PNG, description="Formato de saída da imagem")
):
    """Aplica filtro Bilateral customizado e retorna ZIP."""
    inicio = time.time()

    img_original = await processar_imagem_upload(arquivo)
    img_filtrada = filtro_bilateral(
        img_original,
        d,
        sigma_cor,
        sigma_espaco
    )

    tempo_ms = (time.time() - inicio) * 1000
    logger.info(f"Filtro Bilateral customizado gerado em {tempo_ms:.2f} ms")

    parametros = {
        "d": d,
        "sigma_cor": sigma_cor,
        "sigma_espaco": sigma_espaco
    }

    metadados = {
        "tempo_ms": tempo_ms,
        "filtro": "bilateral",
        "parametros": parametros,
        "descricao": "Filtro Bilateral com parâmetros customizados"
    }

    caminho_zip = criar_zip_resposta(img_original, img_filtrada, metadados, "bilateral_customizado", formato=formato.value)
    background_tasks.add_task(limpar_arquivo_temporario, caminho_zip)

    return FileResponse(
        caminho_zip,
        media_type="application/zip",
        filename="filtro_bilateral_customizado.zip"
    )


# ENDPOINTS MÉDIA CUSTOMIZADO

@app.post("/filtros/media/customizado", response_model=RespostaFiltroJSON, tags=["Filtros Customizados"])
async def media_customizado(
    arquivo: UploadFile = File(..., description="Imagem para processar"),
    kernel_width: int = Form(3, description="Largura do kernel"),
    kernel_height: int = Form(3, description="Altura do kernel"),
    formato: FormatoImagem = Query(FormatoImagem.PNG, description="Formato de saída da imagem")
):
    """Aplica filtro de Média com parâmetros customizados (retorna JSON)."""
    inicio = time.time()

    img_original = await processar_imagem_upload(arquivo)
    img_filtrada = filtro_media(
        img_original,
        kernel_width,
        kernel_height
    )

    tempo_ms = (time.time() - inicio) * 1000
    logger.info(f"Filtro de Média customizado gerado em {tempo_ms:.2f} ms")

    parametros = {
        "kernel_width": kernel_width,
        "kernel_height": kernel_height
    }

    return RespostaFiltroJSON(
        imagem_original=imagem_para_base64(img_original, formato.value),
        imagem_filtrada=imagem_para_base64(img_filtrada, formato.value),
        tempo_ms=tempo_ms,
        filtro="media",
        parametros=parametros
    )


@app.post("/filtros/media/customizado/download", tags=["Filtros Customizados"])
async def media_customizado_download(
    background_tasks: BackgroundTasks,
    arquivo: UploadFile = File(..., description="Imagem para processar"),
    kernel_width: int = Form(3, description="Largura do kernel"),
    kernel_height: int = Form(3, description="Altura do kernel"),
    formato: FormatoImagem = Query(FormatoImagem.PNG, description="Formato de saída da imagem")
):
    """Aplica filtro de Média customizado e retorna ZIP."""
    inicio = time.time()

    img_original = await processar_imagem_upload(arquivo)
    img_filtrada = filtro_media(
        img_original,
        kernel_width,
        kernel_height
    )

    tempo_ms = (time.time() - inicio) * 1000
    logger.info(f"Filtro de Média customizado gerado em {tempo_ms:.2f} ms")

    parametros = {
        "kernel_width": kernel_width,
        "kernel_height": kernel_height
    }

    metadados = {
        "tempo_ms": tempo_ms,
        "filtro": "media",
        "parametros": parametros,
        "descricao": "Filtro de Média com parâmetros customizados"
    }

    caminho_zip = criar_zip_resposta(img_original, img_filtrada, metadados, "media_customizado", formato=formato.value)
    background_tasks.add_task(limpar_arquivo_temporario, caminho_zip)

    return FileResponse(
        caminho_zip,
        media_type="application/zip",
        filename="filtro_media_customizado.zip"
    )


# ENDPOINTS MEDIANA CUSTOMIZADO

@app.post("/filtros/mediana/customizado", response_model=RespostaFiltroJSON, tags=["Filtros Customizados"])
async def mediana_customizado(
    arquivo: UploadFile = File(..., description="Imagem para processar"),
    tamanho: int = Form(3, description="Tamanho do kernel (deve ser ímpar)"),
    formato: FormatoImagem = Query(FormatoImagem.PNG, description="Formato de saída da imagem")
):
    """Aplica filtro de Mediana com parâmetros customizados (retorna JSON)."""
    inicio = time.time()

    img_original = await processar_imagem_upload(arquivo)
    img_filtrada = filtro_mediana(img_original, tamanho)

    tempo_ms = (time.time() - inicio) * 1000
    logger.info(f"Filtro de Mediana customizado gerado em {tempo_ms:.2f} ms")

    parametros = {"tamanho": tamanho}

    return RespostaFiltroJSON(
        imagem_original=imagem_para_base64(img_original, formato.value),
        imagem_filtrada=imagem_para_base64(img_filtrada, formato.value),
        tempo_ms=tempo_ms,
        filtro="mediana",
        parametros=parametros
    )


@app.post("/filtros/mediana/customizado/download", tags=["Filtros Customizados"])
async def mediana_customizado_download(
    background_tasks: BackgroundTasks,
    arquivo: UploadFile = File(..., description="Imagem para processar"),
    tamanho: int = Form(3, description="Tamanho do kernel (deve ser ímpar)"),
    formato: FormatoImagem = Query(FormatoImagem.PNG, description="Formato de saída da imagem")
):
    """Aplica filtro de Mediana customizado e retorna ZIP."""
    inicio = time.time()

    img_original = await processar_imagem_upload(arquivo)
    img_filtrada = filtro_mediana(img_original, tamanho)

    tempo_ms = (time.time() - inicio) * 1000
    logger.info(f"Filtro de Mediana customizado gerado em {tempo_ms:.2f} ms")

    parametros = {"tamanho": tamanho}

    metadados = {
        "tempo_ms": tempo_ms,
        "filtro": "mediana",
        "parametros": parametros,
        "descricao": "Filtro de Mediana com parâmetros customizados"
    }

    caminho_zip = criar_zip_resposta(img_original, img_filtrada, metadados, "mediana_customizado", formato=formato.value)
    background_tasks.add_task(limpar_arquivo_temporario, caminho_zip)

    return FileResponse(
        caminho_zip,
        media_type="application/zip",
        filename="filtro_mediana_customizado.zip"
    )


# ENDPOINTS CANNY CUSTOMIZADO

@app.post("/filtros/canny/customizado", response_model=RespostaFiltroJSON, tags=["Filtros Customizados"])
async def canny_customizado(
    arquivo: UploadFile = File(..., description="Imagem para processar"),
    limiar1: int = Form(100, description="Primeiro limiar para histerese (0-255)"),
    limiar2: int = Form(200, description="Segundo limiar para histerese (0-255)"),
    tamanho_abertura: int = Form(3, description="Tamanho da abertura Sobel (3-7)"),
    aplicar_blur: bool = Form(True, description="Aplicar blur gaussiano antes da detecção"),
    formato: FormatoImagem = Query(FormatoImagem.PNG, description="Formato de saída da imagem")
):
    """Aplica detector Canny com parâmetros customizados (retorna JSON)."""
    inicio = time.time()

    img_original = await processar_imagem_upload(arquivo)
    img_filtrada = borda_canny(
        img_original,
        limiar1,
        limiar2,
        tamanho_abertura,
        aplicar_blur
    )

    tempo_ms = (time.time() - inicio) * 1000
    logger.info(f"Filtro Canny customizado gerado em {tempo_ms:.2f} ms")

    parametros = {
        "limiar1": limiar1,
        "limiar2": limiar2,
        "tamanho_abertura": tamanho_abertura,
        "aplicar_blur": aplicar_blur
    }

    return RespostaFiltroJSON(
        imagem_original=imagem_para_base64(img_original, formato.value),
        imagem_filtrada=imagem_para_base64(img_filtrada, formato.value),
        tempo_ms=tempo_ms,
        filtro="canny",
        parametros=parametros
    )


@app.post("/filtros/canny/customizado/download", tags=["Filtros Customizados"])
async def canny_customizado_download(
    background_tasks: BackgroundTasks,
    arquivo: UploadFile = File(..., description="Imagem para processar"),
    limiar1: int = Form(100, description="Primeiro limiar para histerese (0-255)"),
    limiar2: int = Form(200, description="Segundo limiar para histerese (0-255)"),
    tamanho_abertura: int = Form(3, description="Tamanho da abertura Sobel (3-7)"),
    aplicar_blur: bool = Form(True, description="Aplicar blur gaussiano antes da detecção"),
    formato: FormatoImagem = Query(FormatoImagem.PNG, description="Formato de saída da imagem")
):
    """Aplica detector Canny customizado e retorna ZIP."""
    inicio = time.time()

    img_original = await processar_imagem_upload(arquivo)
    img_filtrada = borda_canny(
        img_original,
        limiar1,
        limiar2,
        tamanho_abertura,
        aplicar_blur
    )

    tempo_ms = (time.time() - inicio) * 1000
    logger.info(f"Filtro Canny customizado gerado em {tempo_ms:.2f} ms")

    parametros = {
        "limiar1": limiar1,
        "limiar2": limiar2,
        "tamanho_abertura": tamanho_abertura,
        "aplicar_blur": aplicar_blur
    }

    metadados = {
        "tempo_ms": tempo_ms,
        "filtro": "canny",
        "parametros": parametros,
        "descricao": "Detector de bordas Canny com parâmetros customizados"
    }

    caminho_zip = criar_zip_resposta(img_original, img_filtrada, metadados, "canny_customizado", formato=formato.value)
    background_tasks.add_task(limpar_arquivo_temporario, caminho_zip)

    return FileResponse(
        caminho_zip,
        media_type="application/zip",
        filename="filtro_canny_customizado.zip"
    )


# ============================================================================
# ENDPOINTS COM NÍVEIS (vêm DEPOIS dos customizados)
# ============================================================================

# ENDPOINTS SOBEL

@app.post("/filtros/sobel/{nivel}", response_model=RespostaFiltroJSON, tags=["Detecção de Bordas"])
async def aplicar_sobel(
    nivel: NivelFiltro = Path(..., description="Nível do filtro (1=baixo, 2=normal, 3=forte)"),
    arquivo: UploadFile = File(..., description="Imagem para processar"),
    formato: FormatoImagem = Query(FormatoImagem.PNG, description="Formato de saída da imagem")
):
    """Aplica filtro Sobel para detecção de bordas (retorna JSON com base64)."""
    inicio = time.time()

    img_original = await processar_imagem_upload(arquivo)
    img_filtrada = borda_sobel(img_original)

    tempo_ms = (time.time() - inicio) * 1000
    logger.info(f"Filtro Sobel (nível {nivel}) gerado em {tempo_ms:.2f} ms")

    return RespostaFiltroJSON(
        imagem_original=imagem_para_base64(img_original, formato.value),
        imagem_filtrada=imagem_para_base64(img_filtrada, formato.value),
        tempo_ms=tempo_ms,
        filtro="sobel",
        nivel=nivel
    )


@app.post("/filtros/sobel/{nivel}/download", tags=["Detecção de Bordas"])
async def download_sobel(
    background_tasks: BackgroundTasks,
    nivel: NivelFiltro = Path(..., description="Nível do filtro"),
    arquivo: UploadFile = File(..., description="Imagem para processar"),
    formato: FormatoImagem = Query(FormatoImagem.PNG, description="Formato de saída da imagem")
):
    """Aplica filtro Sobel e retorna ZIP com imagens."""
    inicio = time.time()

    img_original = await processar_imagem_upload(arquivo)
    img_filtrada = borda_sobel(img_original)

    tempo_ms = (time.time() - inicio) * 1000
    logger.info(f"Filtro Sobel (nível {nivel}) gerado em {tempo_ms:.2f} ms")

    metadados = {
        "tempo_ms": tempo_ms,
        "filtro": "sobel",
        "nivel": nivel,
        "descricao": "Detector de bordas Sobel"
    }

    caminho_zip = criar_zip_resposta(img_original, img_filtrada, metadados, "sobel", nivel, formato.value)
    background_tasks.add_task(limpar_arquivo_temporario, caminho_zip)

    return FileResponse(
        caminho_zip,
        media_type="application/zip",
        filename=f"filtro_sobel_nivel{nivel}.zip"
    )


# ENDPOINTS ROBERTS

@app.post("/filtros/roberts/{nivel}", response_model=RespostaFiltroJSON, tags=["Detecção de Bordas"])
async def aplicar_roberts(
    nivel: NivelFiltro = Path(..., description="Nível do filtro"),
    arquivo: UploadFile = File(..., description="Imagem para processar"),
    formato: FormatoImagem = Query(FormatoImagem.PNG, description="Formato de saída da imagem")
):
    """Aplica filtro Roberts para detecção de bordas (retorna JSON com base64)."""
    inicio = time.time()

    img_original = await processar_imagem_upload(arquivo)
    img_filtrada = borda_roberts(img_original)

    tempo_ms = (time.time() - inicio) * 1000
    logger.info(f"Filtro Roberts (nível {nivel}) gerado em {tempo_ms:.2f} ms")

    return RespostaFiltroJSON(
        imagem_original=imagem_para_base64(img_original, formato.value),
        imagem_filtrada=imagem_para_base64(img_filtrada, formato.value),
        tempo_ms=tempo_ms,
        filtro="roberts",
        nivel=nivel
    )


@app.post("/filtros/roberts/{nivel}/download", tags=["Detecção de Bordas"])
async def download_roberts(
    background_tasks: BackgroundTasks,
    nivel: NivelFiltro = Path(..., description="Nível do filtro"),
    arquivo: UploadFile = File(..., description="Imagem para processar"),
    formato: FormatoImagem = Query(FormatoImagem.PNG, description="Formato de saída da imagem")
):
    """Aplica filtro Roberts e retorna ZIP com imagens."""
    inicio = time.time()

    img_original = await processar_imagem_upload(arquivo)
    img_filtrada = borda_roberts(img_original)

    tempo_ms = (time.time() - inicio) * 1000
    logger.info(f"Filtro Roberts (nível {nivel}) gerado em {tempo_ms:.2f} ms")

    metadados = {
        "tempo_ms": tempo_ms,
        "filtro": "roberts",
        "nivel": nivel,
        "descricao": "Detector de bordas Roberts"
    }

    caminho_zip = criar_zip_resposta(img_original, img_filtrada, metadados, "roberts", nivel, formato.value)
    background_tasks.add_task(limpar_arquivo_temporario, caminho_zip)

    return FileResponse(
        caminho_zip,
        media_type="application/zip",
        filename=f"filtro_roberts_nivel{nivel}.zip"
    )


# ENDPOINTS CANNY

@app.post("/filtros/canny/{nivel}", response_model=RespostaFiltroJSON, tags=["Detecção de Bordas"])
async def aplicar_canny(
    nivel: NivelFiltro = Path(..., description="Nível do filtro"),
    arquivo: UploadFile = File(..., description="Imagem para processar"),
    formato: FormatoImagem = Query(FormatoImagem.PNG, description="Formato de saída da imagem")
):
    """Aplica detector de bordas Canny (retorna JSON com base64)."""
    inicio = time.time()

    img_original = await processar_imagem_upload(arquivo)
    img_filtrada = aplicar_canny_nivel(img_original, nivel)

    tempo_ms = (time.time() - inicio) * 1000
    logger.info(f"Filtro Canny (nível {nivel}) gerado em {tempo_ms:.2f} ms")

    return RespostaFiltroJSON(
        imagem_original=imagem_para_base64(img_original, formato.value),
        imagem_filtrada=imagem_para_base64(img_filtrada, formato.value),
        tempo_ms=tempo_ms,
        filtro="canny",
        nivel=nivel,
        parametros=NIVEIS_CANNY[nivel]
    )


@app.post("/filtros/canny/{nivel}/download", tags=["Detecção de Bordas"])
async def download_canny(
    background_tasks: BackgroundTasks,
    nivel: NivelFiltro = Path(..., description="Nível do filtro"),
    arquivo: UploadFile = File(..., description="Imagem para processar"),
    formato: FormatoImagem = Query(FormatoImagem.PNG, description="Formato de saída da imagem")
):
    """Aplica detector Canny e retorna ZIP com imagens."""
    inicio = time.time()

    img_original = await processar_imagem_upload(arquivo)
    img_filtrada = aplicar_canny_nivel(img_original, nivel)

    tempo_ms = (time.time() - inicio) * 1000
    logger.info(f"Filtro Canny (nível {nivel}) gerado em {tempo_ms:.2f} ms")

    metadados = {
        "tempo_ms": tempo_ms,
        "filtro": "canny",
        "nivel": nivel,
        "parametros": NIVEIS_CANNY[nivel],
        "descricao": "Detector de bordas Canny"
    }

    caminho_zip = criar_zip_resposta(img_original, img_filtrada, metadados, "canny", nivel, formato.value)
    background_tasks.add_task(limpar_arquivo_temporario, caminho_zip)

    return FileResponse(
        caminho_zip,
        media_type="application/zip",
        filename=f"filtro_canny_nivel{nivel}.zip"
    )


# ENDPOINTS GAUSSIANO

@app.post("/filtros/gaussiano/{nivel}", response_model=RespostaFiltroJSON, tags=["Filtros de Blur"])
async def aplicar_gaussiano(
    nivel: NivelFiltro = Path(..., description="Nível do filtro"),
    arquivo: UploadFile = File(..., description="Imagem para processar"),
    formato: FormatoImagem = Query(FormatoImagem.PNG, description="Formato de saída da imagem")
):
    """Aplica filtro Gaussiano (retorna JSON com base64)."""
    inicio = time.time()

    img_original = await processar_imagem_upload(arquivo)
    img_filtrada = aplicar_gaussiano_nivel(img_original, nivel)

    tempo_ms = (time.time() - inicio) * 1000
    logger.info(f"Filtro Gaussiano (nível {nivel}) gerado em {tempo_ms:.2f} ms")

    return RespostaFiltroJSON(
        imagem_original=imagem_para_base64(img_original, formato.value),
        imagem_filtrada=imagem_para_base64(img_filtrada, formato.value),
        tempo_ms=tempo_ms,
        filtro="gaussiano",
        nivel=nivel,
        parametros=NIVEIS_GAUSSIANO[nivel]
    )


@app.post("/filtros/gaussiano/{nivel}/download", tags=["Filtros de Blur"])
async def download_gaussiano(
    background_tasks: BackgroundTasks,
    nivel: NivelFiltro = Path(..., description="Nível do filtro"),
    arquivo: UploadFile = File(..., description="Imagem para processar"),
    formato: FormatoImagem = Query(FormatoImagem.PNG, description="Formato de saída da imagem")
):
    """Aplica filtro Gaussiano e retorna ZIP com imagens."""
    inicio = time.time()

    img_original = await processar_imagem_upload(arquivo)
    img_filtrada = aplicar_gaussiano_nivel(img_original, nivel)

    tempo_ms = (time.time() - inicio) * 1000
    logger.info(f"Filtro Gaussiano (nível {nivel}) gerado em {tempo_ms:.2f} ms")

    metadados = {
        "tempo_ms": tempo_ms,
        "filtro": "gaussiano",
        "nivel": nivel,
        "parametros": NIVEIS_GAUSSIANO[nivel],
        "descricao": "Filtro Gaussiano (blur)"
    }

    caminho_zip = criar_zip_resposta(img_original, img_filtrada, metadados, "gaussiano", nivel, formato.value)
    background_tasks.add_task(limpar_arquivo_temporario, caminho_zip)

    return FileResponse(
        caminho_zip,
        media_type="application/zip",
        filename=f"filtro_gaussiano_nivel{nivel}.zip"
    )


# ENDPOINTS BILATERAL

@app.post("/filtros/bilateral/{nivel}", response_model=RespostaFiltroJSON, tags=["Filtros de Blur"])
async def aplicar_bilateral(
    nivel: NivelFiltro = Path(..., description="Nível do filtro"),
    arquivo: UploadFile = File(..., description="Imagem para processar"),
    formato: FormatoImagem = Query(FormatoImagem.PNG, description="Formato de saída da imagem")
):
    """Aplica filtro Bilateral (retorna JSON com base64)."""
    inicio = time.time()

    img_original = await processar_imagem_upload(arquivo)
    img_filtrada = aplicar_bilateral_nivel(img_original, nivel)

    tempo_ms = (time.time() - inicio) * 1000
    logger.info(f"Filtro Bilateral (nível {nivel}) gerado em {tempo_ms:.2f} ms")

    return RespostaFiltroJSON(
        imagem_original=imagem_para_base64(img_original, formato.value),
        imagem_filtrada=imagem_para_base64(img_filtrada, formato.value),
        tempo_ms=tempo_ms,
        filtro="bilateral",
        nivel=nivel,
        parametros=NIVEIS_BILATERAL[nivel]
    )


@app.post("/filtros/bilateral/{nivel}/download", tags=["Filtros de Blur"])
async def download_bilateral(
    background_tasks: BackgroundTasks,
    nivel: NivelFiltro = Path(..., description="Nível do filtro"),
    arquivo: UploadFile = File(..., description="Imagem para processar"),
    formato: FormatoImagem = Query(FormatoImagem.PNG, description="Formato de saída da imagem")
):
    """Aplica filtro Bilateral e retorna ZIP com imagens."""
    inicio = time.time()

    img_original = await processar_imagem_upload(arquivo)
    img_filtrada = aplicar_bilateral_nivel(img_original, nivel)

    tempo_ms = (time.time() - inicio) * 1000
    logger.info(f"Filtro Bilateral (nível {nivel}) gerado em {tempo_ms:.2f} ms")

    metadados = {
        "tempo_ms": tempo_ms,
        "filtro": "bilateral",
        "nivel": nivel,
        "parametros": NIVEIS_BILATERAL[nivel],
        "descricao": "Filtro Bilateral (preserva bordas)"
    }

    caminho_zip = criar_zip_resposta(img_original, img_filtrada, metadados, "bilateral", nivel, formato.value)
    background_tasks.add_task(limpar_arquivo_temporario, caminho_zip)

    return FileResponse(
        caminho_zip,
        media_type="application/zip",
        filename=f"filtro_bilateral_nivel{nivel}.zip"
    )


# ENDPOINTS MÉDIA

@app.post("/filtros/media/{nivel}", response_model=RespostaFiltroJSON, tags=["Filtros de Blur"])
async def aplicar_media(
    nivel: NivelFiltro = Path(..., description="Nível do filtro"),
    arquivo: UploadFile = File(..., description="Imagem para processar"),
    formato: FormatoImagem = Query(FormatoImagem.PNG, description="Formato de saída da imagem")
):
    """Aplica filtro de Média (retorna JSON com base64)."""
    inicio = time.time()

    img_original = await processar_imagem_upload(arquivo)
    img_filtrada = aplicar_media_nivel(img_original, nivel)

    tempo_ms = (time.time() - inicio) * 1000
    logger.info(f"Filtro de Média (nível {nivel}) gerado em {tempo_ms:.2f} ms")

    return RespostaFiltroJSON(
        imagem_original=imagem_para_base64(img_original, formato.value),
        imagem_filtrada=imagem_para_base64(img_filtrada, formato.value),
        tempo_ms=tempo_ms,
        filtro="media",
        nivel=nivel,
        parametros=NIVEIS_MEDIA[nivel]
    )


@app.post("/filtros/media/{nivel}/download", tags=["Filtros de Blur"])
async def download_media(
    background_tasks: BackgroundTasks,
    nivel: NivelFiltro = Path(..., description="Nível do filtro"),
    arquivo: UploadFile = File(..., description="Imagem para processar"),
    formato: FormatoImagem = Query(FormatoImagem.PNG, description="Formato de saída da imagem")
):
    """Aplica filtro de Média e retorna ZIP com imagens."""
    inicio = time.time()

    img_original = await processar_imagem_upload(arquivo)
    img_filtrada = aplicar_media_nivel(img_original, nivel)

    tempo_ms = (time.time() - inicio) * 1000
    logger.info(f"Filtro de Média (nível {nivel}) gerado em {tempo_ms:.2f} ms")

    metadados = {
        "tempo_ms": tempo_ms,
        "filtro": "media",
        "nivel": nivel,
        "parametros": NIVEIS_MEDIA[nivel],
        "descricao": "Filtro de Média (blur uniforme)"
    }

    caminho_zip = criar_zip_resposta(img_original, img_filtrada, metadados, "media", nivel, formato.value)
    background_tasks.add_task(limpar_arquivo_temporario, caminho_zip)

    return FileResponse(
        caminho_zip,
        media_type="application/zip",
        filename=f"filtro_media_nivel{nivel}.zip"
    )


# ENDPOINTS MEDIANA

@app.post("/filtros/mediana/{nivel}", response_model=RespostaFiltroJSON, tags=["Filtros de Blur"])
async def aplicar_mediana(
    nivel: NivelFiltro = Path(..., description="Nível do filtro"),
    arquivo: UploadFile = File(..., description="Imagem para processar"),
    formato: FormatoImagem = Query(FormatoImagem.PNG, description="Formato de saída da imagem")
):
    """Aplica filtro de Mediana (retorna JSON com base64)."""
    inicio = time.time()

    img_original = await processar_imagem_upload(arquivo)
    img_filtrada = aplicar_mediana_nivel(img_original, nivel)

    tempo_ms = (time.time() - inicio) * 1000
    logger.info(f"Filtro de Mediana (nível {nivel}) gerado em {tempo_ms:.2f} ms")

    return RespostaFiltroJSON(
        imagem_original=imagem_para_base64(img_original, formato.value),
        imagem_filtrada=imagem_para_base64(img_filtrada, formato.value),
        tempo_ms=tempo_ms,
        filtro="mediana",
        nivel=nivel,
        parametros=NIVEIS_MEDIANA[nivel]
    )


@app.post("/filtros/mediana/{nivel}/download", tags=["Filtros de Blur"])
async def download_mediana(
    background_tasks: BackgroundTasks,
    nivel: NivelFiltro = Path(..., description="Nível do filtro"),
    arquivo: UploadFile = File(..., description="Imagem para processar"),
    formato: FormatoImagem = Query(FormatoImagem.PNG, description="Formato de saída da imagem")
):
    """Aplica filtro de Mediana e retorna ZIP com imagens."""
    inicio = time.time()

    img_original = await processar_imagem_upload(arquivo)
    img_filtrada = aplicar_mediana_nivel(img_original, nivel)

    tempo_ms = (time.time() - inicio) * 1000
    logger.info(f"Filtro de Mediana (nível {nivel}) gerado em {tempo_ms:.2f} ms")

    metadados = {
        "tempo_ms": tempo_ms,
        "filtro": "mediana",
        "nivel": nivel,
        "parametros": NIVEIS_MEDIANA[nivel],
        "descricao": "Filtro de Mediana (remove ruído sal e pimenta)"
    }

    caminho_zip = criar_zip_resposta(img_original, img_filtrada, metadados, "mediana", nivel, formato.value)
    background_tasks.add_task(limpar_arquivo_temporario, caminho_zip)

    return FileResponse(
        caminho_zip,
        media_type="application/zip",
        filename=f"filtro_mediana_nivel{nivel}.zip"
    )


# ============================================================================
# ENDPOINT RAIZ
# ============================================================================

@app.get("/", tags=["Info"])
async def raiz():
    """Endpoint raiz com informações da API."""
    return {
        "mensagem": "API de Filtros de Imagem",
        "versao": "1.0.0",
        "documentacao": "/docs",
        "filtros_disponiveis": [
            "sobel", "roberts", "canny",
            "gaussiano", "bilateral", "media", "mediana"
        ],
        "niveis": [1, 2, 3],
        "endpoints_customizados": True
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
