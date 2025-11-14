import cv2
import numpy as np
import base64
import json
import os
import zipfile
from io import BytesIO
from fastapi import UploadFile, HTTPException
from PIL import Image


async def processar_imagem_upload(arquivo: UploadFile) -> np.ndarray:
    """
    Processa o upload de imagem e converte para array NumPy.

    Args:
        arquivo: Arquivo de imagem enviado via upload

    Returns:
        Array NumPy da imagem em formato BGR

    Raises:
        HTTPException: Se o arquivo for inválido ou houver erro no processamento
    """
    # Validar que um arquivo foi enviado
    if not arquivo or not arquivo.filename:
        raise HTTPException(
            status_code=400,
            detail="Nenhum arquivo foi enviado"
        )

    # Validar tipo de arquivo
    extensoes_validas = ['.jpg', '.jpeg', '.png']
    extensao = os.path.splitext(arquivo.filename)[1].lower()

    if extensao not in extensoes_validas:
        raise HTTPException(
            status_code=400,
            detail=f"Formato de arquivo não suportado '{extensao}'. Formatos aceitos: {', '.join(extensoes_validas)}"
        )

    try:
        # Ler conteúdo do arquivo
        conteudo = await arquivo.read()

        # Validar que o arquivo não está vazio
        if len(conteudo) == 0:
            raise HTTPException(
                status_code=400,
                detail="O arquivo enviado está vazio"
            )

        # Validar tamanho (máximo 10MB)
        tamanho_mb = len(conteudo) / (1024 * 1024)
        if tamanho_mb > 10:
            raise HTTPException(
                status_code=413,  # Payload Too Large
                detail=f"Arquivo muito grande ({tamanho_mb:.2f}MB). Tamanho máximo: 10MB"
            )

        # Converter para array NumPy
        np_arr = np.frombuffer(conteudo, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(
                status_code=400,
                detail="Não foi possível decodificar a imagem. O arquivo pode estar corrompido ou não ser uma imagem válida."
            )

        # Validar dimensões mínimas
        altura, largura = img.shape[:2]
        if altura < 10 or largura < 10:
            raise HTTPException(
                status_code=400,
                detail=f"Imagem muito pequena ({largura}x{altura}). Dimensões mínimas: 10x10 pixels"
            )

        return img

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar upload da imagem: {str(e)}"
        )


def converter_para_cinza(img: np.ndarray) -> np.ndarray:
    """
    Converte imagem colorida para escala de cinza.

    Args:
        img: Imagem em formato BGR

    Returns:
        Imagem em escala de cinza
    """
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def imagem_para_base64(img: np.ndarray, formato: str = 'png') -> str:
    """
    Converte imagem NumPy para string base64.

    Args:
        img: Array NumPy da imagem
        formato: Formato de saída ('png', 'jpeg' ou 'jpg')

    Returns:
        String base64 da imagem com prefixo data URI

    Raises:
        HTTPException: Se houver erro na codificação da imagem
    """
    try:
        # Se for escala de cinza, converter para 3 canais para melhor compatibilidade
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Codificar imagem
        formato_lower = formato.lower()
        if formato_lower in ['jpg', 'jpeg']:
            sucesso, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            mime_type = 'image/jpeg'
        else:
            sucesso, buffer = cv2.imencode('.png', img)
            mime_type = 'image/png'

        if not sucesso:
            raise HTTPException(
                status_code=500,
                detail=f"Erro ao codificar imagem no formato {formato}"
            )

        # Converter para base64
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:{mime_type};base64,{img_base64}"

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao converter imagem para base64: {str(e)}"
        )


def criar_zip_resposta(
    img_original: np.ndarray,
    img_filtrada: np.ndarray,
    metadados: dict,
    nome_filtro: str,
    nivel: int = None,
    formato: str = 'png'
) -> str:
    """
    Cria arquivo ZIP contendo imagens e metadados.

    Args:
        img_original: Imagem original
        img_filtrada: Imagem com filtro aplicado
        metadados: Dicionário com informações (tempo_ms, filtro, etc)
        nome_filtro: Nome do filtro aplicado
        nivel: Nível do filtro (opcional)
        formato: Formato da imagem (png, jpeg, jpg). Padrão: 'png'

    Returns:
        Caminho do arquivo ZIP criado

    Raises:
        HTTPException: Se houver erro na criação do ZIP
    """
    try:
        # OpenCV's imencode requer '.jpg' não '.jpeg'
        formato_cv2 = 'jpg' if formato in ['jpg', 'jpeg'] else formato
        extensao = formato  # Mantém a escolha do usuário para extensão do arquivo

        # Criar nome do arquivo
        if nivel:
            nome_zip = f"filtro_{nome_filtro}_nivel{nivel}.zip"
        else:
            nome_zip = f"filtro_{nome_filtro}_customizado.zip"

        caminho_zip = os.path.join("temp", nome_zip)

        # Garantir que a pasta temp existe
        os.makedirs("temp", exist_ok=True)

        # Criar ZIP
        with zipfile.ZipFile(caminho_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Salvar imagem original
            sucesso, buffer_original = cv2.imencode(f'.{formato_cv2}', img_original)
            if not sucesso:
                raise Exception("Falha ao codificar imagem original")
            zipf.writestr(f'original.{extensao}', buffer_original.tobytes())

            # Salvar imagem filtrada
            sucesso, buffer_filtrada = cv2.imencode(f'.{formato_cv2}', img_filtrada)
            if not sucesso:
                raise Exception("Falha ao codificar imagem filtrada")
            zipf.writestr(f'filtrada.{extensao}', buffer_filtrada.tobytes())

            # Salvar metadados
            info_json = json.dumps(metadados, indent=2, ensure_ascii=False)
            zipf.writestr('info.json', info_json)

        return caminho_zip

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao criar arquivo ZIP: {str(e)}"
        )


def limpar_arquivo_temporario(caminho: str):
    """
    Remove arquivo temporário após envio.

    Args:
        caminho: Caminho do arquivo a ser removido
    """
    try:
        if os.path.exists(caminho):
            os.remove(caminho)
    except Exception as e:
        print(f"Aviso: Não foi possível remover arquivo temporário {caminho}: {e}")
