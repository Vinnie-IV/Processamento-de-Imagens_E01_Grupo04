"""
Funções utilitárias para validação de parâmetros da API.
"""

from fastapi import HTTPException


def validar_kernel_impar(kernel_size: int, nome_parametro: str = "kernel") -> None:
    """
    Valida se o tamanho do kernel é ímpar e dentro do range válido.

    Args:
        kernel_size: Tamanho do kernel a validar
        nome_parametro: Nome do parâmetro para mensagem de erro

    Raises:
        HTTPException: Se o kernel for inválido
    """
    if kernel_size <= 0:
        raise HTTPException(
            status_code=400,
            detail=f"{nome_parametro} deve ser maior que zero (recebido: {kernel_size})"
        )

    if kernel_size % 2 == 0:
        raise HTTPException(
            status_code=400,
            detail=f"{nome_parametro} deve ser um número ímpar (recebido: {kernel_size})"
        )

    if kernel_size > 99:
        raise HTTPException(
            status_code=400,
            detail=f"{nome_parametro} muito grande. Máximo permitido: 99 (recebido: {kernel_size})"
        )


def validar_intervalo(valor: int, minimo: int, maximo: int, nome_parametro: str) -> None:
    """
    Valida se um valor está dentro de um intervalo.

    Args:
        valor: Valor a validar
        minimo: Valor mínimo permitido
        maximo: Valor máximo permitido
        nome_parametro: Nome do parâmetro para mensagem de erro

    Raises:
        HTTPException: Se o valor estiver fora do intervalo
    """
    if valor < minimo or valor > maximo:
        raise HTTPException(
            status_code=400,
            detail=f"{nome_parametro} deve estar entre {minimo} e {maximo} (recebido: {valor})"
        )


def validar_tamanho_abertura_canny(tamanho: int) -> None:
    """
    Valida tamanho de abertura para detector Canny (deve ser 3, 5 ou 7).

    Args:
        tamanho: Tamanho da abertura

    Raises:
        HTTPException: Se o tamanho for inválido
    """
    if tamanho not in [3, 5, 7]:
        raise HTTPException(
            status_code=400,
            detail=f"tamanho_abertura deve ser 3, 5 ou 7 (recebido: {tamanho})"
        )


def validar_ordem_limiares_canny(limiar1: int, limiar2: int) -> None:
    """
    Valida que limiar2 > limiar1 para melhor resultado no Canny.

    Args:
        limiar1: Primeiro limiar
        limiar2: Segundo limiar

    Raises:
        HTTPException: Se limiar2 <= limiar1
    """
    if limiar2 <= limiar1:
        raise HTTPException(
            status_code=400,
            detail=f"limiar2 ({limiar2}) deve ser maior que limiar1 ({limiar1})"
        )
