# API de Filtros de Imagem

API REST desenvolvida com FastAPI para aplicar filtros de processamento de imagem com diferentes níveis de intensidade.

## Funcionalidades

- 7 tipos de filtros de imagem
- 3 níveis de intensidade por filtro (baixo, normal, forte)
- Endpoints customizados com parâmetros personalizados
- Retorno em formato JSON (base64) ou ZIP (download de arquivos)
- Logging de tempo de processamento
- Documentação interativa (Swagger)

## Filtros Disponíveis

### Detecção de Bordas

- **Sobel**: Detecta bordas usando operador Sobel
- **Roberts**: Detecta bordas usando operador Roberts
- **Canny**: Detector de bordas Canny com controle de limiares

### Filtros de Blur

- **Gaussiano**: Blur gaussiano com controle de kernel
- **Bilateral**: Filtro bilateral que preserva bordas
- **Média**: Filtro de média (blur uniforme)
- **Mediana**: Filtro de mediana (remove ruído sal e pimenta)

## Instalação

### Pré-requisitos

- Python 3.12.5
- Poetry (gerenciador de dependências)

### Instalando o Poetry

Instale o Poetry usando pip:

```bash
pip install poetry
```

### Passos para Configurar o Projeto

1. Clone o repositório ou navegue até a pasta do projeto:

```bash
cd processamento_imagem/src
```

2. Instale as dependências (Poetry criará um ambiente virtual isolado automaticamente):

```bash
poetry install
```

3. Execute a API em modo desenvolvimento:

```bash
poetry run poe dev
```

Este comando iniciará o servidor com hot-reload (recarrega automaticamente quando o código muda).

**Alternativas:**

- **Executar diretamente**: `poetry run python principal.py`
- **Ativar ambiente virtual**: `poetry shell` e depois `python principal.py`

4. Acesse a documentação interativa:

```
http://localhost:8000/docs
```

## Estrutura do Projeto

```
src/
├── principal.py              # API FastAPI principal
├── filtros/
│   ├── __init__.py
│   ├── deteccao_bordas.py    # Sobel, Roberts, Canny
│   ├── filtros_blur.py       # Gaussiano, Bilateral, Média, Mediana
│   └── utilitarios.py        # Funções auxiliares
├── modelos/
│   ├── __init__.py
│   └── esquemas.py           # Modelos Pydantic
├── temp/                     # Arquivos temporários (ZIP)
├── pyproject.toml            # Configuração Poetry e dependências
├── poetry.lock               # Lock file (gerado automaticamente)
├── .gitignore                # Arquivos ignorados pelo Git
└── README.md
```

## Uso da API

### Endpoints com Níveis Pré-definidos

Todos os filtros têm dois tipos de endpoints:

**1. Endpoint JSON** (retorna base64):

```
POST /filtros/{nome_filtro}/{nivel}
```

**2. Endpoint Download** (retorna ZIP):

```
POST /filtros/{nome_filtro}/{nivel}/download
```

Onde:

- `{nome_filtro}`: sobel, roberts, canny, gaussiano, bilateral, media, mediana
- `{nivel}`: 1 (baixo), 2 (normal), 3 (forte)

### Exemplos de Uso

#### Exemplo 1: Filtro Gaussiano Nível 2 (JSON)

```bash
curl -X POST "http://localhost:8000/filtros/gaussiano/2" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "arquivo=@imagem.jpg"
```

**Resposta:**

```json
{
  "imagem_original": "data:image/png;base64,iVBORw0KGgo...",
  "imagem_filtrada": "data:image/png;base64,iVBORw0KGgo...",
  "tempo_ms": 123.45,
  "filtro": "gaussiano",
  "nivel": 2,
  "parametros": {
    "kernel_width": 15,
    "kernel_height": 15,
    "sigma": 0
  }
}
```

#### Exemplo 2: Filtro Canny Nível 3 (Download ZIP)

```bash
curl -X POST "http://localhost:8000/filtros/canny/3/download" \
  -H "Content-Type: multipart/form-data" \
  -F "arquivo=@imagem.jpg" \
  --output resultado.zip
```

O arquivo ZIP conterá:

- `original.png` - Imagem original
- `filtrada.png` - Imagem com filtro aplicado
- `info.json` - Metadados (tempo, parâmetros, etc)

#### Exemplo 3: Python com requests

```python
import requests

# Endpoint JSON
with open('imagem.jpg', 'rb') as f:
    files = {'arquivo': f}
    response = requests.post(
        'http://localhost:8000/filtros/bilateral/2',
        files=files
    )
    resultado = response.json()
    print(f"Tempo de processamento: {resultado['tempo_ms']:.2f} ms")

# Endpoint Download
with open('imagem.jpg', 'rb') as f:
    files = {'arquivo': f}
    response = requests.post(
        'http://localhost:8000/filtros/sobel/1/download',
        files=files
    )
    with open('resultado.zip', 'wb') as out:
        out.write(response.content)
```

### Endpoints Customizados

Para controle total dos parâmetros, use os endpoints customizados:

#### Gaussiano Customizado

```bash
curl -X POST "http://localhost:8000/filtros/gaussiano/customizado" \
  -H "Content-Type: multipart/form-data" \
  -F "arquivo=@imagem.jpg" \
  -F "parametros={\"kernel_width\":25,\"kernel_height\":25,\"sigma\":0}"
```

#### Canny Customizado

```bash
curl -X POST "http://localhost:8000/filtros/canny/customizado" \
  -H "Content-Type: multipart/form-data" \
  -F "arquivo=@imagem.jpg" \
  -F "parametros={\"limiar1\":80,\"limiar2\":180,\"tamanho_abertura\":3,\"aplicar_blur\":true}"
```

#### Bilateral Customizado

```bash
curl -X POST "http://localhost:8000/filtros/bilateral/customizado" \
  -H "Content-Type: multipart/form-data" \
  -F "arquivo=@imagem.jpg" \
  -F "parametros={\"d\":20,\"sigma_cor\":100,\"sigma_espaco\":100}"
```

## Parâmetros por Nível

### Gaussiano

- Nível 1: kernel (5x5)
- Nível 2: kernel (15x15)
- Nível 3: kernel (35x35)

### Bilateral

- Nível 1: d=9, sigma_cor=25, sigma_espaco=25
- Nível 2: d=15, sigma_cor=50, sigma_espaco=50
- Nível 3: d=25, sigma_cor=75, sigma_espaco=75

### Média

- Nível 1: kernel (3x3)
- Nível 2: kernel (7x7)
- Nível 3: kernel (15x15)

### Mediana

- Nível 1: tamanho=3
- Nível 2: tamanho=7
- Nível 3: tamanho=15

### Canny

- Nível 1: limiar1=50, limiar2=150
- Nível 2: limiar1=100, limiar2=200
- Nível 3: limiar1=150, limiar2=250

## Validações

- Formatos aceitos: JPG, JPEG, PNG
- Tamanho máximo: 10MB
- Imagens coloridas são automaticamente convertidas para escala de cinza quando necessário

## Logs

A API registra o tempo de processamento de cada filtro:

```
INFO: Filtro Gaussiano (nível 2) gerado em 45.67 ms
INFO: Filtro Canny (nível 3) gerado em 89.12 ms
INFO: Filtro Bilateral customizado gerado em 234.56 ms
```

## Documentação Interativa

Acesse `http://localhost:8000/docs` para a documentação Swagger completa, onde você pode:

- Testar todos os endpoints diretamente no navegador
- Ver exemplos de requisições e respostas
- Explorar os modelos de dados

## Tecnologias Utilizadas

- **FastAPI**: Framework web moderno e rápido
- **OpenCV**: Processamento de imagem
- **scikit-image**: Algoritmos de visão computacional
- **NumPy**: Computação numérica
- **Pillow**: Manipulação de imagens
- **Pydantic**: Validação de dados

## Licença

Este projeto foi desenvolvido para fins educacionais.
