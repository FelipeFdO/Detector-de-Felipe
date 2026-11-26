# Detector de Felipe

Identificador facial em tempo real para reconhecer **o Felipe** pela webcam usando visão computacional leve na CPU (sem necessidade de GPU dedicada).

## Como funciona

1. **Detecção de rostos (YuNet):** localiza os rostos no frame da câmera ou em uma imagem.
2. **Extração de características (SFace):** gera um embedding facial de 128 dimensões para cada rosto alinhado.
3. **Classificação (`LogisticRegression` + centróide):** calcula a probabilidade e a similaridade cosseno para decidir se é **`FELIPE`** ou **`NAO E O FELIPE`**.

## Como usar no Windows

Dê um duplo clique em **`run_camera.bat`**.

O script faz tudo automaticamente:
- Verifica se o Python 3.10+ está instalado (e instala o Python 3.12 automaticamente se precisar).
- Cria a `.venv` e instala as dependências de `requirements.txt`.
- Baixa os modelos ONNX (YuNet e SFace) se estiverem faltando.
- Abre a webcam em tempo real (`Q` ou `ESC` para sair).

### Comandos manuais (opcional)

```powershell
# Treinar / retreinar com suas próprias fotos
.\.venv\Scripts\python.exe train.py

# Abrir a webcam
.\.venv\Scripts\python.exe webcam.py

# Testar em uma imagem específica
.\.venv\Scripts\python.exe webcam.py --image "caminho\da\foto.jpg"
```

> **Nota sobre os dados de treino:** a pasta `Dados/` não é enviada ao GitHub por privacidade e tamanho (~1,8 GB). Para retreinar em outro computador, crie `Dados/Felipe` e `Dados/Não Felipe` com suas imagens e rode `train.py`.
