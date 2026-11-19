# AIRFRAME

Experiência visual controlada pelas mãos, com vários **modos de interação**:

| Modo | O que faz |
|---|---|
| **FRAMES** | Desenhe retângulos no ar com duas pinças; mova, redimensione, gire e aplique 18 efeitos só dentro deles |
| **CUTOUT** | Recorte um pedaço da imagem ao vivo com duas pinças: ele congela como adesivo que você move, gira, escala e clona |
| **WEB** | Encoste pontas de dedos e afaste: fios elásticos com física; puxe devagar para esticar, dê um tranco para arrebentar |
| **COUNT** | Calculadora de dedos: cada mão mostra 0–5, a conta aparece ao vivo; 👌 troca a operação; segurar parado salva na fita (Σ) |
| **DRAW** | Pinte com luz neon no ar (pinça = pincel, mão aberta = borracha, swipe = cor) |
| **PHYSICS** | Bolas neon: segure a pinça para inflar, agarre e arremesse, rebata com a mão aberta |

ENERGY e PARTICLES também existem, mas ficam fora da rotação por padrão (habilite em `[ui] modes` no `airframe.toml`).
Troca de modo: **segure o sinal de V (✌) por ~1 s**, `Tab`/`Shift+Tab` ou clique nas abas do topo.
Todo o processamento é local (webcam + MediaPipe na CPU).

## Instalação e execução (Fedora)

```bash
sudo dnf install python3.12 curl          # ou use o uv (recomendado: ele baixa o Python sozinho)
./run.sh                                   # cria .venv, instala dependências, baixa o modelo e abre o app
```

Instalação manual:

```bash
uv venv --python 3.12 .venv               # ou: python3.12 -m venv .venv
uv pip install --python .venv -r requirements.txt
./scripts/download_model.sh               # models/hand_landmarker.task (~7.5 MB)
.venv/bin/python -m airframe
```

Opções: `--fullscreen`, `--camera 1` (ou caminho de um vídeo), `--resolution 640x480`,
`--fps 30`, `--no-mirror`, `--reset-calibration`, `--config outro.toml`.

**Câmera não abre?** Feche outros apps que usam a webcam e verifique `ls /dev/video*` e se o
usuário está no grupo `video` (`sudo usermod -aG video $USER`, depois relogue).
O Fedora atual (Python 3.15) é novo demais para o MediaPipe; por isso o app usa um venv com Python 3.12.

## Gestos — modo FRAMES

| Gesto | Ação |
|---|---|
| **Pinça** (encostar polegar e indicador) | Selecionar / agarrar. O anel no dedo fecha conforme você se aproxima do disparo |
| Pinça dentro de um frame + mover | Arrastar |
| **Toque rápido de pinça num frame** e depois **abrir / fechar polegar e indicador** | Zoom estilo celular: abrir aumenta, fechar diminui. Mão parada ~1 s (ou pinça da outra mão) trava o tamanho. Roda do mouse sobre o frame faz o mesmo |
| Pinça num canto do frame selecionado | Redimensionar por aquele canto (o canto oposto fica fixo) |
| **Duas pinças em área vazia, afastando** | Desenhar um novo frame entre os dois pontos |
| Duas pinças no mesmo frame | Escalar (afastar = aumenta, aproximar = diminui) e girar |
| Arrastar até o ⊗ (centro inferior) e soltar | Excluir o frame |
| Pinça rápida em área vazia | Tirar a seleção |
| Mão aberta | Libera qualquer interação daquela mão |
| **Mão aberta + swipe rápido ← →** | Efeito anterior / próximo (no frame selecionado, ou nos próximos frames) |

O swipe só dispara se os 5 dedos estiverem esticados e a mão percorrer ~28% da largura da tela
em menos de 0,45 s, com pouco movimento vertical. Depois de cada swipe há um intervalo de 0,9 s.

## Gestos — outros modos

| Modo | Gestos |
|---|---|
| CUTOUT | Duas pinças em área vazia + afastar + soltar → contagem "HANDS AWAY" → recorte congelado · pinça = mover · duas pinças nele = escalar/girar · canto = redimensionar · ⊗ = apagar · `D` clonar · `B` borda (adesivo/neon/nenhuma) · `Del` apagar · `Shift+Del` todos |
| WEB | Encostar pontas de dedos (entre as mãos, ou polegar com outro dedo) e afastar = fio · puxar devagar estica · puxão rápido arrebenta · dedos livres empurram os fios · `G` gravidade · `Del` corta tudo |
| COUNT | Mão esquerda da tela = 1º número, direita = 2º (dedos levantados, 0–5) · 👌 (polegar+indicador encostados, outros 3 dedos esticados) = próxima operação + − × ÷ · segurar a conta parada 1,5 s = salvar na fita · `+ - * /` ou `← →` operação · `Z` desfazer · `Del` limpar fita. Aqui o V é o número 2, então a troca de modo é por `Tab` |
| DRAW | Pinça + mover = traço (cada mão desenha o seu) · mão aberta sobre o traço = apaga · swipe = cor · `[` `]` espessura · `Z` desfazer · `Del` limpar |
| PARTICLES | Mover as mãos = rastro · pinça = buraco negro (quanto mais tempo, mais forte) · soltar = explosão · mão aberta = repulsão |
| ENERGY | Duas mãos = esfera (tamanho segue a distância entre elas) · pinça nas duas = carregar · soltar = onda de choque · uma mão = arco elétrico entre polegar e indicador |
| PHYSICS | Segurar pinça em área vazia = inflar bola · pinça numa bola = agarrar, soltar em movimento = arremessar · mão aberta = raquete · `G` gravidade |

O mouse funciona como uma pinça em todos os modos (clique e arraste).

## Teclado e mouse

| Tecla | Ação |
|---|---|
| `Tab` / `Shift+Tab` | Próximo / anterior modo |
| `P` | Screenshot em `~/Pictures/AIRFRAME` |
| `T` | Tema NEON ↔ RETRO (interface de portátil anos 2000: fonte pixel própria, painéis claros; a câmera não muda). Padrão em `[ui] theme` |
| `1`–`9`, `0`, `←` `→` | Escolher efeito (FRAMES) · `←` `→` = cor no DRAW |
| `E` | Efeitos on/off |
| `C` | Painel de calibração · `A` calibração automática |
| `S` | Esqueleto das mãos |
| `F` / `F11` | Tela cheia |
| `Del` · `Shift+Del` | Apagar o frame selecionado · apagar todos |
| `H` | Ajuda |
| `Q` / `Esc` | Sair |
| Mouse | Arrastar em área vazia cria frames, arrastar um frame move. Botão direito ou roda troca o efeito |

## Precisão da pinça e calibração

A pinça é a razão **distância polegar↔indicador / tamanho da palma**, que não depende da distância até a câmera.
O tamanho da palma usa o maior valor entre comprimento e largura, então a medida continua estável quando a mão gira.
Proteções contra disparos acidentais:

* **Histerese**: engata abaixo de `pinch_on` (0,22) e só solta acima de `pinch_off` (0,36).
* **Debounce**: são precisos 3 frames seguidos para engatar e 3 para soltar.
* **Aquecimento**: uma mão recém-detectada não pode fazer pinça nos 5 primeiros frames.
* **Ponto congelado ao soltar**: enquanto os dedos se separam, o frame não é empurrado.
* **Suavização One Euro** na posição, e o retângulo ainda desliza suavemente até o alvo.

**Calibração automática (`A`)**: segure a mão aberta por 2,5 s e depois faça a pinça por 2,5 s.
Os limites são colocados entre as duas medidas da *sua* mão e salvos em
`~/.config/airframe/calibration.json`. No painel `C` também dá para ajustar tudo à mão,
com um medidor ao vivo da razão de cada mão.

## Efeitos

Neon Frame · RGB Split · Pixelation · Glitch · Negative · Thermal · Scanlines · Edge Detection ·
Kaleidoscope · ASCII · Halftone · Pop Art · Frosted Glass · Night Vision · Pencil Sketch ·
Hologram · Liquid Ripple · VHS

Cada um leva de 2 a 12 ms numa região de 640×400 (há teste que garante menos de 25 ms).
Para criar um novo efeito, herde de `airframe.effects.Effect`, implemente `apply(roi, ctx)`
(e opcionalmente `decorate`) e registre em `default_registry()`.

## Arquitetura

```
airframe/
  camera/       captura em thread própria, sempre o frame mais recente, reconexão automática
  tracking/     MediaPipe HandLandmarker (modo VIDEO) em thread própria + IDs estáveis por mão
  gestures/     métricas, filtro One Euro, pinça com histerese, swipe, calibração automática
  interaction/  geometria de retângulos rotacionados, FrameManager, máquina de estados por mão
  modes/        DRAW, PARTICLES, ENERGY, PHYSICS (interface Experience + Pointer unificado mão/mouse)
  effects/      18 efeitos + compositor que aplica só dentro do polígono (máscara se rotacionado)
  rendering/    viewport câmera↔tela, composição (OpenCV), camada de glow aditiva, overlay vetorial (pygame)
  ui/           HUD, barra de efeitos, calibração, splash/erro/onboarding/ajuda
  config/       dataclasses + airframe.toml + calibração do usuário
  app.py        loop principal
```

Fluxo: `camera → tracking → gestures → interaction → effects → rendering/ui`.
O loop de renderização nunca espera a inferência: ele usa o resultado de tracking mais recente.
Na máquina de desenvolvimento a câmera e o tracking rodam a 30 FPS (~12–18 ms de inferência).

Configuração: veja `airframe.toml` (resolução, confiança do detector, limites dos gestos,
suavização, tamanho mínimo dos frames, FPS etc.).

## Testes

```bash
.venv/bin/python -m pytest
```

São 255 testes sem webcam: métricas de landmarks, pinça (histerese, debounce, aquecimento),
suavização, swipe, associação de mãos, calibração automática, geometria (rotação, escala, snap,
redimensionar pelo canto), criação, seleção, arraste, transformação com duas mãos, lixeira,
todos os efeitos (forma, região, desempenho), os modos (traços e borracha, partículas,
carga e disparo da esfera, física de bolas, gesto V, recorte, teia, contagem de dedos 0–5,
sinal de OK e a calculadora) e testes ponta a ponta do app com câmera e
mãos sintéticas rodando em SDL headless.
