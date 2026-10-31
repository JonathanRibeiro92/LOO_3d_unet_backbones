# Comparação de agregações para Dice e IoU

O script compara três cálculos usando as mesmas predições TIFF e máscaras:

1. **Volume completo:** agrega todos os voxels do volume antes de calcular Dice e IoU.
2. **64 patches agrupados:** soma as contagens TP/FP/FN dos 64 patches antes de calcular as métricas. Como os patches cobrem o volume sem sobreposição, deve coincidir com o cálculo do volume.
3. **Média por patch:** calcula Dice e IoU separadamente em cada patch de `64³` voxels e tira a média aritmética das 64 pontuações.

Cada modo é calculado para **foreground apenas** e para a **média macro de background e foreground**. As predições salvas contêm rótulos discretos `0/1` obtidos por `argmax`; o notebook original aplica `>0,5` em cada canal softmax. Como as probabilidades não foram salvas, a média macro é uma aproximação do notebook. O caso InceptionV3 `22_08` evidencia a diferença: a aproximação dá IoU/Dice `0,7605/0,8460`, enquanto o notebook registra `0,7377/0,8341`.

O código usa `smooth=1e-5`, como a biblioteca do notebook. Patches sem foreground verdadeiro contam com pontuação 1 quando também não há foreground previsto. O CSV por amostra informa quantos desses patches existem.

## Executar no WSL/Jupyter

```bash
conda activate tf
cd /mnt/d/mestrado/notebooks/LOO_3d_unet_backbones
jupyter notebook
```

Abra `recalcular_volumes.ipynb` e execute a célula. O script lê as predições em `/mnt/d/mestrado/notebooks/LOO_3d_unet_backbones/` e as máscaras em `/mnt/d/mestrado/notebooks/steel/stl/`.

## Protocolo LOO CV e organização dos notebooks

O experimento usa **Leave-One-Out Cross-Validation (LOO CV)** com 11 amostras (`03_09`, `04_09`, `06_06`, `07_06`, `07_09`, `09_06`, `11_06`, `22_08`, `23_08`, `28_08` e `29_08`). Em cada rodada, uma amostra inteira é separada como teste e as outras 10 são reunidas para treinamento. O código de preparação exclui explicitamente a pasta da amostra de teste (`createTrainLOO`). Os patches do conjunto restante são divididos para treinamento e validação; por exemplo, os notebooks ResNet18 usam `train_test_split` com 10% para validação. Ao final da rodada, o modelo faz predições na amostra que ficou de fora. O processo é repetido até cada amostra ter sido o teste uma vez. Assim, cada arquitetura tem 11 notebooks de experimento.

Os dados são preparados em patches 3D de `64 × 64 × 64`. Nos notebooks originais, a avaliação de IoU e F-score recebe os patches de teste e as predições correspondentes; a reconstrução do volume é usada nas etapas de visualização/exportação. O script e os CSVs deste diretório comparam explicitamente métricas por patch, agregação das contagens dos 64 patches e cálculo sobre o volume completo, usando as predições TIFF já salvas.

### Notebooks de experimento

O nome do arquivo identifica a arquitetura e a amostra deixada de fora:

- `resnet18/3D_Unet_steel_resnet18_<amostra>.ipynb`: U-Net 3D com backbone ResNet18.
- `inceptionv3/3D_Unet_steel_inceptionv3_<amostra>.ipynb`: U-Net 3D com backbone InceptionV3.
- `resunet/Unet-Resblock-<amostra>.ipynb`: U-Net 3D com blocos residuais (ResUNet).

Substitua `<amostra>` por um dos 11 identificadores listados acima. Por exemplo, `3D_Unet_steel_resnet18_04_09.ipynb` treina e avalia a rodada em que `04_09` é a amostra de teste. Os notebooks de cada arquitetura seguem a mesma lógica LOO, com diferenças na arquitetura da rede e em alguns detalhes de treinamento.

### Notebooks auxiliares e de métricas

- `resnet18/ConverteResultadosparaInt8.ipynb`, `inceptionv3/ConverteResultadosparaInt8.ipynb` e `resunet/ConverteResultadosparaInt8.ipynb`: convertem/exportam os resultados de predição para TIFF de 8 bits usados pela reavaliação.
- `resnet18/printSlices.ipynb`, `inceptionv3/printSlices.ipynb` e `resunet/printSlices.ipynb`: exibem fatias dos volumes e resultados para inspeção visual.
- `recalcular_volumes.ipynb`: executa a reavaliação das métricas nos três modos de agregação documentados neste README; o cálculo fica em `recalcular_metricas_volume.py`.
## Arquivos

- `metricas_volume_por_amostra.csv`: resultados por arquitetura e amostra, incluindo os três modos de agregação, seleção de classe e contagem de patches sem foreground.
- `resumo_volume_por_modelo.csv`: média, desvio-padrão amostral (`ddof=1`) e populacional (`ddof=0`) entre as 11 amostras, para cada modo e seleção de classe.
- `.gitignore`: ignora modelos `*.keras`, TIFFs gerados em `3D_result` e checkpoints do Jupyter.

## Médias e desvios-padrão

No arquivo `resumo_volume_por_modelo.csv`, `mean` é a média dos 11 resultados por amostra. `std_sample_ddof1` é o desvio-padrão amostral: usa divisor `n-1` (10 para as 11 amostras) e estima a variação com base na amostra. `std_population_ddof0` é o desvio-padrão populacional: usa divisor `n` (11) e trata as 11 amostras como o conjunto completo considerado. Os campos `ddof=1` e `ddof=0` indicam qual divisor foi usado.

Os valores usam predições já salvas na pasta original. Não incluem predições novas do kernel que estava rodando em `LOO_3d_unet_backbones2`.

## Métricas utilizadas e implementação

As métricas são **IoU (Intersection over Union, ou Jaccard)** e **Dice/F-score**. Para cada classe, o código conta verdadeiros positivos (TP), falsos positivos (FP) e falsos negativos (FN), usando as máscaras discretas de predição e referência:

- **IoU:** `(TP + smooth) / (TP + FP + FN + smooth)`.
- **Dice/F-score (beta = 1):** `(2 × TP + smooth) / (2 × TP + FP + FN + smooth)`.

O valor `smooth = 1e-5` evita divisão por zero. As métricas são calculadas para foreground; os campos `macro` são a média aritmética das pontuações de background e foreground. Os três modos de agregação descritos acima usam essas mesmas fórmulas, mudando apenas quando as contagens são agregadas ou quando as pontuações são calculadas.

**Origem dos cálculos:** o notebook de treinamento/evaluação usa os objetos de métrica da biblioteca `segmentation_models_3D` (`IOUScore` e `FScore`, com `beta=1` para Dice). Já `recalcular_metricas_volume.py` **não importa nem chama essa biblioteca para calcular as métricas**: implementa diretamente as fórmulas acima em Python, contando TP/FP/FN com NumPy. A constante de suavização foi mantida em `1e-5` para corresponder à configuração da biblioteca.

A reavaliação usa rótulos discretos `0/1` dos TIFFs salvos. Portanto, ela não recupera o cálculo sobre as probabilidades softmax do notebook: o README registra que os arquivos de probabilidade não foram salvos e que, para a média macro, os resultados são uma aproximação baseada nos rótulos discretos. As médias e os desvios-padrão entre amostras também são calculados diretamente pelo script com NumPy (`mean`; desvio amostral com `ddof=1`; populacional com `ddof=0`).
## Significado das colunas dos CSVs

### `metricas_volume_por_amostra.csv`

- `model`: arquitetura (`resnet18`, `inceptionv3` ou `resunet`).
- `sample`: amostra reservada para teste em cada rodada LOO.
- `volume_foreground_iou`, `volume_foreground_dice`: IoU e Dice da classe foreground, calculados globalmente sobre todos os voxels do volume.
- `pooled64_foreground_iou`, `pooled64_foreground_dice`: métricas de foreground após somar TP, FP e FN dos 64 patches. Como os patches cobrem o volume sem sobreposição, coincidem com as métricas globais do volume.
- `mean64_foreground_iou`, `mean64_foreground_dice`: média aritmética das métricas de foreground calculadas separadamente em cada patch.
- `volume_macro_iou`, `volume_macro_dice`: média entre as pontuações de background e foreground, calculadas globalmente no volume.
- `pooled64_macro_iou`, `pooled64_macro_dice`: média entre as pontuações das duas classes após agregar as contagens dos 64 patches.
- `mean64_macro_iou`, `mean64_macro_dice`: média aritmética, entre os 64 patches, das médias de classe calculadas em cada patch.
- `patch_count`: número de patches da amostra (64).
- `empty_foreground_patches`: quantos patches não têm voxels foreground na máscara de referência.
- `voxels`: número total de voxels do volume.
- `volume_shape`: dimensões do volume, na ordem Z×Y×X.
- `prediction_rule`: regra usada para transformar a predição salva em rótulos `0/1`; as probabilidades softmax originais não foram salvas.
- `reference_rule`: regra usada para binarizar a máscara de referência (`voxel > 0` é foreground).
- `smooth`: constante de suavização usada nas métricas (`1e-5`).
- `prediction_file`, `reference_file`: caminhos dos TIFFs de predição e máscara de referência usados no cálculo.

### `resumo_volume_por_modelo.csv`

- `model`: arquitetura avaliada.
- `metric`: nome da métrica e do modo de agregação resumido; por exemplo, `volume_foreground_iou` ou `mean64_macro_dice`.
- `n_samples`: número de amostras incluídas no resumo (11).
- `mean`: média dos 11 valores dessa métrica.
- `std_sample_ddof1`: desvio-padrão amostral; divide por `n-1` (10) e estima a variação a partir das amostras.
- `std_population_ddof0`: desvio-padrão populacional; divide por `n` (11) e descreve a variação das 11 amostras consideradas como o conjunto completo.


