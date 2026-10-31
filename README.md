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

## Arquivos

- `metricas_volume_por_amostra.csv`: resultados por arquitetura e amostra, incluindo os três modos de agregação, seleção de classe e contagem de patches sem foreground.
- `resumo_volume_por_modelo.csv`: média, desvio-padrão amostral (`ddof=1`) e populacional (`ddof=0`) entre as 11 amostras, para cada modo e seleção de classe.
- `.gitignore`: contém a regra `*.keras` para manter os modelos Keras fora do Git.

Os valores usam predições já salvas na pasta original. Não incluem predições novas do kernel que estava rodando em `LOO_3d_unet_backbones2`.
