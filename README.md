# trabalho-dl — MLP no CIFAR-10

Classificação do CIFAR-10 (convertido para escala de cinza) com uma Rede Neural MLP, comparando 4 configurações, cada uma executada 3 vezes:

| Config | Dropout | Early Stop |
|---|---|---|
| A | Não | Não |
| B | Não | Sim |
| C | Sim | Não |
| D | Sim | Sim |

## Hiperparâmetros

Baseados em Srivastava et al. (2014), experimentos com MLP:

- 3 camadas ocultas × 1024 neurônios, ReLU; saída softmax com 10 classes
- Dropout: 0.2 na entrada, 0.5 nas camadas ocultas (+ restrição max-norm = 3)
- SGD com momentum 0.95
- Épocas / batch: ver a célula de configuração do notebook (**confirmar no Apêndice B do artigo**)
- Early stopping: `val_loss`, paciência 10, 10% do treino como validação

## Como rodar

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Para o TensorFlow achar a GPU (libs CUDA instaladas pelo pip), rode uma vez:
cat >> .venv/bin/activate <<'EOT'
LD_LIBRARY_PATH="$(ls -d "$VIRTUAL_ENV"/lib/python3*/site-packages/nvidia/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH
EOT
deactivate && source .venv/bin/activate

jupyter notebook mlp_cifar10.ipynb
```

A primeira célula imprime as GPUs detectadas; se aparecer `[]`, o treino roda na CPU (funciona, mas bem mais devagar).

1. Com `QUICK_TEST = True` (padrão), o notebook roda rápido num subconjunto dos dados, só para validar o pipeline (saída em `results_quick/`).
2. Troque para `QUICK_TEST = False` e rode tudo para os 12 treinos completos (saída em `results/`).

Se a execução for interrompida, basta rodar de novo: os runs já salvos em `results/resultados.csv` são pulados.

## Saídas (`results/`)

- `resultados.csv` — acurácia de treino e teste, épocas e tempo de cada run
- `resumo.csv` — média ± desvio por configuração e gap treino × teste
- `barras_treino_teste.png`, `curvas_treino.png`, `exemplos.png` — gráficos para a apresentação
- `histories/` — histórico de treino de cada run

## Referência

N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, R. Salakhutdinov. *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. Journal of Machine Learning Research, 15(56):1929–1958, 2014. https://jmlr.org/papers/v15/srivastava14a.html
