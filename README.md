# trabalho-dl — MLP no CIFAR-10

Classificação do CIFAR-10 (convertido para escala de cinza) com uma Rede Neural MLP, comparando 4 configurações, cada uma executada 3 vezes:

| Config | Dropout | Early Stop |
|---|---|---|
| A | Não | Não |
| B | Não | Sim |
| C | Sim | Não |
| D | Sim | Sim |

## Hiperparâmetros

Arquitetura e dropout de Srivastava et al. (2014), Tabela 2 e Apêndice B.1 (MLP do MNIST):

- 3 camadas ocultas × 1024 neurônios, ReLU; saída softmax com 10 classes
- Dropout: 0.2 na entrada, 0.5 nas camadas ocultas

Treinamento de Hinton et al. (2012), Apêndices A.1 (MNIST) e C (Reuters), igual nas 4 configurações:

- SGD com mini-batches de 100, por 500 épocas. O A.1 (MNIST) usa 3000; o Apêndice C (Reuters, MLP treinada do zero) aplica a mesma receita por 500 épocas. Numa execução de teste com 3000 épocas (A, seed 0), a acurácia de validação estabilizou por volta da época 200 (`results_v2_3000ep/`)
- Momentum sobe linearmente de 0.5 a 0.99 nas primeiras 500 épocas
- Taxa de aprendizado ε multiplicada por 0.998 a cada época; a atualização usa (1 − momentum)·ε
- Pesos iniciais ~ N(0, 0.01); com dropout, norma ao quadrado dos pesos de entrada de cada neurônio ≤ 15 (max-norm √15)
- **Única adaptação:** ε inicial 0.1 em vez de 10. No CIFAR-10 em cinza, ε = 10 e ε = 1 divergem (10% e 16% de acurácia de validação após 30 épocas, com dropout); com 0.1 a validação chega a 41% em 30 épocas

Validação e early stopping:

- Early stopping: `val_loss`, paciência 50, restaura os melhores pesos; 10% do treino como validação. Na primeira rodada, com paciência 10, D parou entre as épocas 67 e 115 ainda aprendendo
- O treino é reembaralhado a cada época via `tf.data`: com `keras.utils.set_random_seed` e arrays NumPy, o `fit` repetiria a mesma ordem de batches em todas as épocas
- Acurácia de treino medida nos 45 mil exemplos de treino (sem a validação)

A primeira rodada (lr 0.01 fixo, momentum 0.95, max-norm 2, 100/300 épocas, paciência 10) está em `results_v1_lr001/`.

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

- `resultados.csv` — acurácia de treino, validação e teste, épocas e tempo de cada run
- `metricas_por_classe.csv`, `matriz_confusao.png` — precisão, recall e F1 por classe da melhor configuração
- `preds/` — previsões no teste de cada run
- `resumo.csv` — média ± desvio por configuração e gap treino × teste
- `barras_treino_teste.png`, `curvas_treino.png`, `exemplos.png` — gráficos para a apresentação
- `histories/` — histórico de treino de cada run

## Referências

N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, R. Salakhutdinov. *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. Journal of Machine Learning Research, 15(56):1929–1958, 2014. https://jmlr.org/papers/v15/srivastava14a.html

G. Hinton, N. Srivastava, A. Krizhevsky, I. Sutskever, R. Salakhutdinov. *Improving neural networks by preventing co-adaptation of feature detectors*. arXiv:1207.0580, 2012. https://arxiv.org/abs/1207.0580
