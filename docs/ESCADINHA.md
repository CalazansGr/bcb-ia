# A escadinha dos 9 algoritmos

Cada degrau é uma **pequena mudança** do anterior. Todos tentam resolver o mesmo problema neste projeto:
a partir do histórico recente do dólar (retornos de 1 a 10 dias, médias móveis, volatilidade, Selic),
prever o que acontece no próximo dia útil. O 7 (k-means) é a exceção: ele não prevê, agrupa.

Código: `src/modelos.py` (1 a 8) e `minillm/` (9). Rode `python run.py avaliar` para ver todos lado a lado.

---

## 1. Regressão linear — "a linha reta mais próxima"
Encontra a reta que passa o mais perto possível dos pontos. Aqui, combina as features com pesos para
chegar a um número (o retorno de amanhã). É o modelo mais simples e o primeiro a testar.
- **Bom para:** relações simples e fáceis de explicar.
- **Limite:** só enxerga linhas retas.

## 2. Regressão logística — "dobre a linha em um S"
Pega a mesma ideia da reta e a dobra em formato de "S" para o resultado ficar entre 0 e 1, como uma
probabilidade. Responde **sim ou não**: "o dólar sobe amanhã?".
- **Bom para:** classificar em duas categorias.
- **Resultado aqui:** 51,6% de acerto de direção, perto do acaso.

## 3. Árvore de decisão — "faça perguntas de sim ou não"
Aprende uma sequência de perguntas ("a volatilidade está acima de X?") que separa os casos.
- **Bom para:** capturar regras e interações, e dá para desenhar e ler.
- **Limite:** sozinha, decora os dados de treino com facilidade.

## 4. Floresta aleatória — "cem árvores votam"
Treina centenas de árvores, cada uma com uma amostra diferente dos dados, e usa a média dos votos.
O erro de cada árvore tende a se cancelar.
- **Bom para:** ser robusta sem muito ajuste.
- **Resultado aqui:** 50,8% de acerto de direção e erro igual ao da referência "amanhã = hoje". Nenhum
  algoritmo desta lista se destacou na previsão (veja o README).

## 5. Gradient boosting — "árvores pequenas consertam os erros que restam"
Em vez de várias árvores independentes, treina uma de cada vez, e cada nova árvore tenta corrigir os
erros das anteriores.
- **Bom para:** dados tabulares, costuma ganhar competições.
- **Limite:** pode ajustar demais o ruído se ficar grande.

## 6. k vizinhos (k-NN) — "os 5 pontos mais próximos votam"
Para prever um dia novo, procura os 5 dias do passado mais parecidos e tira a média do que aconteceu.
Não aprende nada antes: só guarda os exemplos.
- **Bom para:** ideia simples e sem suposições.
- **Limite:** sofre com muitas features e com ruído.

## 7. K-means — "três centros encontram os grupos, sem respostas dadas"
Único algoritmo **sem rótulos** da lista: ninguém diz qual dia é de que tipo. Ele agrupa os dias
parecidos em k grupos. Aqui, agrupa por volatilidade e tendência e encontra três regimes:

| Regime | Dias | Volatilidade diária média |
|---|---|---|
| calmo | 3.594 | 0,65% |
| intermediário | 2.195 | 0,93% |
| agitado | 932 | 1,27% |

Comando: `python run.py regimes`.

## 8. Rede neural — "curvas em S empilhadas em camadas"
Cada neurônio faz uma conta parecida com a regressão logística; empilhar camadas deles permite
aprender formas complicadas, não só retas. Aqui: duas camadas escondidas (32 e 16 neurônios).
- **Bom para:** padrões complexos, com muitos dados.
- **Limite:** pede dados padronizados (veja o `TransformedTargetRegressor` no código) e muitos exemplos.

## 9. LLM — "torne-a enorme e treine com texto"
Uma rede neural, do tipo **transformer**, treinada para uma única tarefa: **prever o próximo caractere
(ou palavra)**. Para gerar um texto, ela prevê, sorteia, anexa o resultado e repete.
Por dentro é, em grande parte, as mesmas contas lineares e curvas dos degraus anteriores, em escala grande.

O que acontece em `minillm/`:

1. **Corpus** (`corpus.py`): cada dia vira um documento de duas partes, os dados e o texto.
   ```
   <dados> data=19/11/2026 dolar=5.8889 dia=+1.35% ... </dados>
   <texto> No pregão de 19/11/2026, o dólar disparou 1,35% e fechou a R$ 5,8889. ... </texto>
   ```
2. **Modelo** (`modelo.py`): embedding de caracteres e de posição, 4 blocos de
   *atenção causal + MLP*, e uma camada final que dá a probabilidade de cada próximo caractere.
   A atenção deixa cada posição olhar para trás no texto (é como o modelo "copia" o número certo).
3. **Treino** (`treino.py`): 2500 passos, AdamW, taxa de aprendizado com aquecimento e queda suave,
   precisão bfloat16 na GPU. Validação nos dias mais recentes, que ficam de fora do treino.
   Todos os números vêm de dados reais do Banco Central; nada é sorteado nem inventado.
4. **Uso**: o modelo recebe só a parte `<dados>` do dia mais recente (números reais do Banco Central)
   e continua escrevendo o `<texto>`. Os números são conferidos automaticamente contra os dados.

**Por que um LLM de verdade (como o Claude) é tão maior:** o mesmo mecanismo, mas com bilhões de
parâmetros e trilhões de palavras de texto. Aqui são 3,3 milhões de parâmetros e um corpus gerado por
regras, por isso o resultado é bem estreito: ele só sabe escrever esse tipo de boletim.
