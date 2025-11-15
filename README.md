# Trabalho de Redes Neurais - Classificação de Doenças Cardíacas

Este repositório contém o projeto **"Classificação de Doenças Cardíacas"**, desenvolvido como parte das atividades avaliativas da disciplina de Fundamentos de Inteligência Artificial no Instituto de Computação, da Universidade Federal do Amazonas (IComp/UFAM).

## Equipe

| Nome | E-mail |
|------|---------|
| Gabriel Conceição dos Santos | gabriel.conceicao@icomp.ufam.edu.br |
| Giovanna Bembom da Silva Bandeira | giovanna.bembom@icomp.ufam.edu.br |
| Luiggy Augusto Lima Alves | luiggy.alves@icomp.ufam.edu.br |
| Maria Flach da Costa | maria.flach@icomp.ufam.edu.br |
| Mariana Ramos Andre Simoes | mariana.simoes@icomp.ufam.edu.br |

# 1. Descrição do projeto: Classificação de Doenças Cardíacas com Redes Neurais
O objetivo deste trabalho é construir um **classificador binário**, utilizando uma **Rede Neural Artificial (RNA)**, para prever a presença (1) ou ausência (0) de doença cardíaca em pacientes com base em atributos clínicos.

O dataset utilizado é o **Heart Disease UCI**, amplamente empregado em estudos de Machine Learning e disponível no Kaggle. O conjunto contém 14 atributos, incluindo idade, pressão arterial, colesterol, tipo de dor no peito e outros indicadores relevantes para a análise.

# Importância do Problema
A detecção precoce de doenças cardíacas é um desafio crítico na área da saúde, uma vez que essas enfermidades estão entre as principais causas de mortalidade no mundo. Identificar pacientes com risco elevado de forma rápida e precisa pode aumentar significativamente as chances de tratamento eficaz e salvar vidas. Por isso, o desenvolvimento de modelos computacionais capazes de auxiliar nesse diagnóstico tem grande relevância médica e social.

Além disso, o processo de avaliação clínica envolve múltiplos fatores, como idade, colesterol, pressão arterial, histórico familiar e resultados de exames específicos. A interpretação desses atributos pode ser complexa e sujeita a erros ou variações entre diferentes profissionais. Nesse contexto, modelos de Inteligência Artificial se destacam pela capacidade de reconhecer padrões complexos e detectar relações não triviais entre os dados, oferecendo uma análise mais consistente e precisa.

Ferramentas de predição baseadas em IA não têm o objetivo de substituir o médico, mas de fornecer suporte à tomada de decisão, ajudando a identificar pacientes de alto risco, priorizar atendimentos e sugerir exames adicionais quando necessário. Essa integração entre conhecimento clínico e tecnologia melhora a eficiência dos sistemas de saúde e contribui para o uso mais racional dos recursos médicos disponíveis.

# Fluxo de Atividades do Notebook

## 1. **Inicialização/Carregamento de Dados:**

É o primeiro bloco, responsável por estabelecer as bases e ferramentas essenciais para o pré-processamento, análise e modelagem da rede. As bibliotecas importadas foram `sys`, `pandas`, `numpy`, `sklearn`, `matplotlib`, `seaborn` e `keras`. Portanto, em geral, esse bloco configura o *pipeline* do projeto.

## 2. **Limpeza:** 

Os dados são carregados via API do Kaggle e limpos (remoção de 723 duplicadas).
O processo de carregamento utiliza o `pandas` para ler o arquivo `heart.csv`. Inicialmente, o dataset contém valores ausentes codificados de forma não padrão. O problema é solucionado por meio da substituição e remoção de linhas que continham esses valores inválidos. 
Além disso, o código realiza a conversão de tipos de dados, para garantir que todas as colunas estejam em formato **numérico**. 

Logo, em resumo, este trecho:
* remove linhas com valores ausentes,
* converte colunas para tipo numérico,
* verifica estatísticas iniciais,
* remove registros duplicados, garantindo dados limpos.

## 3. **Detalhamento da Análise Exploratória (EDA):**

O Detalhamento da Análise Exploratória (EDA) é a fase visual estatística em que ocorre a verificação dos dados, balanceamento de classe e análise de correlação. Esse detalhamento inclui histogramas, gráficos de barras, mapa de calor e relação idade × frequência cardíaca. Esses gráficos ajudam a entender a distribuição dos atributos e possíveis correlações.

## 4. **Criação e Divisão dos Dados de Treinamento:**

É a etapa responsável por estruturar, dividir e escalonar as variáveis, deixando o `dataset` no formato ideal para o treinamento.

Primeiramente, é realizada a separação formal das variavéis **X** e **y**. A coluna `target` é separada para compor a variável **y** (o rótulo a ser previsto), enquanto todas as demais colunas do DataFrame, que representam os atributos clínicos, são reunidas na matriz **X** (os preditores). Em seguida, o código converte **X** e **y** diretamente para arrays NumPy, garantindo melhor uso de memória e maior velocidade de processamento.

Em seguida, a próxima etapa consiste na divisão do dataset em dados de treinamento `(X_train, y_train)` e dados de teste `(X_test, y_test)`. Utiliza-se o padrão de 80% para treino e 20% para teste. Essa separação é essencial para garantir que o modelo aprenda com uma parte dos dados e seja avaliado em outra parte nunca vista, permitindo medir seu desempenho de forma realmente imparcial.

O parâmetro `stratify=y` desempenha um papel fundamental nesse processo. Ele garante que a proporção das classes (pacientes com e sem doença cardíaca) seja preservada tanto no conjunto de treinamento quanto no de teste. Sem essa estratificação, o conjunto de teste poderia, por mero acaso, ficar desequilibrado, comprometendo a avaliação do modelo e produzindo métricas enganosas ou enviesadas.

A última e mais sofisticada etapa do pré-processamento é a padronização das variáveis. Esse procedimento transforma cada característica para que apresente **média próxima de zero e desvio padrão próximo de um**. Essa normalização é especialmente importante para algoritmos que dependem de medidas de distância e para modelos baseados em otimização por gradiente (como Redes Neurais), evitando que *features* naturalmente maiores tenham influência desproporcional no processo de aprendizado.

Para evitar vazamento de dados (*data leakage*), o *scaling* deve ocorrer em duas fases independentes: ajuste e transformação no treino e transformação no teste.

## 5. Treinamento da rede neural

A preparação dos dados começa com a criação de cópias de `y_train` e `y_test`, preservando os valores originais.

Em relação à estrutura da rede, o modelo é uma *Feed-Forward Neural Network* (FFNN) sequencial composta por:
* Entrada: 13 atributos clínicos.
* Camadas ocultas: 16 neurônios (ReLU) + inicialização normal + L2 = 0.001 + dropout 25%; 8 neurônios (ReLU) + mesmas regularizações; 4 neurônios (ReLU).
* Saída: 1 neurônio com sigmoid, retornando a probabilidade de presença de doença.

A arquitetura decrescente (16 → 8 → 4) funciona como um `encoder`, comprimindo informações de forma progressivamente mais abstrata.

Como técnicas de regularização, foram utilizadas:
* Dropout (25%): desativa neurônios aleatoriamente, evitando dependência excessiva de unidades específicas.
* L2 (λ = 0.001): penaliza pesos grandes, incentivando modelos mais simples e generalizáveis.
* Early Stopping `(patience = 15)`: interrompe o treinamento quando não há melhora na validação, restaurando automaticamente os melhores pesos.

## 6. *Pipeline* de Treinamento e Processo de Aprendizado

O treinamento começa com a verificação dos dados binários e com a inicialização aleatória dos pesos do modelo. Em seguida, o modelo é treinado ao longo de várias épocas: a cada época ele realiza o `forward pass` (gerando predições), calcula a perda com `binary_crossentropy` e atualiza os pesos por meio do algoritmo de `backpropagation`.

Após cada época, o modelo também é avaliado no conjunto de validação para medir seu desempenho em dados não vistos. O *Early Stopping* monitora a perda de validação e interrompe automaticamente o treinamento quando não há melhora por 10 épocas consecutivas, restaurando os melhores pesos obtidos. Todo o histórico de aprendizagem — incluindo `loss` e `accuracy` de treino e validação — é salvo no objeto `history`, permitindo análises e visualizações posteriores.

## Licença

Este projeto é de uso acadêmico e foi desenvolvido exclusivamente para fins educacionais no contexto da disciplina.

## Universidade

**Universidade Federal do Amazonas (UFAM)**  
**Instituto de Computação (IComp)**

*Manaus, AM -- 2025*


