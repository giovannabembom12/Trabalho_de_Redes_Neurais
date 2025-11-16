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

# Avaliação do Modelo

## Introdução à Avaliação do Modelo

O objetivo da avaliação é medir o desempenho do classificador binário, baseado em uma Rede Neural Artificial, em prever corretamente a presença (1) ou ausência (0) de doença cardíaca em pacientes, utilizando métricas como acurácia, precisão, recall, F1-score e a matriz de confusão para verificar a eficácia do modelo na classificação correta dos casos. Essa etapa é fundamental para verificar a capacidade do modelo em generalizar e classificar adequadamente novos dados, garantindo sua eficácia na aplicação clínica.

No cenário médico específico do projeto, a escolha de métricas adequadas para avaliar modelos de classificação é crucial, pois falsos negativos e falsos positivos têm consequências distintas. Um falso negativo, que indica erroneamente a ausência de uma doença, pode atrasar o diagnóstico e o tratamento, colocando a vida do paciente em risco. Já um falso positivo pode gerar ansiedade desnecessária e tratamentos invasivos ou custosos. Por isso, métricas como precisão, recall e F1-score, além da matriz de confusão, são fundamentais para balancear esses erros e garantir que o modelo seja confiável e seguro para uso clínico.

## Matriz de Confusão

Matriz de erro ou matriz de confusão são uma tabela criada com o propósito de permitir uma visualização clara de um algoritmo de frequências de classificação para cada classe do modelo. Essas frequências estão divididas em: 

* **Verdadeiro positivo (true positive, TP):** O modelo previu corretamente um dado positivo
* **Falso positivo (false positive, FP):** O modelo previu incorretamente um dado positivo
* **Verdadeiro negativo (true negative, TN):** O modelo previu corretamente um dado negativo 
* **Falso negativo (false negative, FN):** O modelo previu incorretamente uma dado negativa 

Em geral, a aparência de uma matriz de confusão segue o modelo da Tabela 1.

<div align="center">
<img width="628" height="148" alt="Image" src="https://github.com/user-attachments/assets/94224318-79b5-4167-b5e1-e39d605870af" />
</div>

A tabela do projeto, seguindo o mesmo modelo, ficou assim: 

<div align="center">
<img width="703" height="544" alt="Image" src="https://github.com/user-attachments/assets/0e0dc77d-ed2a-41bd-af84-12f3172745f1" />
</div>

Conforme as frequências previamente definidas, sendo actual 0/1 equivalentes a positivo/negativo respectivamente: 

* **(TP):** O modelo previu corretamente 24 dados positivos
* **(FN):** O modelo previu incorretamente 4 dados negativos
* **(TN):** O modelo previu corretamente 28 dados negativos 
* **(FP):** O modelo previu incorretamente 5 dados positivos

## Métricas de Avaliação

Existem quatro métricas amplamente utilizadas para avaliar o desempenho de um modelo de classificação: **acurácia**, **precisão**, **recall** e **F1-score**. Todas essas métricas podem ser obtidas por meio das informações da matriz de confusão, que organiza os acertos e erros do modelo.

A **acurácia** representa a proporção de previsões corretas (positivas e negativas) em relação ao total de amostras. Ela é definida matematicamente por:

Acurácia = $$TP+TN \over TP+TN+FP+FN$$

Quanto maior a acurácia, melhor, pois indica uma maior quantidade de classificações corretas.

No caso do modelo criado, a acurácia calculada a partir da matriz de confusão apresentada é:

Acurácia = $$24+28 \over 24+28+5+4$$ = $$52 \over 61$$ = 0.85

Nem sempre a acurácia é a métrica mais apropriada, especialmente quando os dados são desbalanceados, ou seja, quando há muito mais exemplos de uma classe do que da outra. Nesses casos, métricas específicas por classe oferecem uma avaliação mais fiel.

A **precisão** mede, para cada classe, a proporção de itens classificados como pertencentes àquela classe que realmente são daquela classe. 

**Precisão da classe 0 (saudáveis):**

P<sub>0</sub> = $$TP \over TP+FP$$ = $$24 \over 24+5$$ = $$24 \over 29$$ = 0.83

**Precisão da classe 1 (doentes):**

P<sub>1</sub> = $$TN \over TN+FN$$ = $$28 \over 28+4$$ = $$28 \over 32$$ = 0.88

A precisão é especialmente útil para avaliar falsos positivos e falsos negativos, pois quanto maior ela for, menos vezes o modelo identifica incorretamente um caso como pertencente àquela classe.

O **recall** mede a proporção de exemplos de uma classe que foram corretamente identificados como tal.  

**Recall da classe 0 (saudáveis):**

R<sub>0</sub> = $$TP \over TP+FN$$ = $$24 \over 24+4$$ = $$24 \over 28$$ = 0.86

**Recall da classe 1 (doentes):**

P<sub>1</sub> = $$TN \over TN+FP$$ = $$28 \over 28+5$$ = $$28 \over 33$$ = 0.85

Em contextos médicos, um recall alto é essencial, já que falsos negativos correspondem a pacientes doentes não identificados.

O **F1-score** é a média harmônica entre precisão e recall e é útil para balancear situações em que uma é alta e a outra é baixa. A fórmula é

F1 = $$2 * precisão * recall \over precisão + recall$$

**F1-Score da classe 0 (saudáveis):**

F1<sub>0</sub> = $$2 * 0.83 * 0.86 \over 0.83 + 0.86$$ = $$1.4276 \over 1.69$$ = 0.84

**F1-Score da classe 1 (doentes):**

F1<sub>1</sub> = $$2 * 0.88 * 0.85 \over 0.88 + 0.85$$ = $$1.496 \over 1.73$$ = 0.86

## Análise dos Resultados

Como foi calculado na seção anterior, as métricas de avaliação do modelo são:

| Métricas | 0 (Saudável) | 1 (Com doença) |
| -------- | ------------ | -------------- |
| Acurácia |      0.85    |       0.85     |
| Precisão |      0.83    |       0.88     |
| Recall   |      0.86    |       0.85     |
| F1-Score |      0.84    |       0.86     |

A acurácia do modelo mostra que, aproximadamente, 80 a cada 100 pacientes são classificados corretamente. É uma acurácia boa, mas, considerando o contexto médico, o modelo não deve ser avaliado somente por esse valor, pois não diferencia os tipos de erro, como os pacientes com doença classificados como saudáveis.

A precisão da categoria de pacientes saudáveis indica que 83 a cada 100 pacientes classificados como saudáveis são realmente saudáveis. Ou seja, 17 desses 100 pacientes possuem alguma doença. Esse resultado é preocupante, pois indica que esses indivíduos podem não receber o tratamento adequado. Já a precisão da categoria de pacientes com alguma doença mostra que 88 a cada 100 pacientes classificados como doentes possuem realmente alguma doença. Apesar de estarem 12 saudáveis classificados como doentes — o que poderia gerar gastos desnecessários em tratamentos — é menos crítico que pacientes com alguma doença não receberem tratamento adequado.

O recall de saudáveis mostra que 86 a cada 100 pacientes saudáveis foram classificados corretamente. Ou seja, 14 pacientes saudáveis foram classificados como doentes, o que pode levar a tratamentos desnecessários. Já o recall de doentes mostra que 85 a cada 100 pacientes com alguma doença foram classificados corretamente. Logo, 15 foram classificados como doentes, não tendo acesso ao tratamento de sua doença. 

Por fim, os valores de F1-Score — 0.84 para pacientes saudáveis e 0.86 para pacientes com alguma doença cardíaca — indicam um equilíbrio entre precisão e recall em ambas as classes. Consequentemente, demonstram uma ótima detecção de doenças cardíacas com poucos falsos positivos.

De forma geral, as métricas revelam que o modelo tem um desempenho satisfatório, considerando o contexto de doenças cardíacas. Porém, os falsos negativos (pacientes com alguma doença classificados como saudáveis) geram oportunidade de aprimoramento do modelo futuramente, pois, na área da saúde, esse tipo de erro pode acarretar na perda de vidas. 

# Uma breve conclusão sobre a eficácia do modelo e a importância da normalização dos dados

O modelo de rede neural para a predição de doenças cardíacas demonstrou uma eficácia notável, alcançando uma acurácia de aproximadamente ~85% no conjunto de teste. Essa performance é robusta, com métricas de precisão, recall e F1-score bem balanceadas para ambas as classes (pacientes com e sem doença cardíaca), indicando uma boa capacidade de discriminar entre os dois grupos e, mais importante, de generalizar bem para novos dados não vistos durante o treinamento.

## Analisando o Relatório de Classificação:

**Classe 0 (Não possui doença cardíaca):**

**Precisão (Precision):** 0.83. Dos pacientes que o modelo previu que não tinham doença cardíaca, aproximadamente 83% realmente não tinham. Isso indica uma baixa taxa de falsos positivos para esta classe.

**Recall:** 0.86. Dos pacientes que realmente não tinham doença cardíaca, o modelo identificou corretamente 86% dos casos. Isso significa que 14% dos pacientes sem doença foram classificados erroneamente como tendo a doença (falsos negativos).

**F1-score:** 0.84. É a média harmônica da precisão e recall, oferecendo um bom equilíbrio entre as duas métricas para a classe 0.

**Classe 1 (Possui doença cardíaca):**

**Precisão (Precision):** 0.88. Dos pacientes que o modelo previu que tinham doença cardíaca, cerca de 88% realmente tinham. Isso indica uma taxa aceitável de falsos positivos para esta classe.

**Recall:** 0.85. Dos pacientes que realmente tinham doença cardíaca, o modelo identificou corretamente 85%. Isso é importante, pois significa que a maioria dos casos positivos foi detectada, o que é crucial para doenças como problemas cardíacos.

**F1-score:** 0.86. Reflete um bom balanço entre precisão e recall para a classe 1.

Em resumo, o modelo demonstra uma boa capacidade geral de prever a presença ou ausência de doença cardíaca, com um desempenho ligeiramente melhor na identificação de pacientes com a doença (maior recall para a classe 1), o que é desejável em um contexto médico para minimizar diagnósticos perdidos.

## Importância da Normalização

Contudo, essa eficácia não seria possível sem um passo de pré-processamento fundamental: a normalização dos dados. A importância da normalização reside em vários aspectos:

### Escala Consistente:
Variáveis como “idade” (geralmente entre 29-77) e “colesterol” (colesterol, que pode variar entre 126-564) possuem escalas muito diferentes. Sem normalização, o modelo tenderia a dar um peso desproporcional a características com maiores valores numéricos, independentemente de sua real importância preditiva. A normalização garante que todas as características contribuam igualmente para o cálculo do gradiente durante o treinamento, evitando que características com grandes magnitudes dominem o processo de otimização.

### Convergência Otimizada:
Ao centralizar os dados em torno de zero e escalá-los para terem um desvio padrão de um, a paisagem da função de perda se torna mais simétrica e menos alongada. Isso permite que os algoritmos de otimização encontrem o mínimo global (ou um mínimo próximo) de forma mais rápida e eficiente. Sem a normalização, o otimizador poderia "saltar" excessivamente ou ficar preso em mínimos locais.

### Regularização Aprimorada:
A normalização trabalha em sinergia com técnicas de regularização, como a regularização L2 (kernel_regularizer=regularizers.l2(0.001)) e o Dropout (Dropout(0.25)) aplicados em nosso modelo. Modelos com características normalizadas tendem a ser menos sensíveis a pequenas variações nos dados de entrada, o que contribui para uma melhor generalização e menor overfitting.

### Melhor Interpretabilidade (durante o treinamento):
 Embora as características transformadas não sejam diretamente interpretáveis para um ser humano, internamente o modelo pode aprender relações mais significativas e estáveis entre as características, uma vez que elas estão em uma escala comparável.
Em resumo, a normalização dos dados não é apenas uma boa prática, mas uma condição essencial para que as redes neurais e muitos outros algoritmos de Machine Learning operem de forma eficaz e atinjam seu potencial máximo. Ela permite que o modelo aprenda padrões verdadeiros nos dados, em vez de ser influenciado por diferenças arbitrárias de escala, resultando na capacidade de generalização e eficácia observadas neste modelo de previsão de doenças cardíacas.


## Licença

Este projeto é de uso acadêmico e foi desenvolvido exclusivamente para fins educacionais no contexto da disciplina.

## Universidade

**Universidade Federal do Amazonas (UFAM)**  
**Instituto de Computação (IComp)**

*Manaus, AM - 2025*


