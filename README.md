# Trabalho de Conclusão de Curso
No projeto, eu utilizo  **Deep Learning** para classificar quais gestos estão sendo realizados em vídeo, transcrevendo-os em texto. Isso é realizado por uma arquitetura conhecida como **rede neural convolucional**.

Inicialmente, apenas as letras do alfabeto são contempladas. O sistema construído ate agora faz a tradução em cima de vídeos gerados a partir da Webcam. O software foi construído na linguagem de programação Python e emprega a biblioteca de programação Tensorflow para fazer o treinamento e uso da rede neural profunda. 

Ao final, 97,6% de acurácia no treinamento foi obtida e 98,1% de acurácia na validação desse mesmo conjunto também foi adquirida. Esses resultados exemplificam a construção de um modelo robusto que classifica corretamente os gestos a partir de vídeos. 

Pretendo avaliar alterações na arquitetura dessa rede neural profunda e estender sua base de dados com outros gestos.

## Banco de Dados
![](imgs/base%20de%20dados.gif?raw=true)

### Aplicação da técnica de aumento de dados
![](imgs/bd.PNG?raw=true)

## Classificação  WEBCAM
![](imgs/rede.PNG?raw=true)


##  CNN Implementada
![](imgs/cnn.PNG?raw=true)

## Informações Complementares
Para rodar o projeto, crie uma pasta virtualenv e após ativa-la dê o comando "pip install -r requirements.txt" para baixar as bibliotecas nas versões utilizadas no projeto.

Disponível pra leitura aqui: [TCC.pdf](https://www.docdroid.net/q7HYakT/tcc-juliasilvacastro-1-1-pdf)
