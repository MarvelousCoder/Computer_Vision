# cv-foundations-5

## Conteúdo
 1. [Requisitos](#requisitos)
 2. [Estrutura](#estrutura)
 3. [Uso](#uso)

## Requisitos 
1.  Python 3.5.2	
2.  OpenCV 3.3.0
3.  Keras 2.2.4
4.  Matplotlib 2
5.  Scikit-learn 0.20
6.  Tensorflow 1.11

## Estrutura
- Pasta relatorio com código fonte do relatório
- Arquivo Araujo_Pedro__Sousa_Rafael.pdf com o relatório
- Pasta src contendo o código principal do projeto: pd5.py.

## Uso
- A partir do diretório raiz rodar com configurações padrão:
	```bash
	python ./src/pd5.py --[r1 ou r2 ou bonus]
	```
- [Repositório do github](https://github.com/peluz/cv-foundations-5)
- Requisito 1:

- Requisito 2:
	- Flags de uso:
		- --batchSize tamanho do batch de treinamento e avaliação.
		- --freeze não treinar camadas do extrator de características
		- --pooling para selecionar o método de pooling das características (avg ou max)
		- --train treinar o modelo, não apenas avaliar
		- --model Nome do modelo a ser salvo/carregado
		- --numUnits Número de unidades da camada fully connected
		- --dropProb Probabilidade de dropout
	- Caso use a flag de treino, o modelo especificado será treinado
	- Após, será exibido a acurácia do conjunto de teste, a precisão, recall e f1-score de cada classe, e uma matriz de confusão das classes.