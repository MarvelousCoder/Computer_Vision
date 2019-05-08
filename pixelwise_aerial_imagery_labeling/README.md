# cv-foundations-4

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
- Arquivo Araujo_Pedro__Ramos_Raphael.pdf com o relatório
- Pasta src contendo o código principal do projeto: pd4.py.

## Uso
- A partir do diretório raiz rodar com configurações padrão:
	```bash
	python ./src/pd4.py --r2
	```
- Para executar o requisito 2!
- Para executar o requisito 1 basta executar o arquivo glcm_sklearn.py normalmente
- [Repositório do github](https://github.com/peluz/cv-foundations-4)
- Requisito 1:
	- Caso queira alterar o tamanho da janela e/ou as features, basta mudar as constantes declaradas.
	- NUM_PIXELS_BUILD denota o número de valores de pixels de prédios contidos em cada uma das 83 matrizes de tamanho  171x2269 geradas.
	- NUM_PIXELS_NOT_BUILD denota o número de valores de pixels de não prédios contidos em cada uma das 87 matrizes de tamanho (227, 2167) geradas.
	- 32203917/3 é o número total de pixels de prédios na imagem de treino e 42796083/3 a quantidade dos outros pixels. A divisão por 3 se deve ao fato de que pra cada pixel da imagem temos os 3 valores de intensidade associados (rgb).
	- A função largura_altura(pixels) funciona de tal forma que ela retorna a altura A e a largura L de modo que A*L é exatamente igual a pixels (A e L vão ser inteiros) e a diferença L-A é a menor possível. 
	- Note que os algoritmos sklearn tem o mesmo "padrão" para os métodos. Logo, outros métodos podem ser testados, além do KNN escolhido.
- Requisito 2:
	- Flags de uso:
		- --imageSize tamanho das subimagens que o modelo obterá da imagem de treinamento original, pode ser 100, 200 ou 250.
		- --batchSize tamanho do batch de treinamento e avaliação.
		- --freeze não treinar camadas do extrator de características
		- --randomInit não usar pesos pretreinados
		- --train treinar o modelo, não apenas avaliar
		- --model Nome do modelo a ser salvo/carregado
	- Caso use a flag de treino, o modelo especificado será treinado
	- Após, será exibido o loss, a acurácia e o jaccard da imagem de teste
	- Por fim, será exibida a segmentação da imagem de teste, que será salva no diretório raiz.