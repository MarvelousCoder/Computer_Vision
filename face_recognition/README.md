# cv-foundations-6

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
- Pasta src contendo o código principal do projeto

## Uso
- [Repositório do github](https://github.com/peluz/cv-foundations-6)
- Requisito 1:
	- rodar no diretório raiz do projeto, para gerar o dicionário de imagens:
	```bash
	python src/build_data.py
	```
	- em seguida rodar, para treinar o modelo com diferentes hiperparâmetros e avaliá-los:
	```bash
	python src/train.py
	```
- Requisito 2:
	- rodar no diretório raiz do projeto, para extrair as embeddings:
	```bash
	python3 extract_embeddings.py --dataset turma --embeddings output/embeddings_turma.pickle --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7
	```
	A entrada é o diretório contendo as imagens, o detector de faces e o modelo de extração das embeddings.
	A saída é um serialized db das embeddings faciais, para ser usado como treino.
	- em seguida rodar, para treinar o modelo:
	```bash
	python3 train_model.py --embeddings output/embeddings_turma.pickle --recognizer output/recognizer_turma.pickle --le output/le_turma.pickle
	```
	A entrada é o output do passo anterior e a saída é o modelo de reconhecimento treinado e o label encoder.
	- Por fim, use um dos seguintes comandos para fazer reconhecimento de uma imagem não vista anteriormente. O primeiro comando detecta e reconhece um novo rosto em uma imagem dada como input e o segundo detecta e reconhece no vídeo capturado pela webcam.
	```bash
	python3 recognize.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer_turma.pickle --le output/le_turma.pickle --image images/Raphael.jpg
	```
	```bash
	python3 recognize_video.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer_turma.pickle --le output/le_turma.pickle
	``` 
	- Acesse o seguinte link para baixar o dataset lfw http://vis-www.cs.umass.edu/lfw/lfw.tgz utilizado no pipeline de detecção e reconhecimento construído neste requisito. 