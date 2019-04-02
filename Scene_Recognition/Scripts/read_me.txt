Nome: Raphael Soares Ramos
Matrícula: 14/0160299

Foi utilizado a biblioteca OpenCV, versão 3.4.2, para carregar as imagens. Para realizar operações sobre matrizes e vetores, a biblioteca de computação numérica em Python,
NumPy foi escolhida. Usei ainda, como linguagem, Python 3.5.2. Por fim, foram usadas as bibliotecas Keras e Tensorflow para o treinamento e construção de redes neurais. 
O treinamento foi realizado com o uso de um servidor Intel Xeon 24-cores com 160 GB de memória RAM e uma placa de vídeo GeForce GTX 750.

Para executar o script, apenas use um dos seguintes comandos: 
$ python3 train.py --dataset dataset --model output/modelo.model --label-bin output/modelo_lb.pickle --plot output/modelo_plot.png
$ python3 train.py -d dataset -m output/modelo.model -l output/modelo_lb.pickle -p output/modelo_plot.png

Dataset indica o diretório que contém os dados. A pasta output/ conterá os resultados do script (plots e arquivos .model e pickle).
Para alterar o modelo entre os três utilizados, basta deixar descomentado um dos grupos que pertencem as linhas 126 a 137.
É utilizado ImageDataGenerator em conjunto com fit_generator para que haja real-time augmentation durante o treinamento.
A cyclic learning rate utilizada foi obtida do seguinte repositório: https://github.com/bckenstler/CLR. Conforme citado no relatório.

Para usar o modo exp_range basta substituir a linha correspondente pela debaixo:  
clr = CyclicLR(base_lr=INIT_LR, max_lr=0.006, step_size=8*len(trainX)//BS, mode='exp_range', gamma=0.99994)
Por default o mode é triangular. Eu utilizei os três modos no projeto: triangular, triangular2 e exp_range.
A max_lr utilizada foi de 0.006 no geral, e 0.005 para alguns exp_range.

Usando o script predict.py em conjunto com os arquivos .model e pickle gerados é possível fazer novas predições para imagens não vistas (disponibilizei duas imagens que foram submetidas a classificação).
Basta executar o comando:
$ python3 predict.py -i [path_imagem] -m [path_model] -l [path_pickle] -w [largura utilizada no modelo] -e [altura utilizada no modelo]. 

As larguras e alturas utilizadas são as padrões para cada modelo e estão relatadas no arquivo results.txt, que contém os resultados e hiper-parâmetros para todos os modelos testados.

A pasta resultados contém plots para os modelos testados. Algumas "versões" na pasta resultados não contém os plots de treino (inclusive o melhor resultado para a inception V3, a Inceptionv3-4.0, nem consta no relatório) por um erro no script train.py em que eu estava esquecendo de adaptar o plot de traning loss and accuracy vs epochs caso o treino da rede fosse interrompido antes pelo callback stopper. Esse callback interrompe o treino quando uma quantidade monitorada parou de melhorar. No caso, eu usei loss como medida com uma patience de 15 ou 20 épocas. Além disso, alguns plots da curva de aprendizagem podem estar com título errado na imagem mas o certo é que tá dentro da pasta.

Os arquivos .out disponibilizados contém os logs dos scripts executados para treinar as redes.

Os arquivos .model dos modelos testados estão disponíveis no dropbox: 
https://www.dropbox.com/sh/djh2tlqu7wj36ur/AADXmzVQOJ4MD3c98wIeDuuOa?dl=0.

