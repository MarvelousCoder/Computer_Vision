Nome: Raphael Soares Ramos
Matrícula: 140160299

Conforme mencionado no relatório, este projeto foi desenvolvido foi desenvolvido em Linux 18.04 64 bits, utilizando a versão 3.2.0 do OpenCV e C++11 com compilador gcc 7.3.0.

Para executar a aplicação apenas execute make a partir do terminal. Edite o makefile conforme o requisito a ser verificado!

Use ./pd1 2 <path/imagem> para verificar os requisitos 1 e 2, conforme explicado pela função help.
Para verificar o requisito 3, COMENTE a linha 183 (VideoCapture cap(0);) e DESCOMENTE a linha 184 (VideoCapture cap(argv[2])). Depois, use ./pd1 3 <path/video>.
Para verificar o requisito 4, apenas use ./pd1 4.

O #include <../opencv2/opencv.hpp> assume que a pasta opencv2 que contém os arquivos hpp necessários para a library OpenCV, estão no diretório/pasta acima da que contém o arquivo PD1.cpp.

O arquivo em Python usado para gerar o histograma foi o flattened_histograms.py.

Repositório do git: https://github.com/AngelicCoder/Visao_Imagens.
